#include <zmq.h>

#include "costtogo.hpp"
#include "puzzle/environment.hpp"
#include "search/a_star.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

constexpr const char* kTask = "TASK";
constexpr const char* kUpdate = "UPDATE";
constexpr const char* kUpdated = "UPDATED";
constexpr const char* kResult = "RESULT";
constexpr const char* kError = "ERROR";
constexpr const char* kReady = "READY";
constexpr const char* kStop = "STOP";

constexpr float kCtgEvalWeight = 0.1F;
constexpr int kCtgEvalPopBatchSize = 64;

struct Task {
    std::string client_id;
    std::string task_id;
    std::string puzzle_name;
    std::string model_stem;
    int model_version = 0;
    int depth = 0;
    std::string state;
};

struct Update {
    std::string puzzle_name;
    std::string model_stem;
    int model_version = 0;
};

struct Result {
    std::string client_id;
    std::string task_id;
    int depth = 0;
    double value = 0.0;
    int solved = 0;
    bool failed = false;
    std::string error_message;
};

std::string normalize_puzzle_name(const std::string& puzzle_name) {
    std::string normalized;
    normalized.reserve(puzzle_name.size());
    for (unsigned char ch : puzzle_name) {
        if (ch == '_' || ch == '-' || std::isspace(ch)) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

std::optional<std::size_t> model_input_size_for_puzzle(const std::string& puzzle_name) {
    const std::string key = normalize_puzzle_name(puzzle_name);
    if (key == "cube3") {
        return 54 * 6;
    }

    constexpr const char* prefix = "npuzzle";
    if (key.rfind(prefix, 0) == 0) {
        const std::string dim_text = key.substr(std::string(prefix).size());
        if (
            !dim_text.empty()
            && std::all_of(dim_text.begin(), dim_text.end(), [](unsigned char ch) {
                return std::isdigit(ch);
            })
        ) {
            const int dim = std::stoi(dim_text);
            return static_cast<std::size_t>(dim * dim);
        }
    }

    return std::nullopt;
}

class WorkerState {
public:
    void load(const Update& update) {
        if (
            heuristic_ != nullptr
            && puzzle_name_ == update.puzzle_name
            && model_stem_ == update.model_stem
            && model_version_ == update.model_version
        ) {
            return;
        }

        auto heuristic = std::make_shared<deepcube::costtogo::TorchScriptCostToGo>(
            update.model_stem + ".torchscript"
        );
        const std::optional<std::size_t> input_size = model_input_size_for_puzzle(update.puzzle_name);
        if (input_size) {
            const std::vector<float> warmup_input(*input_size, 0.0F);
            const std::vector<float> warmup_output = heuristic->batch({warmup_input});
            if (warmup_output.size() != 1) {
                throw std::runtime_error("cost-to-go warmup returned wrong result count");
            }
        }

        heuristic_ = std::move(heuristic);
        puzzle_name_ = update.puzzle_name;
        model_stem_ = update.model_stem;
        model_version_ = update.model_version;
    }

    const deepcube::search::CostToGo& heuristicFor(const Task& task) const {
        if (heuristic_ == nullptr) {
            throw std::runtime_error("worker model was not loaded");
        }
        if (
            puzzle_name_ != task.puzzle_name
            || model_stem_ != task.model_stem
            || model_version_ != task.model_version
        ) {
            throw std::runtime_error("worker model does not match task model");
        }
        return *heuristic_;
    }

private:
    std::string puzzle_name_;
    std::string model_stem_;
    int model_version_ = 0;
    std::shared_ptr<const deepcube::search::CostToGo> heuristic_;
};

std::vector<std::string> split(const std::string& raw, char delimiter, std::size_t max_splits) {
    std::vector<std::string> parts;
    std::size_t start = 0;

    while (parts.size() < max_splits) {
        const std::size_t end = raw.find(delimiter, start);
        if (end == std::string::npos) {
            break;
        }
        parts.push_back(raw.substr(start, end - start));
        start = end + 1;
    }

    parts.push_back(raw.substr(start));
    return parts;
}

std::string command_of(const std::string& raw) {
    const std::size_t end = raw.find('\t');
    if (end == std::string::npos) {
        return raw;
    }
    return raw.substr(0, end);
}

Task decode_task(const std::string& raw) {
    const std::vector<std::string> parts = split(raw, '\t', 7);
    if (parts.size() != 8 || parts[0] != kTask) {
        throw std::runtime_error("invalid task message: " + raw);
    }

    return Task{
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        std::stoi(parts[5]),
        std::stoi(parts[6]),
        parts[7],
    };
}

Update decode_update(const std::string& raw) {
    const std::vector<std::string> parts = split(raw, '\t', 3);
    if (parts.size() != 4 || parts[0] != kUpdate) {
        throw std::runtime_error("invalid update message: " + raw);
    }

    return Update{parts[1], parts[2], std::stoi(parts[3])};
}

std::string encode_ready(const std::string& worker_id) {
    return std::string(kReady) + "\t" + worker_id;
}

std::string encode_updated(const std::string& worker_id, int model_version) {
    return std::string(kUpdated) + "\t" + worker_id + "\t" + std::to_string(model_version);
}

std::string encode_result(const Result& result) {
    std::ostringstream encoded;
    encoded << kResult << '\t'
            << result.client_id << '\t'
            << result.task_id << '\t'
            << result.depth << '\t'
            << result.value << '\t'
            << result.solved;
    return encoded.str();
}

std::string encode_error(const std::string& client_id, const std::string& task_id, const std::string& message) {
    return std::string(kError) + "\t" + client_id + "\t" + task_id + "\t" + message;
}

std::string zmq_error(const std::string& action) {
    return action + ": " + zmq_strerror(zmq_errno());
}

std::string recv_string(void* socket) {
    zmq_msg_t message;
    if (zmq_msg_init(&message) != 0) {
        throw std::runtime_error(zmq_error("zmq_msg_init failed"));
    }

    const int received = zmq_msg_recv(&message, socket, 0);
    if (received < 0) {
        const std::string error = zmq_error("zmq_msg_recv failed");
        zmq_msg_close(&message);
        throw std::runtime_error(error);
    }

    const char* data = static_cast<const char*>(zmq_msg_data(&message));
    std::string value(data, data + zmq_msg_size(&message));
    zmq_msg_close(&message);
    return value;
}

void send_string(void* socket, const std::string& value) {
    const int sent = zmq_send(socket, value.data(), value.size(), 0);
    if (sent < 0 || static_cast<std::size_t>(sent) != value.size()) {
        throw std::runtime_error(zmq_error("zmq_send failed"));
    }
}

Result process_task(const Task& task, const WorkerState& worker_state, int max_states) {
    try {
        const deepcube::search::CostToGo& heuristic = worker_state.heuristicFor(task);
        const std::unique_ptr<deepcube::puzzle::Environment> puzzle =
            deepcube::puzzle::createEnvironment(task.puzzle_name, task.state);
        const std::vector<float> input = puzzle->costToGoInput();
        const std::vector<float> values = heuristic.batch({input});
        if (values.size() != 1) {
            throw std::runtime_error("cost-to-go returned wrong result count");
        }

        deepcube::search::SearchResult search_result = deepcube::search::aStarSearch(
            deepcube::puzzle::createEnvironment(task.puzzle_name, task.state),
            heuristic,
            kCtgEvalWeight,
            max_states,
            kCtgEvalPopBatchSize
        );

        return Result{
            task.client_id,
            task.task_id,
            task.depth,
            values[0],
            search_result.solved ? 1 : 0,
            false,
            "",
        };
    } catch (const std::exception& error) {
        Result failed;
        failed.client_id = task.client_id;
        failed.task_id = task.task_id;
        failed.depth = task.depth;
        failed.failed = true;
        failed.error_message = error.what();
        return failed;
    }
}

class ZmqWorker {
public:
    ZmqWorker(std::string worker_endpoint, int max_states)
        : worker_endpoint_(std::move(worker_endpoint)),
          worker_id_(std::to_string(getpid())),
          max_states_(max_states) {}

    ~ZmqWorker() {
        stop();
    }

    void run() {
        context_ = zmq_ctx_new();
        if (context_ == nullptr) {
            throw std::runtime_error(zmq_error("zmq_ctx_new failed"));
        }

        socket_ = zmq_socket(context_, ZMQ_DEALER);
        if (socket_ == nullptr) {
            throw std::runtime_error(zmq_error("zmq_socket failed"));
        }

        const int linger = 0;
        if (zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger)) != 0) {
            throw std::runtime_error(zmq_error("zmq_setsockopt failed"));
        }

        if (zmq_connect(socket_, worker_endpoint_.c_str()) != 0) {
            throw std::runtime_error(zmq_error("zmq_connect failed"));
        }

        send_string(socket_, encode_ready(worker_id_));

        zmq_pollitem_t poll_items[] = {
            {socket_, 0, ZMQ_POLLIN, 0},
        };

        while (true) {
            const int event_count = zmq_poll(poll_items, 1, 100);
            if (event_count < 0) {
                throw std::runtime_error(zmq_error("zmq_poll failed"));
            }

            if ((poll_items[0].revents & ZMQ_POLLIN) == 0) {
                continue;
            }

            const std::string raw = recv_string(socket_);
            const std::string command = command_of(raw);
            if (command == kStop) {
                break;
            }
            if (command == kUpdate) {
                const Update update = decode_update(raw);
                worker_state_.load(update);
                send_string(socket_, encode_updated(worker_id_, update.model_version));
            } else if (command == kTask) {
                const Task task = decode_task(raw);
                const Result result = process_task(task, worker_state_, max_states_);
                if (result.failed) {
                    send_string(socket_, encode_error(result.client_id, result.task_id, result.error_message));
                } else {
                    send_string(socket_, encode_result(result));
                }
            } else {
                throw std::runtime_error("invalid worker command: " + raw);
            }
        }
    }

    void stop() {
        if (socket_ != nullptr) {
            zmq_close(socket_);
            socket_ = nullptr;
        }
        if (context_ != nullptr) {
            zmq_ctx_term(context_);
            context_ = nullptr;
        }
    }

private:
    std::string worker_endpoint_;
    std::string worker_id_;
    int max_states_ = 0;

    WorkerState worker_state_;

    void* context_ = nullptr;
    void* socket_ = nullptr;
};

std::string require_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error("missing required argument: " + name);
}

int require_positive_int_arg(int argc, char** argv, const std::string& name) {
    const int value = std::stoi(require_arg(argc, argv, name));
    if (value <= 0) {
        throw std::runtime_error("argument must be positive: " + name);
    }
    return value;
}

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " --worker-endpoint ENDPOINT --max-states N\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string worker_endpoint = require_arg(argc, argv, "--worker-endpoint");
        const int max_states = require_positive_int_arg(argc, argv, "--max-states");

        ZmqWorker worker(worker_endpoint, max_states);
        worker.run();
    } catch (const std::exception& error) {
        usage(argv[0]);
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
