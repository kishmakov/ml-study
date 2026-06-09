#include <zmq.h>

#include "costtogo.hpp"
#include "puzzle/environment.hpp"
#include "search/a_star.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
constexpr int kCtgEvalMaxStates = 65000;
constexpr int kCtgEvalPopBatchSize = 4;

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

class WorkerState {
public:
    void load(const Update& update) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (
            heuristic_ != nullptr
            && puzzle_name_ == update.puzzle_name
            && model_stem_ == update.model_stem
            && model_version_ == update.model_version
        ) {
            return;
        }

        heuristic_ = std::make_shared<deepcube::costtogo::TorchScriptCostToGo>(
            update.model_stem + ".torchscript"
        );
        puzzle_name_ = update.puzzle_name;
        model_stem_ = update.model_stem;
        model_version_ = update.model_version;
    }

    std::shared_ptr<const deepcube::search::CostToGo> heuristicFor(const Task& task) const {
        std::lock_guard<std::mutex> lock(mutex_);
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
        return heuristic_;
    }

private:
    mutable std::mutex mutex_;
    std::string puzzle_name_;
    std::string model_stem_;
    int model_version_ = 0;
    std::shared_ptr<const deepcube::search::CostToGo> heuristic_;
};

template <typename T>
class BlockingQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            queue_.push(std::move(value));
        }
        cv_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return closed_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool closed_ = false;
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

std::string encode_model_input(const std::vector<float>& input) {
    std::ostringstream encoded;
    encoded << std::setprecision(9);
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (i != 0) {
            encoded << ',';
        }
        encoded << input[i];
    }
    return encoded.str();
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

class WorkerPool {
public:
    WorkerPool(
        std::size_t thread_count,
        BlockingQueue<Task>& tasks,
        BlockingQueue<Result>& results,
        const WorkerState& worker_state
    )
        : tasks_(tasks),
          results_(results),
          worker_state_(worker_state) {
        workers_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                worker_loop();
            });
        }
    }

    ~WorkerPool() {
        stop();
    }

    void stop() {
        if (stopped_.exchange(true)) {
            return;
        }

        tasks_.close();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    void worker_loop() {
        while (!stopped_) {
            std::optional<Task> task = tasks_.pop();
            if (!task) {
                break;
            }

            try {
                const std::shared_ptr<const deepcube::search::CostToGo> heuristic =
                    worker_state_.heuristicFor(*task);
                const std::unique_ptr<deepcube::puzzle::Environment> puzzle =
                    deepcube::puzzle::createEnvironment(task->puzzle_name, task->state);
                const std::vector<float> input = puzzle->costToGoInput();
                const std::vector<float> values = heuristic->batch({input});
                if (values.size() != 1) {
                    throw std::runtime_error("cost-to-go returned wrong result count");
                }

                deepcube::search::SearchResult search_result = deepcube::search::aStarSearch(
                    deepcube::puzzle::createEnvironment(task->puzzle_name, task->state),
                    *heuristic,
                    kCtgEvalWeight,
                    kCtgEvalMaxStates,
                    kCtgEvalPopBatchSize
                );

                results_.push(Result{
                    task->client_id,
                    task->task_id,
                    task->depth,
                    values[0],
                    search_result.solved ? 1 : 0,
                    false,
                    "",
                });
            } catch (const std::exception& error) {
                Result failed;
                failed.client_id = task->client_id;
                failed.task_id = task->task_id;
                failed.depth = task->depth;
                failed.failed = true;
                failed.error_message = error.what();
                results_.push(std::move(failed));
            }
        }
    }

    BlockingQueue<Task>& tasks_;
    BlockingQueue<Result>& results_;
    const WorkerState& worker_state_;
    std::vector<std::thread> workers_;
    std::mutex output_mutex_;
    std::atomic<bool> stopped_{false};
};

class ZmqWorker {
public:
    ZmqWorker(std::string worker_endpoint, std::size_t worker_count)
        : worker_endpoint_(std::move(worker_endpoint)),
          worker_id_(std::to_string(getpid())),
          workers_(worker_count, tasks_, results_, worker_state_) {}

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

        while (running_) {
            Result result;
            while (results_.try_pop(result)) {
                if (result.failed) {
                    send_string(socket_, encode_error(result.client_id, result.task_id, result.error_message));
                } else {
                    send_string(socket_, encode_result(result));
                }
            }

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
                try {
                    tasks_.push(decode_task(raw));
                } catch (const std::exception& error) {
                    send_string(socket_, encode_error("", "", error.what()));
                }
            } else {
                throw std::runtime_error("invalid worker command: " + raw);
            }
        }
    }

    void stop() {
        running_ = false;
        workers_.stop();
        tasks_.close();
        results_.close();

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

    BlockingQueue<Task> tasks_;
    BlockingQueue<Result> results_;
    WorkerState worker_state_;
    WorkerPool workers_;

    void* context_ = nullptr;
    void* socket_ = nullptr;
    std::atomic<bool> running_{true};
};

std::string require_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    throw std::runtime_error("missing required argument: " + name);
}

std::size_t optional_size_arg(int argc, char** argv, const std::string& name, std::size_t default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            const int parsed = std::stoi(argv[i + 1]);
            if (parsed <= 0) {
                throw std::runtime_error("argument must be positive: " + name);
            }
            return static_cast<std::size_t>(parsed);
        }
    }
    return default_value;
}

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " --worker-endpoint ENDPOINT [--threads N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string worker_endpoint = require_arg(argc, argv, "--worker-endpoint");
        const std::size_t thread_count = optional_size_arg(argc, argv, "--threads", 12);

        ZmqWorker worker(worker_endpoint, thread_count);
        worker.run();
    } catch (const std::exception& error) {
        usage(argv[0]);
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
