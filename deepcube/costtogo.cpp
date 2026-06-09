#include "costtogo.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/torch.h>

namespace deepcube::costtogo {
namespace {

void configure_torch_runtime() {
    static const bool configured = [] {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        return true;
    }();
    (void)configured;
}

torch::jit::script::Module load_model(const std::string& model_path) {
    configure_torch_runtime();
    return torch::jit::load(model_path, torch::kCPU);
}

}  // namespace

TorchScriptCostToGo::TorchScriptCostToGo(const std::string& model_path)
    : model_(load_model(model_path)) {
    model_.eval();
}

std::vector<float> TorchScriptCostToGo::batch(
    const std::vector<std::vector<float>>& states
) const {
    if (states.empty()) {
        return {};
    }

    const std::size_t input_size = states.front().size();
    std::vector<float> flat;
    flat.reserve(states.size() * input_size);
    for (const std::vector<float>& state : states) {
        if (state.size() != input_size) {
            throw std::runtime_error("cost-to-go batch contains mixed input sizes");
        }
        flat.insert(flat.end(), state.begin(), state.end());
    }

    torch::Tensor input = torch::from_blob(
        flat.data(),
        {static_cast<long>(states.size()), static_cast<long>(input_size)},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    torch::NoGradGuard no_grad;
    torch::Tensor output = model_.forward({input}).toTensor();

    output = output.reshape({static_cast<long>(states.size())})
        .clamp_min(0.0)
        .to(torch::kCPU)
        .contiguous();

    const float* data = output.data_ptr<float>();
    return std::vector<float>(data, data + states.size());
}

}  // namespace deepcube::costtogo
