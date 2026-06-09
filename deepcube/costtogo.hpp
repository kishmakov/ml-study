#pragma once

#include "search/a_star.hpp"

#include <string>
#include <vector>

#include <torch/script.h>

namespace deepcube::costtogo {

class TorchScriptCostToGo : public search::CostToGo {
public:
    explicit TorchScriptCostToGo(const std::string& model_path);

    std::vector<float> batch(
        const std::vector<std::vector<float>>& states
    ) const override;

private:
    mutable torch::jit::script::Module model_;
};

}  // namespace deepcube::costtogo
