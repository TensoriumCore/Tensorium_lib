#pragma once

#include "TestRegistry.hpp"
#include <vector>

namespace tensorium::tests {

struct TestRunnerOptions {
    bool                    list_only = false;
    bool                    verbose = false;
    std::vector<std::string> filters;
};

int run_registered_tests(const TestRunnerOptions &opts);

} // namespace tensorium::tests

