#include "framework/TestRunner.hpp"

#include <iostream>
#include <string>
#include <vector>

using tensorium::tests::TestRunnerOptions;
using tensorium::tests::run_registered_tests;

int main(int argc, char **argv) {
    TestRunnerOptions opts;

    std::vector<std::string> filters;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: TensoriumTests [--test prefix] [--list] [--verbose]\n";
            return 0;
        }
        if (arg == "--verbose") {
            opts.verbose = true;
            continue;
        }
        if (arg == "--list") {
            opts.list_only = true;
            continue;
        }
        if (arg == "--test" && i + 1 < argc) {
            filters.emplace_back(argv[++i]);
            continue;
        }
        if (arg == "--all") {
            filters.clear();
            continue;
        }
    }

    opts.filters = filters;
    return run_registered_tests(opts);
}

