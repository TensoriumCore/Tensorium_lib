#include "TestRunner.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

namespace tensorium::tests {
namespace {

bool matches_filter(const std::string &name, const std::vector<std::string> &filters) {
    if (filters.empty())
        return true;
    return std::any_of(filters.begin(), filters.end(), [&](const std::string &pattern) {
        return name.rfind(pattern, 0) == 0; // prefix match
    });
}

} // namespace

int run_registered_tests(const TestRunnerOptions &opts) {
    const auto &tests = TestRegistry::instance().tests();

    std::vector<const TestCase *> selected;
    selected.reserve(tests.size());
    for (const auto &test : tests) {
        if (matches_filter(test.name, opts.filters))
            selected.push_back(&test);
    }

    if (selected.empty()) {
        std::cerr << "No tests matched the provided filters." << std::endl;
        return 1;
    }

    if (opts.list_only) {
        for (const auto *test : selected) {
            std::cout << std::left << std::setw(32) << test->name << " - " << test->description
                      << '\n';
        }
        return 0;
    }

    size_t failures = 0;
    for (const auto *test : selected) {
        if (opts.verbose)
            std::cout << "[RUN] " << test->name << " - " << test->description << std::endl;
        const auto start = std::chrono::steady_clock::now();
        try {
            test->func();
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start);
            std::cout << "[OK ] " << test->name << " (" << std::setprecision(3) << std::fixed
                      << elapsed.count() << " ms)" << std::endl;
        } catch (const std::exception &ex) {
            ++failures;
            std::cout << "[ERR] " << test->name << " :: " << ex.what() << std::endl;
        } catch (...) {
            ++failures;
            std::cout << "[ERR] " << test->name << " :: unknown exception" << std::endl;
        }
    }

    if (failures == 0) {
        std::cout << "All tests passed (" << selected.size() << ")" << std::endl;
        return 0;
    }
    std::cout << failures << " test(s) failed out of " << selected.size() << std::endl;
    return 1;
}

} // namespace tensorium::tests

