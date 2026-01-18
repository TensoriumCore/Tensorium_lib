#include "framework/TestRunner.hpp"

int main() {
    tensorium::tests::TestRunnerOptions opts;
    opts.filters = {"bssn"};
    return tensorium::tests::run_registered_tests(opts);
}

