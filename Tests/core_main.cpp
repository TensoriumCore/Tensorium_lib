#include "framework/TestRunner.hpp"

int main() {
    tensorium::tests::TestRunnerOptions opts;
    opts.filters = {"core"};
    return tensorium::tests::run_registered_tests(opts);
}

