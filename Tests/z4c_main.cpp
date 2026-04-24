#include "framework/TestRunner.hpp"

int main() {
    tensorium::tests::TestRunnerOptions opts;
    opts.filters = {"z4c"};
    return tensorium::tests::run_registered_tests(opts);
}
