#include "TestRegistry.hpp"

namespace tensorium::tests {

TestRegistry &TestRegistry::instance() {
    static TestRegistry registry;
    return registry;
}

void TestRegistry::add(const std::string &name, const std::string &description,
                       std::function<void()> fn) {
    tests_.push_back(TestCase{name, description, std::move(fn)});
}

} // namespace tensorium::tests

