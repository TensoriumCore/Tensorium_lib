#pragma once

#include <functional>
#include <string>
#include <vector>

namespace tensorium::tests {

struct TestCase {
    std::string              name;
    std::string              description;
    std::function<void()>    func;
};

class TestRegistry {
  public:
    static TestRegistry &instance();

    void add(const std::string &name, const std::string &description,
             std::function<void()> fn);

    const std::vector<TestCase> &tests() const noexcept { return tests_; }

  private:
    std::vector<TestCase> tests_;
};

#define TENSORIUM_TEST_CONCAT_IMPL(a, b) a##b
#define TENSORIUM_TEST_CONCAT(a, b) TENSORIUM_TEST_CONCAT_IMPL(a, b)

#define REGISTER_TEST(NAME, DESC, ...)                                                                 \
    namespace {                                                                                        \
    struct TENSORIUM_TEST_CONCAT(TestRegistrator_, __LINE__) {                                         \
        TENSORIUM_TEST_CONCAT(TestRegistrator_, __LINE__)() {                                          \
            ::tensorium::tests::TestRegistry::instance().add(NAME, DESC, __VA_ARGS__);                 \
        }                                                                                              \
    };                                                                                                 \
    static TENSORIUM_TEST_CONCAT(TestRegistrator_, __LINE__)                                           \
        TENSORIUM_TEST_CONCAT(test_reg_inst_, __LINE__);                                               \
    }

} // namespace tensorium::tests
