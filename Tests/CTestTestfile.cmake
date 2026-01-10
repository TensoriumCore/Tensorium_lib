# CMake generated Testfile for 
# Source directory: /Users/at0m/Desktop/Tensorium_lib/Tests
# Build directory: /Users/at0m/Desktop/Tensorium_lib/Tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[core]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTestCore")
set_tests_properties([=[core]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;152;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[grid]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTestGrid")
set_tests_properties([=[grid]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;153;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[bssn]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTestBSSN")
set_tests_properties([=[bssn]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;154;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[bssn.initial_data]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTests" "--test" "bssn.initial_data")
set_tests_properties([=[bssn.initial_data]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;155;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[bssn.constraints]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTests" "--test" "bssn.constraints")
set_tests_properties([=[bssn.constraints]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;156;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[bssn.ricci]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTests" "--test" "bssn.ricci")
set_tests_properties([=[bssn.ricci]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;157;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
add_test([=[bssn.solvers]=] "/Users/at0m/Desktop/Tensorium_lib/Tests/TensoriumTests" "--test" "bssn.solvers")
set_tests_properties([=[bssn.solvers]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;158;add_test;/Users/at0m/Desktop/Tensorium_lib/Tests/CMakeLists.txt;0;")
