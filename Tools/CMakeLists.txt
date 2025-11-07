
add_executable(tensorium-fdump
    tools/FunctionAnalyzer/main.cpp
    tools/FunctionAnalyzer/Analyzer.cpp
    tools/FunctionAnalyzer/Register.cpp
)

target_link_libraries(tensorium-fdump PRIVATE tensorium_core LLVM)
