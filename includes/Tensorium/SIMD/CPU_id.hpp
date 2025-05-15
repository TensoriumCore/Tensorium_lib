#pragma once 

#include <cpuid.h>
#include <cstring>
#include <string>
#include <iostream>

inline std::string get_cpu_brand() {
	char brand[0x40] = {0};
	unsigned int regs[4] = {0};
	for (int i = 0; i < 3; ++i) {
		__cpuid(0x80000002 + i, regs[0], regs[1], regs[2], regs[3]);
		std::memcpy(brand + i * 16, regs, sizeof(regs));
	}
	return std::string(brand);
}

inline size_t detect_optimal_block_size() {
	std::string brand = get_cpu_brand();

	if (brand.find("Xeon Phi") != std::string::npos) return 256;   
	if (brand.find("Xeon")     != std::string::npos) return 128;
	if (brand.find("Ryzen")    != std::string::npos) return 96;
	if (brand.find("Apple")    != std::string::npos) return 64;
	if (brand.find("Core(TM)") != std::string::npos) return 128;
	std::cout << "Unknown CPU brand. Defaulting to 64." << std::endl;
	return 64; 
}

