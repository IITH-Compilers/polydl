#ifndef CONFIG_PROCESSOR_HPP
#define CONFIG_PROCESSOR_HPP

#include<string.h>
#include<vector>
#include <unordered_map>

struct SystemConfig {
	long L1; // in bytes
	long L2; // in bytes
	long L3; // in bytes
};

typedef struct SystemConfig SystemConfig;

struct Config {
	SystemConfig *systemConfig;
	std::vector<std::unordered_map<std::string, int>*> *programParameterVector;
	int datatypeSize;
};

typedef struct Config Config;

void ReadConfig(std::string configFile, Config* config);
void FreeConfig(Config* config);
void PrintConfig(Config* config);

#endif