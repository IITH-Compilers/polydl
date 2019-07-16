#include <ConfigProcessor.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;


void ReadCacheConfig(ifstream& inFile, Config* config);
void CheckIfConfigIsFullySpecified(Config* config);
void InitializeConfig(Config* config);
void ReadDataTypeConfig(ifstream& inFile, Config* config);
void ReadParams(ifstream& inFile, Config* config);
void PrintConfig(Config* config);

void ReadConfig(string configFile, Config* config) {
	const string CACHE_HEADER = "cache";
	const string DATATYPE_SIZE_HEADER = "datatype_size";
	const string PARAMS_HEADER = "params";

	/* Initialization */
	InitializeConfig(config);
	config->datatypeSize = -1;

	ifstream inFile;
	inFile.open(configFile);

	if (!inFile) {
		cout << "Unable to open config file: " << configFile << endl;
		exit(1);
	}

	string line;
	while (getline(inFile, line))
	{
		if (line == CACHE_HEADER) {
			ReadCacheConfig(inFile, config);
		}
		else if (line == DATATYPE_SIZE_HEADER) {
			ReadDataTypeConfig(inFile, config);
		}
		else if (line == PARAMS_HEADER) {
			ReadParams(inFile, config);
		}
	}

	CheckIfConfigIsFullySpecified(config);
	inFile.close();
}

void ReadCacheConfig(ifstream& inFile, Config* config) {
	string line;
	while (getline(inFile, line)) {
		if (line == "\n" || line.empty()) {
			break;
		}

		istringstream iss(line);
		string cache;
		string size;
		if (!(iss >> cache >> size)) {
			cout << "Error reading the line in config file: " << line << endl;
			exit(1);
		}

		try {
			if (cache == "L1") {
				config->systemConfig->L1 = stol(size, nullptr, 10);
			}
			else if (cache == "L2") {
				config->systemConfig->L2 = stol(size, nullptr, 10);
			}
			else if (cache == "L3") {
				config->systemConfig->L3 = stol(size, nullptr, 10);
			}
			else {
				cout << "Cache in config file not known: " << cache << endl;
				exit(1);
			}
		}
		catch (const invalid_argument) {
			cerr << "Invalid cache size while reading the config file" << endl;
			exit(1);
		}
	}
}

void ReadParams(ifstream& inFile, Config* config) {
	string line;

	vector<string> paramNames;
	if (getline(inFile, line)) {
		if (line == "\n" || line.empty()) {
		}
		else {
			istringstream iss(line);

			string paramName;
			while (iss >> paramName) {
				paramNames.push_back(paramName);
			}
		}
	}

	if (paramNames.size() > 0) {
		while (getline(inFile, line)) {
			if (line == "\n" || line.empty()) {
				break;
			}

			unordered_map<std::string, int>* params = new unordered_map<std::string, int>();
			istringstream iss(line);
			string valueStr;
			int i = 0;

			while (iss >> valueStr) {
				try {
					int value = stoi(valueStr, nullptr, 10);
					params->insert({ paramNames.at(i), value });
				}
				catch (const invalid_argument) {
					cerr << "Invalid datatype size while reading the config file" << endl;
					exit(1);
				}

				i++;
			}

			if (i != paramNames.size()) {
				cout << "Parsing: " << line << endl;
				cout << "Expected param values: " << paramNames.size() << " actual: "
					<< i << endl;
				exit(1);
			}

			config->programParameterVector->push_back(params);
		}
	}
}

void ReadDataTypeConfig(ifstream& inFile, Config* config) {
	string line;
	while (getline(inFile, line)) {
		if (line == "\n" || line.empty()) {
			break;
		}

		try {
			config->datatypeSize = stol(line, nullptr, 10);
		}
		catch (const invalid_argument) {
			cerr << "Invalid datatype size while reading the config file" << endl;
			exit(1);
		}
	}
}

void InitializeConfig(Config* config) {
	config->systemConfig = new SystemConfig;
	config->programParameterVector = new vector<unordered_map<string, int>*>();
	config->datatypeSize = 0;
	config->systemConfig->L1 = 0;
	config->systemConfig->L2 = 0;
	config->systemConfig->L3 = 0;
}

void CheckIfConfigIsFullySpecified(Config* config) {
	if (config->systemConfig->L1 == 0) {
		cout << "L1 cache size not found in config file" << endl;
		exit(1);
	}

	if (config->systemConfig->L2 == 0) {
		cout << "L2 cache size not found in config file" << endl;
		exit(1);
	}

	if (config->systemConfig->L3 == 0) {
		cout << "L3 cache size not found in config file" << endl;
		exit(1);
	}

	if (config->datatypeSize == 0) {
		cout << "Data type size not found in config file" << endl;
		exit(1);
	}
}

void PrintConfig(Config* config) {
	cout << "Datatype size: " << config->datatypeSize << endl;
	cout << "L1 cache size: " << config->systemConfig->L1 << endl;
	cout << "L2 cache size: " << config->systemConfig->L2 << endl;
	cout << "L3 cache size: " << config->systemConfig->L3 << endl;

	cout << "Program parameters:" << endl;
	for (int i = 0; i < config->programParameterVector->size(); i++) {
		unordered_map<string, int>* params = config->programParameterVector->at(i);
		for (auto it = params->begin(); it != params->end(); it++) {
			cout << it->first << ": " << it->second << " ";
		}

		cout << endl;
	}
}

void FreeConfig(Config* config) {
	delete config->systemConfig;
	for (int i = 0; i < config->programParameterVector->size(); i++) {
		delete config->programParameterVector->at(i);
	}

	delete config->programParameterVector;
	delete config;
}