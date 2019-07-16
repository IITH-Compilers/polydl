#include <OptionsProcessor.hpp>
#include <stdlib.h>
#include <iostream>
using namespace std;

void ReadUserInput(int argc, char **argv, UserInput *userInput) {
	int i = 1;

	/*
	Example command line:
	./polyscientist --input conv2d.c --config conv2d_config
	*/
	string inputPrefix = "--input";
	string configPrefix = "--config";
	for (i = 1; i < argc;) {
		if (argv[i] == inputPrefix) {
			userInput->inputFile = argv[i + 1];
			i += 2;
		}
		else if (argv[i] == configPrefix) {
			userInput->configFile = argv[i + 1];
			i += 2;
		}
		else {
			printf("Unexpected command line input: %s. Exiting\n", argv[i]);
			exit(1);
		}
	}

	if (userInput->inputFile.empty()) {
		printf("Input file not specified. Exiting\n");
		exit(1);
	}
	else {
		cout << "Input file: " << userInput->inputFile << endl;
	}

	if (userInput->configFile.empty()) {
		printf("Config file not specified. Exiting\n");
		exit(1);
	}
	else {
		cout << "Config file: " << userInput->configFile << endl;
	}
}