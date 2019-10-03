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
	string diagnostic = "--diagnostic";
	string minimalOutput = "--minout";
	string perarray = "--perarray";
	string parameters = "--parameters";
	string cachesizes = "--cachesizes";
	string datatypesize = "--datatypesize";

	userInput->interactive = false;
	userInput->minOutput = false;
	userInput->perarray = false;

	for (i = 1; i < argc;) {
		if (argv[i] == inputPrefix) {
			userInput->inputFile = argv[i + 1];
			i += 2;
		}
		else if (argv[i] == configPrefix) {
			userInput->configFile = argv[i + 1];
			i += 2;
		}
		else if (argv[i] == diagnostic) {
			userInput->interactive = true;
			i++;
		}
		else if (argv[i] == minimalOutput) {
			userInput->minOutput = true;
			i++;
		}
		else if (argv[i] == perarray) {
			userInput->perarray = true;
			i++;
		}
		else if (argv[i] == parameters) {
			userInput->parameters = argv[i + 1];
			i += 2;
		}
		else if (argv[i] == cachesizes) {
			userInput->cachesizes = argv[i + 1];
			i += 2;
		}
		else if (argv[i] == datatypesize) {
			userInput->datatypesize = argv[i + 1];
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

	if (userInput->configFile.empty()
		&& !userInput->interactive) {
		if (userInput->parameters.empty()
			|| userInput->cachesizes.empty() ||
			userInput->datatypesize.empty()) {
			printf("Config file not specified. Exiting\n");
			exit(1);
		}
	}
	else {
		cout << "Config file: " << userInput->configFile << endl;
	}
}