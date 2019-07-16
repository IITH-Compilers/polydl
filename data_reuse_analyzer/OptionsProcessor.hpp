#ifndef OPTIONS_PROCESSOR_HPP
#define OPTIONS_PROCESSOR_HPP

#include <string>

struct UserInput {
	std::string inputFile;
	std::string configFile;
};

typedef struct UserInput UserInput;
void ReadUserInput(int argc, char **argv, UserInput *userInput);


#endif