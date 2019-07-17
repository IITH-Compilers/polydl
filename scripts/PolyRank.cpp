#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <fstream>
using namespace std;

#define DEBUG 1

struct ProgramVariant {
	string config;
	int version;
	double gflops;
	int L1, L2, L3;
};

typedef struct ProgramVariant ProgramVariant;

/* Function declarations begin */
void OrchestrateProgramVariantsRanking(int argc, char **argv);
void FreeProgramVariants(vector<ProgramVariant*> *programVariants);
void ReadProgramVariants(string line, vector<ProgramVariant*> *programVariants);
void PrintProgramVariant(ProgramVariant *var);
/* Function declarations end */


int main(int argc, char **argv) {
	cout << "Hello from PolyRank" << endl;
	OrchestrateProgramVariantsRanking(argc, argv);
	return 0;
}

void OrchestrateProgramVariantsRanking(int argc, char **argv) {
	if (argc < 1) {
		cout << "Input file not specified." << endl;
		exit(1);
	}

	string inputFile = argv[1];
	ifstream inFile;
	inFile.open(inputFile);

	if (!inFile) {
		cout << "Unable to open the program characterization file: "
			<< inputFile << endl;
		exit(1);
	}

	vector<ProgramVariant*> *programVariants = new vector<ProgramVariant*>();

	/* Each line holds performance data on multiple variants of the program.
	Therefore, we perform rank ordering on each line of the CSV file*/
	string line;
	bool header = true;
	while (getline(inFile, line))
	{
		// We will skip the header file
		if (header) {
			header = false;
			continue;
		}

		ReadProgramVariants(line, programVariants);
		FreeProgramVariants(programVariants);
	}

	delete programVariants;


	inFile.close();
}

void ReadProgramVariants(string line, vector<ProgramVariant*> *programVariants) {
	/* The columns are assumed to be the following:
	Config	<Version	GFLOPS	L1	L2	L3> <Version	GFLOPS	L1	L2	L3> ...
	*/

	istringstream iss(line);
	string config;

	if (!(getline(iss, config, ','))) {
		cout << "Error reading the line in config file: " << line << endl;
		exit(1);
	}

	cout << "config: " << config << endl;

	string version, gflops, L1, L2, L3;
	while (getline(iss, version, ',') &&
		getline(iss, gflops, ',') &&
		getline(iss, L1, ',') &&
		getline(iss, L2, ',') &&
		getline(iss, L3, ',')) {
		ProgramVariant* var = new ProgramVariant;
		var->config = config;
		var->version = stoi(version);
		var->gflops = stod(gflops);
		var->L1 = stoi(L1);
		var->L2 = stoi(L2);
		var->L3 = stoi(L3);
		programVariants->push_back(var);

		if (DEBUG) {
			PrintProgramVariant(var);
		}
	}
}

void PrintProgramVariant(ProgramVariant *var) {
	cout << "config: " << var->config << endl;
	cout << "version: " << var->version << endl;
	cout << "gflops: " << var->gflops << endl;
	cout << "L1: " << var->L1 << endl;
	cout << "L2: " << var->L2 << endl;
	cout << "L3: " << var->L3 << endl;
}

void FreeProgramVariants(vector<ProgramVariant*> *programVariants) {
	for (int i = 0; i < programVariants->size(); i++) {
		delete programVariants->at(i);
	}

	programVariants->clear();
}