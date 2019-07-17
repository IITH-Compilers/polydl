#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>
using namespace std;

#define DEBUG 1

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

#define L1Cost 32768
#define L2Cost 1048576
#define L3Cost 40370176

struct ProgramVariant {
	string config;
	int version;
	double gflops;
	int L1, L2, L3;
	int polyRank, actualRank;
	int userDefinedCost;
};

typedef struct ProgramVariant ProgramVariant;

/* Function declarations begin */
void OrchestrateProgramVariantsRanking(int argc, char **argv);
void FreeProgramVariants(vector<ProgramVariant*> *programVariants);
void ReadProgramVariants(string line, vector<ProgramVariant*> *programVariants);
void PrintProgramVariant(ProgramVariant *var);
void RankProgramVariants(vector<ProgramVariant*> *programVariants);
void InitializeRanks(vector<ProgramVariant*> *programVariants);
void PrintProgramVariants(vector<ProgramVariant*> *programVariants);
bool compareBygflops(const ProgramVariant* a, const ProgramVariant* b);
void InitializeRanks(ProgramVariant *programVariant);
void AssignActualRankBasedOnOrder(vector<ProgramVariant*> *programVariants);
void AssignPolyRanks(vector<ProgramVariant*> *programVariants);
bool PolyRankingComplete(vector<ProgramVariant*> *programVariants);
bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b);
void AssignPolyRanksBasedOnUserDefinedCost(
	vector<ProgramVariant*> *programVariants);
bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b);
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
		RankProgramVariants(programVariants);
		FreeProgramVariants(programVariants);
	}

	delete programVariants;


	inFile.close();
}

bool compareBygflops(const ProgramVariant* a, const ProgramVariant* b) {
	return a->gflops > b->gflops;
}

void RankProgramVariants(vector<ProgramVariant*> *programVariants) {
	sort(programVariants->begin(), programVariants->end(),
		compareBygflops);
	AssignActualRankBasedOnOrder(programVariants);
	AssignPolyRanks(programVariants);

	if (DEBUG) {
		cout << "________________________" << endl;
		PrintProgramVariants(programVariants);
		cout << endl;
	}
}

/* Poly ranking logic begins */
void AssignPolyRanks(vector<ProgramVariant*> *programVariants) {
	/*LOGIC to rank the program variants based on thier data reuse
	patterns in cache resides here*/

	for (int i = 0; i < programVariants->size(); i++) {
		programVariants->at(i)->userDefinedCost =
			programVariants->at(i)->L1 * L1Cost +
			programVariants->at(i)->L2 * L2Cost +
			programVariants->at(i)->L3 * L3Cost;
	}

	AssignPolyRanksBasedOnUserDefinedCost(programVariants);
}

bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b) {
	return a->userDefinedCost < b->userDefinedCost;
}

void AssignPolyRanksBasedOnUserDefinedCost(
	vector<ProgramVariant*> *programVariants) {
	sort(programVariants->begin(), programVariants->end(),
		compareByUserDefinedCost);

	if (programVariants->size() > 0) {
		int currentRank = 1;
		int currentUserDefinedCost =
			programVariants->at(0)->userDefinedCost;
		programVariants->at(0)->polyRank = currentRank;

		for (int i = 1; i < programVariants->size(); i++) {
			if (programVariants->at(i)->userDefinedCost >
				currentUserDefinedCost) {
				currentRank++;
				currentUserDefinedCost =
					programVariants->at(i)->userDefinedCost;
			}

			programVariants->at(i)->polyRank = currentRank;
		}
	}
}

/* Poly ranking logic ends */

bool PolyRankingComplete(vector<ProgramVariant*> *programVariants) {
	for (int i = 0; i < programVariants->size(); i++) {
		if (programVariants->at(i)->polyRank == -1) {
			return false;
		}
	}

	return true;
}

void AssignActualRankBasedOnOrder(vector<ProgramVariant*> *programVariants)
{
	for (int i = 0; i < programVariants->size(); i++) {
		programVariants->at(i)->actualRank = i + 1;
	}
}

void InitializeRanks(vector<ProgramVariant*> *programVariants) {
	for (int i = 0; i < programVariants->size(); i++) {
		InitializeRanks(programVariants->at(i));
	}
}


void InitializeRanks(ProgramVariant *programVariant) {
	programVariant->polyRank = -1;
	programVariant->actualRank = -1;
	programVariant->userDefinedCost = 0;
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
		InitializeRanks(var);
		programVariants->push_back(var);
	}
}

void PrintProgramVariant(ProgramVariant *var) {
	cout << "config: " << var->config << endl;
	cout << "version: " << var->version << endl;
	cout << "gflops: " << var->gflops << endl;
	cout << "L1: " << var->L1 << endl;
	cout << "L2: " << var->L2 << endl;
	cout << "L3: " << var->L3 << endl;
	cout << "polyRank: " << var->polyRank << endl;
	cout << "actualRank: " << var->actualRank << endl;
}

void PrintProgramVariants(vector<ProgramVariant*> *programVariants) {
	for (int i = 0; i < programVariants->size(); i++) {
		PrintProgramVariant(programVariants->at(i));
	}
}

void FreeProgramVariants(vector<ProgramVariant*> *programVariants) {
	for (int i = 0; i < programVariants->size(); i++) {
		delete programVariants->at(i);
	}

	programVariants->clear();
}