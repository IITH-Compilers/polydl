#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>
using namespace std;

#define DEBUG 1
#define TOP_K 1

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))


/*
#define L1Cost 3
#define L2Cost 17
#define L3Cost 60
*/

/*Latency related*/
#define L1Cost 4
#define L2Cost 14
#define L3Cost 60
#define MemCost 84


/*Bandwidth related:
L1: 192 B/cycle
L2: 64 B/cycle
L3: 64 B/cycle
Mem: 2 B/cycle

Source: https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(server)
*/
#define SecondaryL1Cost (1.0/192.0)
#define SecondaryL2Cost (1.0/64.0)
#define SecondaryL3Cost (1.0/64.0)
#define SecondaryMemCost 0.0

struct ProgramVariant {
	string config;
	int version;
	double gflops;
	int L1, L2, L3, Mem;
	long L1DataSetSize;
	long L2DataSetSize;
	long L3DataSetSize;
	long MemDataSetSize;
	int polyRank, actualRank;
	double userDefinedCost;
	double secondaryCost;
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
void WriteRanksToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile);
void WritePerfToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile);
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


	string suffix = "_ranks.csv";
	ofstream outFile;
	string outputFile = inputFile + suffix;
	outFile.open(outputFile);

	if (outFile.is_open()) {
		cout << "Writing to file " << outputFile << endl;
	}
	else {
		cout << "Could not open the file: " << outputFile << endl;
		exit(1);
	}

	string suffix2 = "_top" + to_string(TOP_K) + "_perf.csv";
	ofstream outFile2;
	string outputFile2 = inputFile + suffix2;
	outFile2.open(outputFile2);

	if (outFile2.is_open()) {
		cout << "Writing to file " << outputFile2 << endl;
	}
	else {
		cout << "Could not open the file: " << outputFile2 << endl;
		exit(1);
	}

	outFile2 << "Config,Max_GFLOPS,Poly_Top_" + to_string(TOP_K)
		+ "GFLOPS" << endl;

	vector<ProgramVariant*> *programVariants =
		new vector<ProgramVariant*>();

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
		WriteRanksToFile(programVariants, outFile);
		WritePerfToFile(programVariants, outFile2);
		FreeProgramVariants(programVariants);
	}

	delete programVariants;

	inFile.close();
	outFile.close();
}

void WriteRanksToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile) {
	if (programVariants->size() >= 0) {
		outFile << programVariants->at(0)->config << endl;
	}

	outFile << "ActualRank,PolyRank,GFLOPS,Version" << endl;
	sort(programVariants->begin(), programVariants->end(),
		compareBygflops);
	for (int i = 0; i < programVariants->size(); i++) {
		outFile << programVariants->at(i)->actualRank << ","
			<< programVariants->at(i)->polyRank << ","
			<< programVariants->at(i)->gflops << ","
			<< programVariants->at(i)->version << endl;
	}

	outFile << endl;
}

void WritePerfToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile) {
	double maxGflops = 0;
	double maxPolyKFlops = 0;

	for (int i = 0; i < programVariants->size(); i++) {
		maxGflops = max(maxGflops, programVariants->at(i)->gflops);

		if (programVariants->at(i)->polyRank <= TOP_K) {
			maxPolyKFlops = max(maxPolyKFlops,
				programVariants->at(i)->gflops);
		}
	}

	if (programVariants->size() >= 0) {
		outFile << programVariants->at(0)->config << ","
			<< maxGflops << "," << maxPolyKFlops << endl;
	}
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

	/*TODO: Assign weights to reuses. Idea: Take the intersection of
source and target iteration data sets and compute its cardinality.
The cardinality can be the weight of the reuse*/

	for (int i = 0; i < programVariants->size(); i++) {
		double totalReuses = programVariants->at(i)->L1 +
			programVariants->at(i)->L2 +
			programVariants->at(i)->L3 +
			programVariants->at(i)->Mem;

		double totalDataSetSize =
			programVariants->at(i)->L1DataSetSize +
			programVariants->at(i)->L2DataSetSize +
			programVariants->at(i)->L3DataSetSize +
			programVariants->at(i)->MemDataSetSize;

		programVariants->at(i)->userDefinedCost =
			(programVariants->at(i)->L1 / totalReuses) * L1Cost +
			(programVariants->at(i)->L2 / totalReuses) * L2Cost +
			(programVariants->at(i)->L3 / totalReuses) * L3Cost +
			(programVariants->at(i)->Mem / totalReuses) * MemCost;

		programVariants->at(i)->secondaryCost =
			(programVariants->at(i)->L1DataSetSize)
			* SecondaryL1Cost +
			(programVariants->at(i)->L2DataSetSize)
			* SecondaryL2Cost +
			(programVariants->at(i)->L3DataSetSize)
			* SecondaryL3Cost +
			(programVariants->at(i)->MemDataSetSize)
			* SecondaryMemCost;
	}

	AssignPolyRanksBasedOnUserDefinedCost(programVariants);
}

bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b) {

	/*We use the following statistics in the order shown as the criteria
	for ordering two program variants. The lower the value of a given metric, the better the variant is considered:
	1. userDefinedCost
	2. MemDataSetSize
	3. L3DataSetSize
	4. L2DataSetSize
	5. L1DataSetSize */

	if (a->userDefinedCost == b->userDefinedCost) {
		return a->secondaryCost < b->secondaryCost;
	}
	else {
		return a->userDefinedCost < b->userDefinedCost;
	}
}

void AssignPolyRanksBasedOnUserDefinedCost(
	vector<ProgramVariant*> *programVariants) {
	sort(programVariants->begin(), programVariants->end(),
		compareByUserDefinedCost);

	if (programVariants->size() > 0) {
		int currentRank = 1;
		double currentUserDefinedCost =
			programVariants->at(0)->userDefinedCost;
		programVariants->at(0)->polyRank = currentRank;

		for (int i = 1; i < programVariants->size(); i++) {
			if (programVariants->at(i)->userDefinedCost >
				currentUserDefinedCost) {
				currentRank++;
				currentUserDefinedCost =
					programVariants->at(i)->userDefinedCost;
			}

			programVariants->at(i)->polyRank = i + 1;
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
	Config
	<Version	GFLOPS	L1	L2	L3 Mem L1DataSetSize	L2DataSetSize	L3DataSetSize	MemDataSetSize>
	<Version	GFLOPS	L1	L2	L3 Mem L1DataSetSize	L2DataSetSize	L3DataSetSize	MemDataSetSize> ...
	*/

	istringstream iss(line);
	string config;

	if (!(getline(iss, config, ','))) {
		cout << "Error reading the line in config file: " << line << endl;
		exit(1);
	}

	cout << "config: " << config << endl;

	string version, gflops, L1, L2, L3, Mem;
	string L1DataSetSize, L2DataSetSize, L3DataSetSize, MemDataSetSize;
	while (getline(iss, version, ',') &&
		getline(iss, gflops, ',') &&
		getline(iss, L1, ',') &&
		getline(iss, L2, ',') &&
		getline(iss, L3, ',') &&
		getline(iss, Mem, ',') &&
		getline(iss, L1DataSetSize, ',') &&
		getline(iss, L2DataSetSize, ',') &&
		getline(iss, L3DataSetSize, ',') &&
		getline(iss, MemDataSetSize, ',')) {
		ProgramVariant* var = new ProgramVariant;
		var->config = config;
		var->version = stoi(version);
		var->gflops = stod(gflops);
		var->L1 = stoi(L1);
		var->L2 = stoi(L2);
		var->L3 = stoi(L3);
		var->Mem = stoi(Mem);
		var->L1DataSetSize = stol(L1DataSetSize);
		var->L2DataSetSize = stol(L2DataSetSize);
		var->L3DataSetSize = stol(L3DataSetSize);
		var->MemDataSetSize = stol(MemDataSetSize);
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
	cout << "Mem: " << var->Mem << endl;
	cout << "L1DataSetSize: " << var->L1DataSetSize << endl;
	cout << "L2DataSetSize: " << var->L2DataSetSize << endl;
	cout << "L3DataSetSize: " << var->L3DataSetSize << endl;
	cout << "MemDataSetSize: " << var->MemDataSetSize << endl;
	cout << "userDefinedCost: " << var->userDefinedCost << endl;
	cout << "secondaryCost: " << var->secondaryCost << endl;
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