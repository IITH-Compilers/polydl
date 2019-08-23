#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>
using namespace std;

#define DEBUG 0
#define TOP_K 1
#define TOP_PERCENT 0.05

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))


#define DATASETSIZETHRESHOLD 0.05
#define TOTALDATASETSIZETHRESHOLD 0.5
/*Latency related*/
#define L1Cost 4
#define L2Cost 14 // 26
#define L3Cost 60
#define MemCost 84


/*Bandwidth related:
L1: 192 B/cycle : R/W together
L2: 64 B/cycle : R/W together: 96 B/cycle
L3: 8 B/cycle : R/W together: 16 B/cycle
Mem: 4 B/cycle

*/
#define SecondaryL1Cost (1.0/192.0)
#define SecondaryL2Cost (1.0/96.0)
#define SecondaryL3Cost (1.0/16.0)
#define SecondaryMemCost (1.0/4.0)


struct ProgramVariant {
	string config;
	string version;
	double gflops;
	int L1, L2, L3, Mem;
	long L1DataSetSize;
	long L2DataSetSize;
	long L3DataSetSize;
	long MemDataSetSize;
	long TotalDataSetSize;
	long PessiL1DataSetSize;
	long PessiL2DataSetSize;
	long PessiL3DataSetSize;
	long PessiMemDataSetSize;
	long PessiTotalDataSetSize;
	int polyRank, actualRank;
	double userDefinedCost;
	double secondaryCost;
	int wins;
};

typedef struct ProgramVariant ProgramVariant;

struct UserOptions {
	bool headers;
	bool perfseparaterow;
	/* false. For a single row holding the performance of all variants
	   true. The performance of different variants beings in different rows*/

	bool decisiontree;
	bool usepessidata;
	bool computeattributeimportance;
	bool lo_to_hi_decisiontree;
};

typedef struct UserOptions UserOptions;

/* Function declarations begin */
void OrchestrateProgramVariantsRanking(int argc, char **argv);
void FreeProgramVariants(vector<ProgramVariant*> *programVariants);
void ReadProgramVariants(string line, vector<ProgramVariant*> *programVariants, UserOptions* userOptions);
void PrintProgramVariant(ProgramVariant *var);
void RankProgramVariants(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions);
void InitializeRanks(vector<ProgramVariant*> *programVariants);
void PrintProgramVariants(vector<ProgramVariant*> *programVariants);
bool compareBygflops(const ProgramVariant* a, const ProgramVariant* b);
void InitializeRanks(ProgramVariant *programVariant);
void AssignActualRankBasedOnOrder(vector<ProgramVariant*> *programVariants);
void AssignPolyRanks(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions);
bool PolyRankingComplete(vector<ProgramVariant*> *programVariants);
bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b);
void AssignPolyRanksBasedOnUserDefinedCost(
	vector<ProgramVariant*> *programVariants);
bool compareByUserDefinedCost(const ProgramVariant* a,
	const ProgramVariant* b);
void WriteRanksToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile, UserOptions *userOptions);
void WritePerfToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile, UserOptions* userOptions);
UserOptions* ProcessInputArguments(int argc, char **argv);
void RankProgramVariantsAndWriteResults(
	string inputFile,
	vector<ProgramVariant*> *programVariants,
	ofstream& outFile, ofstream& outFile2, UserOptions* userOptions);
void RankUsingDecisionTree(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions);
bool compareByWins(const ProgramVariant* a, const ProgramVariant* b);
void AssignPolyRankBasedOnOrder(vector<ProgramVariant*> *programVariants);
int FindWinner(ProgramVariant *a, ProgramVariant* b,
	UserOptions* userOptions);
bool ExceedsByAThreshold(long size1, long size2, double threshold = DATASETSIZETHRESHOLD);
void ComputeAttributeImportanceFromHigherToLower(string inputFile,
	vector<ProgramVariant*> *programVariants);
long GetSizeAtIndex(ProgramVariant* var, int index);
string GetNameAtIndex(int index);
void RankUsingLoToHiDecisionTree(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions);
int FindWinnerLoToHi(ProgramVariant *a, ProgramVariant* b,
	UserOptions* userOptions);
/* Function declarations end */


int main(int argc, char **argv) {
	cout << "Hello from PolyRank" << endl;
	OrchestrateProgramVariantsRanking(argc, argv);
	return 0;
}

UserOptions* ProcessInputArguments(int argc, char **argv) {
	string arg;
	string NO_HEADER = "--noheader";
	string PERF_SEPARATE_ROW = "--perfseparaterow";
	string DECISION_TREE = "--decisiontree";
	string USE_PESSI_DATA = "--usepessidata";
	string COMPUTE_ATTRIBUTE_IMPORTANCE = "--computeattributeimportance";
	string LO_TO_HI_DECISION_TREE = "--lo_to_hi_decisiontree";
	UserOptions* userOptions = new UserOptions;
	userOptions->headers = true;
	userOptions->perfseparaterow = false;
	userOptions->decisiontree = false;
	userOptions->usepessidata = false;
	userOptions->computeattributeimportance = false;
	userOptions->lo_to_hi_decisiontree = false;

	for (int i = 2; i < argc; i++) {
		arg = argv[i];

		if (argv[i] == NO_HEADER) {
			userOptions->headers = false;
		}

		if (argv[i] == PERF_SEPARATE_ROW) {
			userOptions->perfseparaterow = true;
		}

		if (argv[i] == DECISION_TREE) {
			userOptions->decisiontree = true;
		}

		if (argv[i] == USE_PESSI_DATA) {
			userOptions->usepessidata = true;
		}

		if (argv[i] == COMPUTE_ATTRIBUTE_IMPORTANCE) {
			userOptions->computeattributeimportance = true;
		}

		if (argv[i] == LO_TO_HI_DECISION_TREE) {
			userOptions->lo_to_hi_decisiontree = true;
		}
	}

	return userOptions;
}

void OrchestrateProgramVariantsRanking(int argc, char **argv) {
	if (argc < 2) {
		cout << "Input file not specified." << endl;
		exit(1);
	}

	string inputFile = argv[1];
	UserOptions* userOptions = ProcessInputArguments(argc, argv);

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

	if (userOptions->perfseparaterow == false) {
		outFile2 << "Config,";
	}

	outFile2 << "Max_GFLOPS, Poly_Top_" + to_string(TOP_K)
		+ "GFLOPS,numVariants,Poly_Top_" + to_string(TOP_PERCENT)
		+ ",Min_GFLOPS, Median_GFLOPS" << endl;

	vector<ProgramVariant*> *programVariants =
		new vector<ProgramVariant*>();

	/* Each line holds performance data on multiple variants of the program.
	Therefore, we perform rank ordering on each line of the CSV file*/
	string line;
	bool header = userOptions->headers;


	while (getline(inFile, line))
	{
		// We will skip the header file
		if (header) {
			header = false;
			continue;
		}

		ReadProgramVariants(line, programVariants, userOptions);

		if (userOptions->perfseparaterow == false) {
			RankProgramVariantsAndWriteResults(inputFile, programVariants,
				outFile, outFile2, userOptions);
		}
	}

	if (userOptions->perfseparaterow == true) {
		RankProgramVariantsAndWriteResults(inputFile, programVariants,
			outFile, outFile2, userOptions);
	}

	delete programVariants;
	delete userOptions;

	inFile.close();
	outFile.close();
	outFile2.close();
}

void WriteRanksToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile, UserOptions *userOptions) {
	if (programVariants->size() >= 0 &&
		userOptions->perfseparaterow == false) {
		outFile << programVariants->at(0)->config << endl;
	}

	outFile << "ActualRank,PolyRank,GFLOPS,Version,wins" << endl;
	sort(programVariants->begin(), programVariants->end(),
		compareBygflops);
	for (int i = 0; i < programVariants->size(); i++) {
		outFile << programVariants->at(i)->actualRank << ","
			<< programVariants->at(i)->polyRank << ","
			<< programVariants->at(i)->gflops << ","
			<< programVariants->at(i)->version << ","
			<< programVariants->at(i)->wins << endl;
	}

	outFile << endl;
}

void RankProgramVariantsAndWriteResults(
	string inputFile,
	vector<ProgramVariant*> *programVariants,
	ofstream& outFile, ofstream& outFile2, UserOptions* userOptions) {

	if (userOptions->computeattributeimportance) {
		ComputeAttributeImportanceFromHigherToLower(inputFile, programVariants);
	}

	RankProgramVariants(programVariants, userOptions);
	WriteRanksToFile(programVariants, outFile, userOptions);
	WritePerfToFile(programVariants, outFile2, userOptions);
	FreeProgramVariants(programVariants);
}

void WritePerfToFile(vector<ProgramVariant*> *programVariants,
	ofstream& outFile, UserOptions* userOptions) {
	double maxGflops = 0;
	double maxPolyKFlops = 0;
	double maxPolyTopPercentFlops = 0;
	int numVariants = programVariants->size();
	int maxPercentRank = max(TOP_PERCENT * numVariants, 1);
	double minGflops = 0;
	double medianGflops = 0;
	int medianIndex = programVariants->size() / 2;

	for (int i = 0; i < programVariants->size(); i++) {
		maxGflops = max(maxGflops, programVariants->at(i)->gflops);

		if (programVariants->at(i)->polyRank <= TOP_K) {
			maxPolyKFlops = max(maxPolyKFlops,
				programVariants->at(i)->gflops);
		}

		if (programVariants->at(i)->polyRank <= maxPercentRank) {
			maxPolyTopPercentFlops = max(maxPolyTopPercentFlops,
				programVariants->at(i)->gflops);
		}

		if (i == 0) {
			minGflops = programVariants->at(i)->gflops;
		}
		else {
			minGflops = min(minGflops, programVariants->at(i)->gflops);
		}
	}

	if (programVariants->size() >= 0) {
		medianGflops = programVariants->at(medianIndex)->gflops;

		if (userOptions->perfseparaterow == false) {
			outFile << programVariants->at(0)->config << ",";
		}

		outFile << maxGflops << "," << maxPolyKFlops << ","
			<< numVariants << "," << maxPolyTopPercentFlops << ","
			<< minGflops << "," << medianGflops << endl;

	}
}

bool compareBygflops(const ProgramVariant* a, const ProgramVariant* b) {
	return a->gflops > b->gflops;
}

void RankProgramVariants(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions) {
	sort(programVariants->begin(), programVariants->end(),
		compareBygflops);
	AssignActualRankBasedOnOrder(programVariants);
	AssignPolyRanks(programVariants, userOptions);

	if (DEBUG) {
		cout << "________________________" << endl;
		PrintProgramVariants(programVariants);
		cout << endl;
	}
}

/* Poly ranking logic begins */
void AssignPolyRanks(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions) {
	/*LOGIC to rank the program variants based on thier data reuse
	patterns in cache resides here*/

	/*TODO: Assign weights to reuses. Idea: Take the intersection of
source and target iteration data sets and compute its cardinality.
The cardinality can be the weight of the reuse*/

	if (userOptions->decisiontree) {
		RankUsingDecisionTree(programVariants, userOptions);
		return;
	}

	if (userOptions->lo_to_hi_decisiontree) {
		RankUsingLoToHiDecisionTree(programVariants, userOptions);
		return;
	}

	for (int i = 0; i < programVariants->size(); i++) {
		double totalReuses = programVariants->at(i)->L1 +
			programVariants->at(i)->L2 +
			programVariants->at(i)->L3 +
			programVariants->at(i)->Mem;

		if (userOptions->usepessidata == false) {
			programVariants->at(i)->userDefinedCost =
				(programVariants->at(i)->L1DataSetSize) * L1Cost +
				(programVariants->at(i)->L2DataSetSize) * L2Cost +
				(programVariants->at(i)->L3DataSetSize) * L3Cost +
				(programVariants->at(i)->MemDataSetSize) * MemCost;


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
		else {
			programVariants->at(i)->userDefinedCost =
				(programVariants->at(i)->PessiL1DataSetSize) * L1Cost +
				(programVariants->at(i)->PessiL2DataSetSize) * L2Cost +
				(programVariants->at(i)->PessiL3DataSetSize) * L3Cost +
				(programVariants->at(i)->PessiMemDataSetSize) * MemCost;


			programVariants->at(i)->secondaryCost =
				(programVariants->at(i)->PessiL1DataSetSize)
				* SecondaryL1Cost +
				(programVariants->at(i)->PessiL2DataSetSize)
				* SecondaryL2Cost +
				(programVariants->at(i)->PessiL3DataSetSize)
				* SecondaryL3Cost +
				(programVariants->at(i)->PessiMemDataSetSize)
				* SecondaryMemCost;
		}
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
	programVariant->wins = 0;
}

void ReadProgramVariants(string line, vector<ProgramVariant*> *programVariants, UserOptions* userOptions) {
	/* The columns are assumed to be the following:
	Config
	<Version	GFLOPS	L1	L2	L3 Mem L1DataSetSize	L2DataSetSize	L3DataSetSize	MemDataSetSize>
	<Version	GFLOPS	L1	L2	L3 Mem L1DataSetSize	L2DataSetSize	L3DataSetSize	MemDataSetSize> ...
	*/

	istringstream iss(line);
	string config;

	if (userOptions->perfseparaterow == false) {
		if (!(getline(iss, config, ','))) {
			cout << "Error reading the line in config file: " << line << endl;
			exit(1);
		}

		cout << "config: " << config << endl;
	}

	string version, gflops, L1, L2, L3, Mem;
	string L1DataSetSize, L2DataSetSize, L3DataSetSize, MemDataSetSize;
	string PessiL1DataSetSize, PessiL2DataSetSize, PessiL3DataSetSize, PessiMemDataSetSize;
	while (getline(iss, version, ',') &&
		getline(iss, gflops, ',') &&
		getline(iss, L1, ',') &&
		getline(iss, L2, ',') &&
		getline(iss, L3, ',') &&
		getline(iss, Mem, ',') &&
		getline(iss, L1DataSetSize, ',') &&
		getline(iss, L2DataSetSize, ',') &&
		getline(iss, L3DataSetSize, ',') &&
		getline(iss, MemDataSetSize, ',') &&
		getline(iss, PessiL1DataSetSize, ',') &&
		getline(iss, PessiL2DataSetSize, ',') &&
		getline(iss, PessiL3DataSetSize, ',') &&
		getline(iss, PessiMemDataSetSize, ',')
		) {
		ProgramVariant* var = new ProgramVariant;
		var->config = config;
		var->version = version;

		try {
			var->gflops = stod(gflops);
			var->L1 = stoi(L1);
			var->L2 = stoi(L2);
			var->L3 = stoi(L3);
			var->Mem = stoi(Mem);
			var->L1DataSetSize = stol(L1DataSetSize);
			var->L2DataSetSize = stol(L2DataSetSize);
			var->L3DataSetSize = stol(L3DataSetSize);
			var->MemDataSetSize = stol(MemDataSetSize);
			var->TotalDataSetSize = var->L1DataSetSize + var->L2DataSetSize
				+ var->L3DataSetSize + var->MemDataSetSize;

			var->PessiL1DataSetSize = stol(PessiL1DataSetSize);
			var->PessiL2DataSetSize = stol(PessiL2DataSetSize);
			var->PessiL3DataSetSize = stol(PessiL3DataSetSize);
			var->PessiMemDataSetSize = stol(PessiMemDataSetSize);
			var->PessiTotalDataSetSize = var->PessiL1DataSetSize +
				var->PessiL2DataSetSize + var->PessiL3DataSetSize
				+ var->PessiMemDataSetSize;

			InitializeRanks(var);
			programVariants->push_back(var);
		}
		catch (const invalid_argument) {
			cerr << "Error parsing the line: " << line << endl;
			delete var;
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

void RankUsingDecisionTree(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions) {
	int winner; // 0: first, 1: second
	for (int i = 0; i < programVariants->size(); i++) {
		for (int j = i + 1; j < programVariants->size(); j++) {
			if (i != j) {
				winner = FindWinner(programVariants->at(i),
					programVariants->at(j), userOptions);
				if (winner == 0) {
					programVariants->at(i)->wins += 1;
				}
				else if (winner == 1) {
					programVariants->at(j)->wins += 1;
				}
			}
		}
	}

	sort(programVariants->begin(), programVariants->end(),
		compareByWins);
	AssignPolyRankBasedOnOrder(programVariants);
}

void RankUsingLoToHiDecisionTree(vector<ProgramVariant*> *programVariants,
	UserOptions* userOptions) {
	int winner; // 0: first, 1: second
	for (int i = 0; i < programVariants->size(); i++) {
		for (int j = i + 1; j < programVariants->size(); j++) {
			if (i != j) {
				winner = FindWinnerLoToHi(programVariants->at(i),
					programVariants->at(j), userOptions);
				if (winner == 0) {
					programVariants->at(i)->wins += 1;
				}
				else if (winner == 1) {
					programVariants->at(j)->wins += 1;
				}
			}
		}
	}

	sort(programVariants->begin(), programVariants->end(),
		compareByWins);
	AssignPolyRankBasedOnOrder(programVariants);
}

void ComputeAttributeImportanceFromHigherToLower(string inputFile, vector<ProgramVariant*> *programVariants) {
	string suffix = "_attr_importance_hi_to_lo.csv";
	ofstream outFile;
	string outputFile = inputFile + suffix;
	outFile.open(outputFile);

	outFile <<
		"Attribute,Accuracy,TotalPairs,Positive,PosPctDiff,"
		<< "Negative,NegativePctDiff" << endl;
	for (int index = 10; index >= 1; index--) {
		// Find out the importance of TotalDataSetSize
		long size1, size2;
		double gflops1, gflops2;
		double posPctDiff = 0, negPctDiff = 0;
		int pos = 0, neg = 0;
		int total = 0;
		double accuracy = 0;
		for (int i = 0; i < programVariants->size(); i++) {
			for (int j = i + 1; j < programVariants->size(); j++) {
				total++;
				size1 = GetSizeAtIndex(programVariants->at(i), index);
				size2 = GetSizeAtIndex(programVariants->at(j), index);

				if (size1 > 0 && size2 > 0) {
					gflops1 = programVariants->at(i)->gflops;
					gflops2 = programVariants->at(j)->gflops;

					if ((size1 < size2) && (gflops1 > gflops2)) {
						// Positive case
						pos++;
						posPctDiff += ((double)(size2 - size1)) / ((double)size1);
					}
					else if ((size1 > size2) && (gflops1 < gflops2)) {
						// Positive case
						pos++;
						posPctDiff += ((double)(size1 - size2)) / ((double)size2);
					}
					else if ((size1 < size2) && (gflops1 < gflops2)) {
						// Negative case
						neg++;
						negPctDiff += ((double)(size2 - size1)) / ((double)size1);
					}
					else if ((size1 > size2) && (gflops1 > gflops2)) {
						// Negative case
						neg++;
						negPctDiff += ((double)(size1 - size2)) / ((double)size2);
					}

					if (DEBUG) {
						if (pos == 1) {
							cout << "Positive case" << endl;
							cout << "size1: " << size1 << " size2: " << size2
								<< " gflops1: " << gflops1 << "gflops2: " << gflops2
								<< " posPctDiff: " << posPctDiff << endl;
						}

						if (neg == 1) {
							cout << "Negative case" << endl;
							cout << "size1: " << size1 << " size2: " << size2
								<< " gflops1: " << gflops1 << "gflops2: " << gflops2
								<< " negPctDiff: " << negPctDiff << endl;
						}
					}
				}
			}
		}

		if (pos > 0) {
			posPctDiff = (posPctDiff / (double)pos) * 100.0;
		}

		if (neg > 0) {
			negPctDiff = (negPctDiff / (double)neg) * 100.0;
		}

		if (pos > 0 || neg > 0) {
			accuracy = ((double)(pos)) / (((double)(pos)) + ((double)(neg)));
		}

		outFile << GetNameAtIndex(index) << "," << accuracy << ","
			<< total << "," << pos
			<< "," << posPctDiff << "," << neg << "," << negPctDiff << endl;
	}

	outFile.close();
}


long GetSizeAtIndex(ProgramVariant* var, int index) {
	if (index == 1) {
		return var->L1DataSetSize;
	}
	else if (index == 2) {
		return var->L2DataSetSize;
	}
	else if (index == 3) {
		return var->L3DataSetSize;
	}
	else if (index == 4) {
		return var->MemDataSetSize;
	}
	else if (index == 5) {
		return var->TotalDataSetSize;
	}
	else if (index == 6) {
		return var->PessiL1DataSetSize;
	}
	else if (index == 7) {
		return var->PessiL2DataSetSize;
	}
	else if (index == 8) {
		return var->PessiL3DataSetSize;
	}
	else if (index == 9) {
		return var->PessiMemDataSetSize;
	}
	else if (index == 10) {
		return var->PessiTotalDataSetSize;
	}
	else {
		cout << "Wrong index: " << index << endl;
		exit(1);
		return 0;
	}
}

string GetNameAtIndex(int index) {
	if (index == 1) {
		return "L1DataSetSize";
	}
	else if (index == 2) {
		return "L2DataSetSize";
	}
	else if (index == 3) {
		return "L3DataSetSize";
	}
	else if (index == 4) {
		return "MemDataSetSize";
	}
	else if (index == 5) {
		return "TotalDataSetSize";
	}
	else if (index == 6) {
		return "PessiL1DataSetSize";
	}
	else if (index == 7) {
		return "PessiL2DataSetSize";
	}
	else if (index == 8) {
		return "PessiL3DataSetSize";
	}
	else if (index == 9) {
		return "PessiMemDataSetSize";
	}
	else if (index == 10) {
		return "PessiTotalDataSetSize";
	}
	else {
		cout << "Wrong index: " << index << endl;
		exit(1);
		return 0;
	}
}

int FindWinnerLoToHi(ProgramVariant *a, ProgramVariant* b,
	UserOptions* userOptions) {
	int winner = -1;

	long aL1DataSetSize, aL2DataSetSize, aL3DataSetSize, aMemDataSetSize,
		aTotalDataSetSize;
	long bL1DataSetSize, bL2DataSetSize, bL3DataSetSize, bMemDataSetSize,
		bTotalDataSetSize;

	if (userOptions->usepessidata == false) {
		aL1DataSetSize = a->L1DataSetSize;
		aL2DataSetSize = a->L2DataSetSize;
		aL3DataSetSize = a->L3DataSetSize;
		aMemDataSetSize = a->MemDataSetSize;
		aTotalDataSetSize = a->TotalDataSetSize;
		bL1DataSetSize = b->L1DataSetSize;
		bL2DataSetSize = b->L2DataSetSize;
		bL3DataSetSize = b->L3DataSetSize;
		bMemDataSetSize = b->MemDataSetSize;
		bTotalDataSetSize = b->TotalDataSetSize;
	}
	else {
		aL1DataSetSize = a->PessiL1DataSetSize;
		aL2DataSetSize = a->PessiL2DataSetSize;
		aL3DataSetSize = a->PessiL3DataSetSize;
		aMemDataSetSize = a->PessiMemDataSetSize;
		aTotalDataSetSize = a->PessiTotalDataSetSize;
		bL1DataSetSize = b->PessiL1DataSetSize;
		bL2DataSetSize = b->PessiL2DataSetSize;
		bL3DataSetSize = b->PessiL3DataSetSize;
		bMemDataSetSize = b->PessiMemDataSetSize;
		bTotalDataSetSize = b->PessiTotalDataSetSize;
	}

	if (ExceedsByAThreshold(aL1DataSetSize, bL1DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L1DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL1DataSetSize, aL1DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L1DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aL2DataSetSize, bL2DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L2DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL2DataSetSize, aL2DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L2DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aL3DataSetSize, bL3DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L3DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL3DataSetSize, aL3DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L3DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aMemDataSetSize, bMemDataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "MemDataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bMemDataSetSize, aMemDataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "MemDataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aTotalDataSetSize,
		bTotalDataSetSize,
		TOTALDATASETSIZETHRESHOLD)) {
		winner = 0;
		if (DEBUG)
			cout << "TotalDataSetSize_Winner" << endl;

	}
	else if (ExceedsByAThreshold(bTotalDataSetSize,
		aTotalDataSetSize,
		TOTALDATASETSIZETHRESHOLD)) {
		winner = 1;
		if (DEBUG)
			cout << "TotalDataSetSize_Winner" << endl;
	}

	return winner;
}

int FindWinner(ProgramVariant *a, ProgramVariant* b,
	UserOptions* userOptions) {
	int winner = 0;

	long aL1DataSetSize, aL2DataSetSize, aL3DataSetSize, aMemDataSetSize,
		aTotalDataSetSize;
	long bL1DataSetSize, bL2DataSetSize, bL3DataSetSize, bMemDataSetSize,
		bTotalDataSetSize;

	if (userOptions->usepessidata == false) {
		aL1DataSetSize = a->L1DataSetSize;
		aL2DataSetSize = a->L2DataSetSize;
		aL3DataSetSize = a->L3DataSetSize;
		aMemDataSetSize = a->MemDataSetSize;
		aTotalDataSetSize = a->TotalDataSetSize;
		bL1DataSetSize = b->L1DataSetSize;
		bL2DataSetSize = b->L2DataSetSize;
		bL3DataSetSize = b->L3DataSetSize;
		bMemDataSetSize = b->MemDataSetSize;
		bTotalDataSetSize = b->TotalDataSetSize;
	}
	else {
		aL1DataSetSize = a->PessiL1DataSetSize;
		aL2DataSetSize = a->PessiL2DataSetSize;
		aL3DataSetSize = a->PessiL3DataSetSize;
		aMemDataSetSize = a->PessiMemDataSetSize;
		aTotalDataSetSize = a->PessiTotalDataSetSize;
		bL1DataSetSize = b->PessiL1DataSetSize;
		bL2DataSetSize = b->PessiL2DataSetSize;
		bL3DataSetSize = b->PessiL3DataSetSize;
		bMemDataSetSize = b->PessiMemDataSetSize;
		bTotalDataSetSize = b->PessiTotalDataSetSize;
	}

	if (ExceedsByAThreshold(aTotalDataSetSize,
		bTotalDataSetSize,
		TOTALDATASETSIZETHRESHOLD)) {
		winner = 1;
		if (DEBUG)
			cout << "TotalDataSetSize_Winner" << endl;

	}
	else if (ExceedsByAThreshold(bTotalDataSetSize,
		aTotalDataSetSize,
		DATASETSIZETHRESHOLD)) {
		winner = 0;
		if (DEBUG)
			cout << "TotalDataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aMemDataSetSize, bMemDataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "MemDataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bMemDataSetSize, aMemDataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "MemDataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aL3DataSetSize, bL3DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L3DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL3DataSetSize, aL3DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L3DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aL2DataSetSize, bL2DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L2DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL2DataSetSize, aL2DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L2DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(aL1DataSetSize, bL1DataSetSize)) {
		winner = 1;
		if (DEBUG)
			cout << "L1DataSetSize_Winner" << endl;
	}
	else if (ExceedsByAThreshold(bL1DataSetSize, aL1DataSetSize)) {
		winner = 0;
		if (DEBUG)
			cout << "L1DataSetSize_Winner" << endl;
	}

	return winner;
}

bool ExceedsByAThreshold(long size1, long size2,
	double threshold) {
	if (size1 > (1 + threshold) * size2) {
		return true;
	}
	else {
		return false;
	}
}

bool compareByWins(const ProgramVariant* a, const ProgramVariant* b) {
	return a->wins > b->wins;
}

void AssignPolyRankBasedOnOrder(vector<ProgramVariant*> *programVariants)
{
	for (int i = 0; i < programVariants->size(); i++) {
		programVariants->at(i)->polyRank = i + 1;
	}
}