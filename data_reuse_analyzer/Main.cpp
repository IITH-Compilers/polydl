#include <pet.h>
#include <iostream>
#include <isl/union_set.h>
#include <isl/flow.h>
#include <stdlib.h>
#include <barvinok/isl.h>
#include <string.h>
#include <vector>
#include <isl/space.h>
#include <unordered_map>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

/* Function header declarations begin */
struct WorkingSetSize {
	isl_basic_map* dependence;
	isl_set* source;
	isl_set* target;
	isl_union_pw_qpolynomial* minSize;
	isl_union_pw_qpolynomial* maxSize;
};

typedef struct WorkingSetSize WorkingSetSize;
struct ArgComputeWorkingSetSizesForDependence {
	pet_scop *scop;
	vector<WorkingSetSize*>* workingSetSizes;
};

typedef struct ArgComputeWorkingSetSizesForDependence  ArgComputeWorkingSetSizesForDependence;

void ComputeDataReuseWorkingSets(const char *fileName);
void PrintScop(isl_ctx* ctx, struct pet_scop *scop);
void PrintExpressions(isl_printer *printer, pet_expr *expr);
pet_scop* ParseScop(isl_ctx* ctx, const char *fileName);
isl_union_flow* ComputeDataDependences(isl_ctx* ctx, pet_scop* scop);
isl_stat ComputeWorkingSetSizesForDependence(isl_map* dep, void *user);
void PrintUnionFlow(isl_union_flow* flow);
vector<WorkingSetSize*>* ComputeWorkingSetSizesForDependences(
	isl_union_flow *flow,
	pet_scop *scop);
void PrintUnionMap(isl_union_map* map);
void PrintMap(isl_map* map);
isl_stat ComputeWorkingSetSizesForDependenceBasicMap(isl_basic_map* dep,
	void *user);
void PrintBasicMap(isl_basic_map* map);
void PrintUnionSet(isl_union_set* set);
void PrintSet(isl_set* set);
void PrintUnionPwQpolynomial(isl_union_pw_qpolynomial* poly);
isl_union_pw_qpolynomial* ComputeDataSetSize(isl_basic_set* sourceDomain,
	isl_set* source, isl_set* target, pet_scop* scop);
isl_union_pw_qpolynomial* ComputeDataSetSize(isl_union_set* WS,
	isl_union_map *may_reads, isl_union_map *may_writes);
void FreeWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes);
void PrintWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes);
void SimplifyWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes,
	string fileName);
void PrintSpace(isl_space* space);
string SimplifyUnionPwQpolynomial(isl_union_pw_qpolynomial* size,
	unordered_map<string, int>* paramValues);
unordered_map<string, int>* GetParameterValues(vector<WorkingSetSize*>* workingSetSizes);
int findInParamsMap(unordered_map<string, int>* map, string key);
long ExtractIntegerFromUnionPwQpolynomial(isl_union_pw_qpolynomial* poly);
string GetParameterValuesString(unordered_map<string, int>* paramValues);
/* Function header declarations end */

int main(int argc, char **argv) {
	string fileName = "../apps/padded_conv_fp_stride_1_libxsmm_core2.c";
	if (argc >= 2) {
		fileName = argv[1];
		cout << "file name : " << fileName << endl;

	}

	ComputeDataReuseWorkingSets(fileName.c_str());
	return 0;
}

void ComputeDataReuseWorkingSets(const char *fileName) {
	isl_ctx* ctx = isl_ctx_alloc_with_pet_options();
	pet_scop *scop = ParseScop(ctx, fileName);
	isl_union_flow* flow = ComputeDataDependences(ctx, scop);
	string fileNameStr = string(fileName);
	vector<WorkingSetSize*>* workingSetSizes =
		ComputeWorkingSetSizesForDependences(flow, scop);
	PrintWorkingSetSizes(workingSetSizes);
	SimplifyWorkingSetSizes(workingSetSizes, fileNameStr);
	FreeWorkingSetSizes(workingSetSizes);
	pet_scop_free(scop);
	isl_union_flow_free(flow);
	isl_ctx_free(ctx);
}


vector<WorkingSetSize*>* ComputeWorkingSetSizesForDependences(
	isl_union_flow *flow,
	pet_scop *scop) {
	/*TODO: The following code works for perfectly nested loops
	only. It needs to be extended to cover data dependences that span
	across loops*/

	/* Here we assume that only may_dependences will be present because
	ComputeDataDependences() function is specifying only may_read,
	and may_write references */
	isl_union_map *may_dependences = isl_union_flow_get_may_dependence(
		flow);
	PrintUnionMap(may_dependences);

	vector<WorkingSetSize*>* workingSetSizes =
		new vector<WorkingSetSize*>();
	ArgComputeWorkingSetSizesForDependence* arg =
		(ArgComputeWorkingSetSizesForDependence*)malloc(
			sizeof(ArgComputeWorkingSetSizesForDependence));
	arg->scop = scop;
	arg->workingSetSizes = workingSetSizes;
	isl_union_map_foreach_map(may_dependences,
		&ComputeWorkingSetSizesForDependence, arg);

	isl_union_map_free(may_dependences);
	return workingSetSizes;
}

isl_stat ComputeWorkingSetSizesForDependence(isl_map* dep, void *user) {
	isl_map_foreach_basic_map(dep,
		&ComputeWorkingSetSizesForDependenceBasicMap,
		user);
	return isl_stat_ok;
}

isl_stat ComputeWorkingSetSizesForDependenceBasicMap(isl_basic_map* dep,
	void *user) {
	ArgComputeWorkingSetSizesForDependence* arg =
		(ArgComputeWorkingSetSizesForDependence*)user;
	pet_scop *scop = arg->scop;
	vector<WorkingSetSize*>* workingSetSizes = arg->workingSetSizes;

	isl_basic_set* sourceDomain = isl_basic_map_domain(
		isl_basic_map_copy(dep));

	isl_set* source = isl_basic_set_lexmin(
		isl_basic_set_copy(sourceDomain));

	isl_set* target = isl_set_apply(isl_set_copy(source),
		isl_map_from_basic_map(dep));

	isl_set* minTarget = isl_set_lexmin(isl_set_copy(target));
	isl_set* maxTarget = isl_set_lexmax(isl_set_copy(target));

	isl_union_pw_qpolynomial* minWSSize =
		ComputeDataSetSize(sourceDomain, source, minTarget, scop);

	isl_union_pw_qpolynomial* maxWSSize =
		ComputeDataSetSize(sourceDomain, source, maxTarget, scop);

	WorkingSetSize* workingSetSize =
		(WorkingSetSize*)malloc(sizeof(WorkingSetSize));
	workingSetSize->dependence = dep;
	workingSetSize->source = source;
	workingSetSize->target = target;
	workingSetSize->minSize = minWSSize;
	workingSetSize->maxSize = maxWSSize;
	workingSetSizes->push_back(workingSetSize);

	isl_basic_set_free(sourceDomain);
	isl_set_free(minTarget);
	isl_set_free(maxTarget);

	return isl_stat_ok;
}

isl_union_pw_qpolynomial* ComputeDataSetSize(isl_basic_set* sourceDomain,
	isl_set* source, isl_set* target, pet_scop* scop) {

	/* itersUptoSourceExcludingSource := sourceDomain << source */
	isl_union_set* itersUptoSourceExcludingSource =
		isl_union_map_domain(
			isl_union_set_lex_lt_union_set(
				isl_union_set_from_basic_set(
					isl_basic_set_copy(sourceDomain)),
				isl_union_set_from_set(isl_set_copy(source))));

	/* itersUptoTargetIncludingTarget := sourceDomain <<= target */
	isl_union_set* itersUptoTargetIncludingTarget =
		isl_union_map_domain(
			isl_union_set_lex_le_union_set(
				isl_union_set_from_basic_set(
					isl_basic_set_copy(sourceDomain)),
				isl_union_set_from_set(isl_set_copy(target))));

	/* WS :=  itersUptoTargetIncludingTarget - itersUptoSourceExcludingSource */

	isl_union_set* WS =
		isl_union_set_subtract(
			itersUptoTargetIncludingTarget,
			itersUptoSourceExcludingSource);

	isl_union_map *may_reads = pet_scop_get_may_reads(scop);
	isl_union_map *may_writes = pet_scop_get_may_writes(scop);

	isl_union_pw_qpolynomial* WSSize = ComputeDataSetSize(
		WS, may_reads, may_writes);
	isl_union_set_free(WS);
	isl_union_map_free(may_reads);
	isl_union_map_free(may_writes);
	return WSSize;
}

isl_union_pw_qpolynomial* ComputeDataSetSize(isl_union_set* WS,
	isl_union_map *may_reads, isl_union_map *may_writes) {
	isl_union_set* readSet =
		isl_union_set_apply(isl_union_set_copy(WS),
			isl_union_map_copy(may_reads));
	isl_union_set* writeSet =
		isl_union_set_apply(isl_union_set_copy(WS),
			isl_union_map_copy(may_writes));
	isl_union_set* dataSet = isl_union_set_union(readSet, writeSet);
	return isl_union_set_card(dataSet);
}


void SimplifyWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes,
	string fileName) {
	cout << "Number of working set sizes: " << workingSetSizes->size()
		<< endl;

	string DIR = "stats";
	string suffix = "_ws_stats.tsv";
	ofstream file;
	string fullFileName = /*DIR + "/" + */ fileName + suffix;
	file.open(fullFileName);

	if (file.is_open()) {
		cout << "Writing to file " << fullFileName << endl;
	}
	else {
		cout << "Could not open the file: " << fullFileName << endl;
	}

	unordered_map<string, int>* paramValues = GetParameterValues(
		workingSetSizes);

	file << "Parameters: " << GetParameterValuesString(paramValues)
		<< endl;
	file << "dependence \t min_WS_size \t max_WS_size\n";
	for (int i = 0; i < workingSetSizes->size(); i++) {
		isl_union_pw_qpolynomial* minSizePoly =
			workingSetSizes->at(i)->minSize;
		string minSize = SimplifyUnionPwQpolynomial(
			minSizePoly,
			paramValues);

		isl_union_pw_qpolynomial* maxSizePoly =
			workingSetSizes->at(i)->maxSize;
		string maxSize = SimplifyUnionPwQpolynomial(
			maxSizePoly,
			paramValues);

		if (!minSize.empty() || !maxSize.empty()) {
			file << isl_basic_map_to_str(
				workingSetSizes->at(i)->dependence)
				<< "\t";
			file << minSize << "\t";
			file << maxSize << endl;
		}
	}

	file << endl;

	paramValues->clear();
	file.close();
	delete paramValues;
}

unordered_map<string, int>* GetParameterValues(vector<WorkingSetSize*>* workingSetSizes) {
	unordered_map<string, int>* paramValues =
		new unordered_map<string, int>();

	if (workingSetSizes->size() > 0) {
		isl_union_pw_qpolynomial* repPoly =
			workingSetSizes->at(0)->minSize;
		isl_space* space = isl_union_pw_qpolynomial_get_space(repPoly);
		isl_size numParams =
			isl_space_dim(space, isl_dim_param);

		cout << "Enter values for the following parameters:" << endl;
		for (int j = 0; j < numParams; j++) {
			string name(isl_space_get_dim_name(
				space, isl_dim_param, (unsigned)j));
			cout << name << " ";
		}

		cout << endl;

		for (int j = 0; j < numParams; j++) {
			string name(isl_space_get_dim_name(
				space, isl_dim_param, (unsigned)j));
			int val;
			cin >> val;
			paramValues->insert({ {name, val} });
		}

		isl_space_free(space);
	}

	return paramValues;
}

string GetParameterValuesString(unordered_map<string, int>* paramValues) {
	string params = "";
	for (auto i : *paramValues) {
		params += i.first + " = " + to_string(i.second) + "; ";
	}

	return params;
}

string SimplifyUnionPwQpolynomial(isl_union_pw_qpolynomial* size,
	unordered_map<string, int>* paramValues) {
	isl_set* context = isl_set_universe(
		isl_union_pw_qpolynomial_get_space(size));

	/*Add constraints now*/
	isl_constraint *c;
	isl_local_space *ls;
	isl_space* space;
	space = isl_set_get_space(context);
	ls = isl_local_space_from_space(isl_space_copy(space));

	isl_size numParams =
		isl_set_dim(context, isl_dim_param);

	for (int j = 0; j < numParams; j++) {
		string name(isl_space_get_dim_name(
			space, isl_dim_param, (unsigned)j));
		int val = findInParamsMap(paramValues, name);

		c = isl_constraint_alloc_equality(
			isl_local_space_copy(ls));
		c = isl_constraint_set_coefficient_si(
			c, isl_dim_param, j, -1);
		c = isl_constraint_set_constant_si(c, val);
		context = isl_set_add_constraint(context, c);
	}

	isl_union_pw_qpolynomial* gistSize =
		isl_union_pw_qpolynomial_gist_params(
			isl_union_pw_qpolynomial_copy(size),
			context);

	long sizeInteger = ExtractIntegerFromUnionPwQpolynomial(gistSize);
	cout << "gistSize: " << sizeInteger << endl;

	string sizeString;
	if (sizeInteger != -1) {
		sizeString = to_string(sizeInteger);
	}

	isl_space_free(space);
	isl_local_space_free(ls);
	return sizeString;
}

long ExtractIntegerFromUnionPwQpolynomial(
	isl_union_pw_qpolynomial* polynomial)
{
	/*The string representation typically will be of the following kind:
	[pad_w, ifwp, pad_h, ifhp, nIfm, ofwp, ofhp, nOfm, kw, kh, nImg, ofh, ofw] -> { 290 }

	This function extracts the integer -- in this case 290 from the string
	representation such as the above.*/

	long val = -1;
	cout << "Converting: " << endl;
	PrintUnionPwQpolynomial(polynomial);
	string poly(isl_union_pw_qpolynomial_to_str(polynomial));

	string openingBrace = "{";
	string closingBrace = "}";

	size_t begin = -1, end = -1;
	size_t found = poly.find(openingBrace);
	if (found != string::npos) {
		begin = found + 1;
	}

	found = poly.find(closingBrace);
	if (found != string::npos) {
		end = found - 1;
	}

	if (begin != -1 && end != -1 && begin < end) {
		string valStr = poly.substr(begin, end - begin + 1);
		try {
			val = stol(valStr, nullptr, 10);
			cout << "Converted string " << valStr << " to integer "
				<< val << endl;
			return val;
		}
		catch (const invalid_argument) {
			cerr << "Invalid argument" << endl;
		}
	}

	return val;
}

int findInParamsMap(unordered_map<string, int>* map, string key) {
	unordered_map<string, int>::const_iterator find =
		map->find(key);

	if (find == map->end()) {
		cout << "Parameter value not found, exiting." << endl;
		exit(1);
		return -1;
	}
	else {
		return find->second;
	}
}

void PrintWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes) {
	cout << "Number of working set sizes: " << workingSetSizes->size()
		<< endl;
	for (int i = 0; i < workingSetSizes->size(); i++) {
		cout << "*********************************************" << endl;
		cout << "*********************************************" << endl;
		cout << "dependence: " << endl;
		PrintBasicMap(workingSetSizes->at(i)->dependence);

		cout << "source: " << endl;
		PrintSet(workingSetSizes->at(i)->source);

		cout << "target: " << endl;
		PrintSet(workingSetSizes->at(i)->target);

		cout << "MinSize: " << endl;
		PrintUnionPwQpolynomial(workingSetSizes->at(i)->minSize);

		cout << "MaxSize: " << endl;
		PrintUnionPwQpolynomial(workingSetSizes->at(i)->maxSize);
		cout << "*********************************************" << endl;
		cout << "*********************************************" << endl;
		cout << endl;
	}
}


void FreeWorkingSetSizes(vector<WorkingSetSize*>* workingSetSizes) {
	for (int i = 0; i < workingSetSizes->size(); i++) {
		if (workingSetSizes->at(i) == NULL) {
			continue;
		}

		if (workingSetSizes->at(i)->dependence) {
			isl_basic_map_free(workingSetSizes->at(i)->dependence);
		}

		if (workingSetSizes->at(i)->source) {
			isl_set_free(workingSetSizes->at(i)->source);
		}

		if (workingSetSizes->at(i)->target) {
			isl_set_free(workingSetSizes->at(i)->target);
		}

		if (workingSetSizes->at(i)->minSize) {
			isl_union_pw_qpolynomial_free(
				workingSetSizes->at(i)->minSize);
		}

		if (workingSetSizes->at(i)->maxSize) {
			isl_union_pw_qpolynomial_free(
				workingSetSizes->at(i)->maxSize);
		}

		free(workingSetSizes->at(i));
	}

	delete workingSetSizes;
}

isl_union_flow* ComputeDataDependences(isl_ctx* ctx, pet_scop* scop) {
	/*TODO: Compute other types of depdences also - RAW, WAR, WAR
	apart from RAR that are currently being computed*/

	/*TODO: Print the array because of which the dependence is formed -
	use "full" dependence structrues*/
	isl_schedule* schedule = pet_scop_get_schedule(scop);
	isl_union_map *may_reads = pet_scop_get_may_reads(scop);
	isl_union_map *may_writes = pet_scop_get_may_writes(scop);

	// RAR dependences
	isl_union_access_info* access_info =
		isl_union_access_info_from_sink(isl_union_map_copy(may_reads));
	access_info = isl_union_access_info_set_may_source(access_info,
		isl_union_map_copy(may_reads));
	isl_union_access_info_set_schedule(access_info,
		isl_schedule_copy(schedule));

	// Compute the RAR dependences
	isl_union_flow *RAR =
		isl_union_access_info_compute_flow(access_info);
	cout << "RAR dependences are:" << endl;
	if (RAR == NULL) {
		cout << "No RAR dependences found" << endl;
	}
	else {
		cout << "Calling PrintUnionAccessInfo" << endl;
		PrintUnionFlow(RAR);
	}

	isl_union_map_free(may_writes);
	isl_union_map_free(may_reads);
	isl_schedule_free(schedule);
	return RAR;
}

void PrintUnionPwQpolynomial(isl_union_pw_qpolynomial* poly) {
	isl_printer *printer = isl_printer_to_file(
		isl_union_pw_qpolynomial_get_ctx(poly), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_union_pw_qpolynomial(printer, poly);
	cout << endl;

	//printer = isl_printer_set_output_format(printer, ISL_FORMAT_C);
	//isl_printer_print_union_pw_qpolynomial(printer, poly);
	isl_printer_free(printer);
}

void PrintBasicMap(isl_basic_map* map) {
	isl_printer *printer = isl_printer_to_file(
		isl_basic_map_get_ctx(map), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_basic_map(printer, map);
	cout << endl;
	isl_printer_free(printer);
}

void PrintMap(isl_map* map) {
	isl_printer *printer = isl_printer_to_file(
		isl_map_get_ctx(map), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_map(printer, map);
	cout << endl;
	isl_printer_free(printer);
}

void PrintSpace(isl_space* space) {
	isl_printer *printer = isl_printer_to_file(
		isl_space_get_ctx(space), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_space(printer, space);
	cout << endl;
	isl_printer_free(printer);
}

void PrintUnionMap(isl_union_map* map) {
	isl_printer *printer = isl_printer_to_file(
		isl_union_map_get_ctx(map), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_union_map(printer, map);
	cout << endl;
	isl_printer_free(printer);
}

void PrintSet(isl_set* set) {
	isl_printer *printer = isl_printer_to_file(
		isl_set_get_ctx(set), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_set(printer, set);
	cout << endl;
	isl_printer_free(printer);
}

void PrintUnionSet(isl_union_set* set) {
	isl_printer *printer = isl_printer_to_file(
		isl_union_set_get_ctx(set), stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	isl_printer_print_union_set(printer, set);
	cout << endl;
	isl_printer_free(printer);
}

void PrintUnionFlow(isl_union_flow* flow) {
	isl_ctx* ctx = isl_union_flow_get_ctx(flow);
	isl_printer *printer = isl_printer_to_file(ctx, stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);
	printer = isl_printer_print_union_flow(printer, flow);
	cout << endl;
	isl_printer_free(printer);
}

pet_scop* ParseScop(isl_ctx* ctx, const char *fileName) {
	pet_options_set_autodetect(ctx, 0);
	pet_scop *scop = pet_scop_extract_from_C_source(ctx, fileName, NULL);
	return scop;
}

void PrintScop(isl_ctx* ctx, struct pet_scop *scop) {
	isl_printer *printer = isl_printer_to_file(ctx, stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);

	cout << "pet_scop_get_context:" << endl;
	isl_printer_print_set(printer, pet_scop_get_context(scop));
	cout << endl;

	cout << "pet_scop_get_schedule:" << endl;
	isl_printer_print_schedule(printer, pet_scop_get_schedule(scop));
	cout << endl;

	cout << "pet_scop_get_instance_set:" << endl;
	isl_printer_print_union_set(printer, pet_scop_get_instance_set(scop));
	cout << endl;

	cout << "pet_scop_get_may_reads:" << endl;
	isl_printer_print_union_map(printer, pet_scop_get_may_reads(scop));
	cout << endl;
	return;

	// Print Context
	isl_set *context = scop->context;
	cout << "Context:" << endl;
	isl_printer_print_set(printer, context);

	// Print context_value
	cout << "context_value:" << endl;
	isl_printer_print_set(printer, scop->context_value);

	// Print context_value
	// cout << "schedule:" << endl;
	// isl_printer_print_schedule(printer, scop->schedule);

	cout << endl << "#pet_types:" << scop->n_type << endl;
	for (int i = 0; i < scop->n_type; i++) {
		cout << "name: " << scop->types[i]->name << endl;
		cout << "definition: " << scop->types[i]->definition << endl;
	}

	cout << "scop->n_array: " << scop->n_array << endl;
	for (int i = 0; i < scop->n_array; i++) {

		if (scop->arrays[i]->context) {
			cout << "scop->arrays[i]->context: " << endl;
			isl_printer_print_set(printer, scop->arrays[i]->context);
		}

		if (scop->arrays[i]->extent) {
			cout << "scop->arrays[i]->extent: " << endl;
			isl_printer_print_set(printer, scop->arrays[i]->extent);
		}

		if (scop->arrays[i]->value_bounds) {
			cout << "scop->arrays[i]->value_bounds: " << endl;
			isl_printer_print_set(printer, scop->arrays[i]->value_bounds);
		}
	}

	cout << "n_stmt: " << scop->n_stmt << endl;
	for (int i = 0; i < scop->n_stmt; i++) {
		cout << "Printing " << i << " th statement" << endl;
		cout << "scop->stmts[i]->domain: " << endl;
		isl_printer_print_set(printer, scop->stmts[i]->domain);

		cout << "Body: " << endl;
		pet_tree_dump(scop->stmts[i]->body);
		cout << endl;
		if (pet_tree_is_loop(scop->stmts[i]->body)) {
			cout << "It is a LOOP" << endl;
		}
		else {
			cout << "It is NOT a loop" << endl;
		}

		cout << "scop->stmts[i]->n_arg: " << scop->stmts[i]->n_arg << endl;
		for (int j = 0; j < scop->stmts[i]->n_arg; i++) {
			pet_expr_dump(scop->stmts[i]->args[i]);
			cout << endl;
		}

		if (pet_tree_get_type(scop->stmts[i]->body) == pet_tree_expr) {
			PrintExpressions(printer,
				pet_tree_expr_get_expr(scop->stmts[i]->body));
		}
	}

	isl_printer_free(printer);
}

void PrintExpressions(isl_printer *printer, pet_expr *expr) {
	if (expr) {
		if (pet_expr_get_type(expr) == pet_expr_access) {
			cout << "Access relation: " << endl;
			pet_expr_dump(expr);
			cout << endl << "pet_expr_get_n_arg(expr): "
				<< pet_expr_get_n_arg(expr) << endl;
			if (/* pet_expr_is_affine(expr) && */
				pet_expr_access_is_read(expr)) {
				cout << "Encountered a READ expression: " << endl;
				isl_printer_print_union_map(printer,
					pet_expr_access_get_may_read(expr));
			}
			else if (/* pet_expr_is_affine(expr) && */
				pet_expr_access_is_write(expr)) {
				cout << "Encountered a WRITE expression: " << endl;
				isl_printer_print_union_map(printer,
					pet_expr_access_get_may_write(expr));
			}

			cout << endl;
		}
		else {
			for (int i = 0; i < pet_expr_get_n_arg(expr); i++) {
				PrintExpressions(printer, pet_expr_get_arg(expr, i));
			}
		}
	}
}



