#include <pet.h>
#include <iostream>
#include <isl/union_set.h>
#include <isl/flow.h>
#include <stdlib.h>
#include <barvinok/isl.h>
#include <string.h>
using namespace std;

/* Function header declarations begin */
void ComputeDataReuseWorkingSets(const char *fileName);
void PrintScop(isl_ctx* ctx, struct pet_scop *scop);
void PrintExpressions(isl_printer *printer, pet_expr *expr);
pet_scop* ParseScop(isl_ctx* ctx, const char *fileName);
isl_union_flow* ComputeDataDependences(isl_ctx* ctx, pet_scop* scop);
isl_stat ComputeWorkingSetSizesForDependence(isl_map* dep, void *user);
void PrintUnionFlow(isl_union_flow* flow);
void ComputeWorkingSetSizesForDependences(isl_union_flow *flow,
	pet_scop *scop);
void PrintUnionMap(isl_union_map* map);
void PrintMap(isl_map* map);
isl_stat ComputeWorkingSetSizesForDependenceBasicMap(isl_basic_map* dep,
	void *user);
void PrintBasicMap(isl_basic_map* map);
void PrintUnionSet(isl_union_set* set);
void PrintSet(isl_set* set);
void PrintUnionPwQpolynomial(isl_union_pw_qpolynomial* poly);
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
	ComputeWorkingSetSizesForDependences(flow, scop);
	isl_ctx_free(ctx);
	pet_scop_free(scop);
	isl_union_flow_free(flow);
}

void ComputeWorkingSetSizesForDependences(isl_union_flow *flow,
	pet_scop *scop) {
	/*TODO: The following code works for perfectly nested loops
	only. It needs to be extended to cover data dependences that span
	across loops*/

	/* Here we assume that only may_dependences will be present because
	ComputeDataDependences() function is specifying only may_read,
	and may_write references */
	isl_union_map *may_dependences = isl_union_flow_get_may_dependence(
		flow);
	cout << "May dependences: " << endl;
	PrintUnionMap(may_dependences);

	isl_union_map_foreach_map(may_dependences,
		&ComputeWorkingSetSizesForDependence, scop);

	isl_union_map_free(may_dependences);
}

isl_stat ComputeWorkingSetSizesForDependence(isl_map* dep, void *user) {
	isl_map_foreach_basic_map(dep,
		&ComputeWorkingSetSizesForDependenceBasicMap,
		user);
	return isl_stat_ok;
}

isl_stat ComputeWorkingSetSizesForDependenceBasicMap(isl_basic_map* dep,
	void *user) {
	pet_scop *scop = (pet_scop*)user;
	cout << "Dependence basic_map: " << endl;
	PrintBasicMap(dep);

	isl_basic_set* sourceDomain = isl_basic_map_domain(dep);

	isl_set* source = isl_basic_set_lexmin(
		isl_basic_set_copy(sourceDomain));

	isl_set* target = isl_set_lexmin(
		isl_set_apply(isl_set_copy(source),
			isl_map_from_basic_map(dep)));

	cout << "source: " << endl;
	PrintSet(source);

	cout << "target: " << endl;
	PrintSet(target);

	/* itersUptoSourceExcludingSource := sourceDomain << source */
	isl_union_set* itersUptoSourceExcludingSource =
		isl_union_map_domain(
			isl_union_set_lex_lt_union_set(
				isl_union_set_from_basic_set(
					isl_basic_set_copy(sourceDomain)),
				isl_union_set_from_set(isl_set_copy(source))));

	cout << "itersUptoSourceExcludingSource:" << endl;
	PrintUnionSet(itersUptoSourceExcludingSource);

	/* itersUptoTargetIncludingTarget := sourceDomain <<= target */
	isl_union_set* itersUptoTargetIncludingTarget =
		isl_union_map_domain(
			isl_union_set_lex_le_union_set(
				isl_union_set_from_basic_set(
					isl_basic_set_copy(sourceDomain)),
				isl_union_set_from_set(isl_set_copy(target))));

	cout << "itersUptoTargetIncludingTarget:" << endl;
	PrintUnionSet(itersUptoTargetIncludingTarget);

	/* WS :=  itersUptoTargetIncludingTarget - itersUptoSourceExcludingSource */

	isl_union_set* WS =
		isl_union_set_subtract(
			itersUptoTargetIncludingTarget,
			itersUptoSourceExcludingSource);

	cout << "Working Set Iteration space:" << endl;
	PrintUnionSet(WS);

	isl_union_map *may_reads = pet_scop_get_may_reads(scop);
	isl_union_map *may_writes = pet_scop_get_may_writes(scop);

	isl_union_set* readSet =
		isl_union_set_apply(isl_union_set_copy(WS), may_reads);
	isl_union_set* writeSet =
		isl_union_set_apply(isl_union_set_copy(WS), may_writes);
	isl_union_set* dataSet = isl_union_set_union(readSet, writeSet);
	isl_union_pw_qpolynomial* WSSize = isl_union_set_card(dataSet);

	cout << "Working Set Size:" << endl;
	PrintUnionPwQpolynomial(WSSize);

	isl_set_free(source);
	isl_set_free(target);
	isl_basic_set_free(sourceDomain);
	isl_union_set_free(WS);
	isl_union_pw_qpolynomial_free(WSSize);
	return isl_stat_ok;
}

isl_union_flow* ComputeDataDependences(isl_ctx* ctx, pet_scop* scop) {
	/*TODO: Compute other types of depdences also - RAW, WAR, WAR
	apart from RAR that are currently being computed*/
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
	cout << "Calling pet_scop_extract_from_C_source" << endl;
	pet_scop *scop = pet_scop_extract_from_C_source(ctx, fileName, NULL);
	cout << "scop: " << scop << endl;
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



