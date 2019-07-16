#include <Utility.hpp>
#include <iostream>
using namespace std;

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