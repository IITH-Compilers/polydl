#include <pet.h>
#include <iostream>
using namespace std;

void ComputeDataReuseWorkingSets(char *fileName);
void ParseScop(isl_ctx* ctx, char *fileName);
void PrintScop(isl_ctx* ctx, struct pet_scop *scop);

int main(int argc, char **argv) {
	char *fileName = "../apps/padded_conv_fp_stride_1_libxsmm_core2.c";
	if (argc >= 2) {
		fileName = argv[1];
		cout << "file name : " << fileName << endl;

	}

	ComputeDataReuseWorkingSets(fileName);
	return 0;
}

void ComputeDataReuseWorkingSets(char *fileName) {
	isl_ctx* ctx = isl_ctx_alloc_with_pet_options();
	ParseScop(ctx, fileName);
	isl_ctx_free(ctx);
}

void ParseScop(isl_ctx* ctx, char *fileName) {
	pet_options_set_autodetect(ctx, 0);
	cout << "Calling pet_scop_extract_from_C_source" << endl;
	struct pet_scop *scop = pet_scop_extract_from_C_source(ctx, fileName, NULL);
	cout << "scop: " << scop << endl;
	PrintScop(ctx, scop);
	pet_scop_free(scop);
}

void PrintScop(isl_ctx* ctx, struct pet_scop *scop) {
	isl_printer *printer = isl_printer_to_file(ctx, stdout);
	printer = isl_printer_set_output_format(printer, ISL_FORMAT_ISL);

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

	isl_printer_free(printer);
}



