#include <pet.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	char *fileName = "../apps/padded_conv_fp_stride_1_libxsmm_core2.c";
	if (argc >= 2)
	{
		fileName = argv[1];
		cout << "file name : " << fileName << endl;

	}




	isl_ctx*  ctx = isl_ctx_alloc();
	cout << "Calling pet_scop_extract_from_C_source" << endl;
	struct pet_scop *scop = pet_scop_extract_from_C_source(ctx, fileName, NULL);
	cout << "scop: " << scop << endl; 
	return 0;
}
