#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <pet.h>
#include <isl/union_set.h>
#include <isl/flow.h>
#include <barvinok/isl.h>
#include <isl/space.h>

void PrintScop(isl_ctx* ctx, struct pet_scop *scop);
void PrintExpressions(isl_printer *printer, pet_expr *expr);
void PrintUnionFlow(isl_union_flow* flow);
void PrintUnionMap(isl_union_map* map);
void PrintMap(isl_map* map);
void PrintBasicMap(isl_basic_map* map);
void PrintUnionSet(isl_union_set* set);
void PrintSet(isl_set* set);
void PrintUnionPwQpolynomial(isl_union_pw_qpolynomial* poly);
void PrintSpace(isl_space* space);
void PrintScopOriginal(isl_ctx *ctx, pet_scop* scop);
void PrintBasicSet(isl_basic_set* set);
#endif