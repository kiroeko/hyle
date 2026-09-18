#pragma once
#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif
void hyle_test_gallocr_fail_at(int allocation);
int hyle_test_gallocr_calls(void);
int hyle_test_gallocr_live(void);
ggml_gallocr_t hyle_test_gallocr_new(ggml_backend_buffer_type_t);
void hyle_test_gallocr_free(ggml_gallocr_t);
bool hyle_test_gallocr_reserve(ggml_gallocr_t, struct ggml_cgraph *);
bool hyle_test_gallocr_alloc_graph(ggml_gallocr_t, struct ggml_cgraph *);
#ifdef __cplusplus
}
#endif
