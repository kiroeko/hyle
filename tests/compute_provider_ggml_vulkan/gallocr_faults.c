// Compile the exact patched C source with local allocator failure injection.
// Product code has no test hook, alternate planner, or process malloc override.
#include <stdlib.h>

static int calls, fail_at, live;
void hyle_test_gallocr_fail_at(int allocation) { calls = 0; fail_at = allocation; }
int hyle_test_gallocr_calls(void) { return calls; }
int hyle_test_gallocr_live(void) { return live; }
static void * fault_malloc(size_t bytes) {
    if (++calls == fail_at) return NULL;
    void * result = malloc(bytes);
    if (result) ++live;
    return result;
}
static void * fault_calloc(size_t count, size_t bytes) {
    if (++calls == fail_at) return NULL;
    void * result = calloc(count, bytes);
    if (result) ++live;
    return result;
}
static void * fault_realloc(void * pointer, size_t bytes) {
    if (++calls == fail_at) return NULL;
    void * result = realloc(pointer, bytes);
    if (result && !pointer) ++live;
    return result;
}
static void fault_free(void * pointer) {
    if (pointer) --live;
    free(pointer);
}

#define malloc fault_malloc
#define calloc fault_calloc
#define realloc fault_realloc
#define free fault_free
#define ggml_hash_set_free hyle_test_hash_set_free
#define ggml_op_can_inplace hyle_test_op_can_inplace
#define ggml_tallocr_new hyle_test_tallocr_new
#define ggml_tallocr_alloc hyle_test_tallocr_alloc
#define ggml_gallocr_new_n hyle_test_gallocr_new_n
#define ggml_gallocr_new hyle_test_gallocr_new
#define ggml_gallocr_free hyle_test_gallocr_free
#define ggml_gallocr_reserve_n_size hyle_test_gallocr_reserve_n_size
#define ggml_gallocr_reserve_n hyle_test_gallocr_reserve_n
#define ggml_gallocr_reserve hyle_test_gallocr_reserve
#define ggml_gallocr_alloc_graph hyle_test_gallocr_alloc_graph
#define ggml_gallocr_get_buffer_size hyle_test_gallocr_get_buffer_size
#define ggml_backend_alloc_ctx_tensors_from_buft_size hyle_test_alloc_ctx_size
#define ggml_backend_alloc_ctx_tensors_from_buft hyle_test_alloc_ctx_buft
#define ggml_backend_alloc_ctx_tensors hyle_test_alloc_ctx

// The upstream header declares the renamed functions inside this TU.
#include "ggml-alloc.c"

void hyle_test_hash_set_free(struct ggml_hash_set * set) {
    fault_free(set->used);
    fault_free(set->keys);
}
