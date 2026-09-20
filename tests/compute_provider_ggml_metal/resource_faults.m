// Source inclusion changes only allocation/completion boundaries in this test
// executable. Production objects and their native math/encoding are unchanged.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include "ggml-metal-device.h"

extern void hyle_metal_fault_map_enter(void);
extern void hyle_metal_fault_map_leave(void);

// Restrict C++ allocation injection to our actual resource collections. Metal's
// private driver allocations have a different failure contract (NSError/nil).
static bool scoped_pipelines_add(ggml_metal_pipelines_t ppls, const char *name,
                                  ggml_metal_pipeline_t pipeline) {
    hyle_metal_fault_map_enter();
    const bool result = ggml_metal_pipelines_add(ppls, name, pipeline);
    hyle_metal_fault_map_leave();
    return result;
}
static ggml_metal_pipeline_t scoped_pipelines_get(ggml_metal_pipelines_t ppls,
                                                 const char *name, bool *failed) {
    hyle_metal_fault_map_enter();
    const ggml_metal_pipeline_t result = ggml_metal_pipelines_get(ppls, name, failed);
    hyle_metal_fault_map_leave();
    return result;
}

static _Thread_local int allocation_left = -1;
static _Thread_local int allocation_failed;
static _Thread_local int64_t completion_failure;
static pthread_mutex_t allocations_mutex = PTHREAD_MUTEX_INITIALIZER;
static void *allocations[512];
static int live_allocations;

void hyle_metal_fault_arm(int index) {
    allocation_left = index;
    allocation_failed = 0;
}
int hyle_metal_fault_disarm(void) {
    allocation_left = -1;
    return allocation_failed;
}
int hyle_metal_fault_live(void) {
    pthread_mutex_lock(&allocations_mutex);
    const int result = live_allocations;
    pthread_mutex_unlock(&allocations_mutex);
    return result;
}
void hyle_metal_fault_completion(int64_t code) { completion_failure = code; }

static void *resource_calloc(size_t count, size_t bytes) {
    if (allocation_left > 0 && --allocation_left == 0) {
        allocation_left = -1;
        allocation_failed = 1;
        return NULL;
    }
    void *result = calloc(count, bytes);
    if (result) {
        pthread_mutex_lock(&allocations_mutex);
        int slot = 0;
        while (slot < 512 && allocations[slot]) ++slot;
        if (slot == 512) abort();
        allocations[slot] = result;
        ++live_allocations;
        pthread_mutex_unlock(&allocations_mutex);
    }
    return result;
}
static void resource_free(void *pointer) {
    pthread_mutex_lock(&allocations_mutex);
    for (int i = 0; i < 512; ++i)
        if (pointer && allocations[i] == pointer) {
            allocations[i] = NULL;
            --live_allocations;
            break;
        }
    pthread_mutex_unlock(&allocations_mutex);
    free(pointer);
}

#define calloc resource_calloc
#define free resource_free
#define ggml_metal_pipelines_add scoped_pipelines_add
#define ggml_metal_pipelines_get scoped_pipelines_get
#include "ggml-metal-device.m"
#undef ggml_metal_pipelines_get
#undef ggml_metal_pipelines_add
#define ggml_metal_checked_sync hyle_metal_actual_checked_sync
#include "ggml-metal-context.m"
#undef ggml_metal_checked_sync
#undef free
#undef calloc

int64_t ggml_metal_checked_sync(ggml_metal_t context) {
    // Inject only after real work and the original native error check drain.
    const int64_t actual = hyle_metal_actual_checked_sync(context);
    const int64_t injected = completion_failure;
    completion_failure = 0;
    return actual ? actual : injected;
}
