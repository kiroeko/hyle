#include "hyle/c/io.h"

#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(hyle_io_backend) == sizeof(uint32_t),
               "IO backend must have a fixed C ABI width");
_Static_assert(sizeof(hyle_io_http2_role) == sizeof(uint32_t),
               "HTTP/2 role must have a fixed C ABI width");
_Static_assert(offsetof(hyle_io_capabilities, backend) == 4,
               "IO capabilities backend ABI offset");
_Static_assert(offsetof(hyle_io_capabilities, bits) == 8,
               "IO capabilities bits ABI offset");
_Static_assert(sizeof(hyle_io_capabilities) == 16,
               "IO capabilities frozen v1 ABI size");

_Static_assert(sizeof(((hyle_io_proactor_options*)0)->struct_size) ==
                   sizeof(uint32_t),
               "proactor options struct_size ABI width");
_Static_assert(offsetof(hyle_io_proactor_options, reserved) == 4,
               "proactor options reserved ABI offset");
_Static_assert(offsetof(hyle_io_proactor_options, entries) == 8,
               "proactor options entries ABI offset");
_Static_assert(offsetof(hyle_io_proactor_options, defer_taskrun) == 12,
               "proactor options defer ABI offset");
_Static_assert(sizeof(((hyle_io_proactor_options*)0)->defer_taskrun) ==
                   sizeof(uint32_t),
               "proactor options defer ABI width");
_Static_assert(offsetof(hyle_io_proactor_options, backing) == 16,
               "proactor options allocator ABI offset");
_Static_assert(HYLE_IO_PROACTOR_OPTIONS_REQUIRED_SIZE == 8,
               "proactor options semantic required prefix");
_Static_assert(sizeof(hyle_io_proactor_options) == 48,
               "proactor options 64-bit ABI size");

typedef struct io_options_required_prefix {
  uint32_t struct_size;
  uint32_t reserved;
} io_options_required_prefix;

/* A physical prefix containing only the first allocator subfield. The options
 * decoder must treat the entire hyle_allocator as absent. */
typedef struct io_options_partial_allocator {
  uint32_t struct_size;
  uint32_t reserved;
  uint32_t entries;
  uint32_t defer_taskrun;
  unsigned char backing_fragment[sizeof(void*)];
} io_options_partial_allocator;

_Static_assert(sizeof(io_options_required_prefix) == 8,
               "physical proactor required prefix size");
_Static_assert(offsetof(io_options_partial_allocator, backing_fragment) ==
                   offsetof(hyle_io_proactor_options, backing),
               "physical proactor allocator fragment offset");

typedef struct io_test_allocator_state {
  hyle_allocator base;
  size_t allocations;
  size_t frees;
  size_t reallocations;
} io_test_allocator_state;

static void* io_test_alloc(void* raw, size_t size, size_t alignment) {
  io_test_allocator_state* state = (io_test_allocator_state*)raw;
  void* out = state->base.alloc(state->base.state, size, alignment);
  if (out != NULL) ++state->allocations;
  return out;
}

static void io_test_free(void* raw, void* ptr, size_t size,
                         size_t alignment) {
  io_test_allocator_state* state = (io_test_allocator_state*)raw;
  if (ptr != NULL) ++state->frees;
  state->base.free(state->base.state, ptr, size, alignment);
}

static void* io_test_realloc(void* raw, void* ptr, size_t old_size,
                             size_t new_size, size_t alignment) {
  io_test_allocator_state* state = (io_test_allocator_state*)raw;
  void* out = state->base.realloc(state->base.state, ptr, old_size, new_size,
                                  alignment);
  ++state->reallocations;
  return out;
}

#define CHECK_OPTIONS(expr)          \
  do {                               \
    if (!(expr)) return __LINE__;    \
  } while (0)

static int check_proactor_options_contract(void) {
  hyle_io_proactor* p;

  /* Every business option is absent from the semantic minimum prefix. */
  {
    io_options_required_prefix prefix = {
        (uint32_t)sizeof(io_options_required_prefix), 0};
    p = hyle_io_proactor_create(
        (const hyle_io_proactor_options*)(const void*)&prefix);
    CHECK_OPTIONS(p != NULL);
    hyle_io_proactor_destroy(p);
  }

  /* Reject before reading reserved: this object physically contains only the
   * size word and truthfully declares that four-byte prefix. */
  {
    uint32_t too_short = (uint32_t)sizeof(too_short);
    p = hyle_io_proactor_create(
        (const hyle_io_proactor_options*)(const void*)&too_short);
    CHECK_OPTIONS(p == NULL);
    CHECK_OPTIONS(strstr(hyle_io_last_error(), "invalid argument") != NULL);
  }

  {
    io_options_required_prefix invalid = {
        (uint32_t)sizeof(io_options_required_prefix), 1};
    p = hyle_io_proactor_create(
        (const hyle_io_proactor_options*)(const void*)&invalid);
    CHECK_OPTIONS(p == NULL);
    CHECK_OPTIONS(strstr(hyle_io_last_error(), "reserved") != NULL);
  }

  {
    hyle_io_proactor_options invalid = HYLE_IO_PROACTOR_OPTIONS_INIT;
    invalid.defer_taskrun = 2;
    p = hyle_io_proactor_create(&invalid);
    CHECK_OPTIONS(p == NULL);
    CHECK_OPTIONS(strstr(hyle_io_last_error(), "0 or 1") != NULL);
  }

  /* One allocator pointer is not a usable partial vtable. If the decoder
   * byte-overlaid this prefix onto the system default, it would manufacture a
   * mixed invalid allocator; the whole-field rule retains the default. */
  {
    io_options_partial_allocator partial;
    memset(&partial, 0, sizeof(partial));
    partial.struct_size = (uint32_t)sizeof(partial);
    partial.entries = 256;
    p = hyle_io_proactor_create(
        (const hyle_io_proactor_options*)(const void*)&partial);
    CHECK_OPTIONS(p != NULL);
    hyle_io_proactor_destroy(p);
  }

  /* A complete allocator is applied as one value and owns the opaque handle. */
  {
    hyle_io_proactor_options complete = HYLE_IO_PROACTOR_OPTIONS_INIT;
    io_test_allocator_state state = {hyle_system_allocator(), 0, 0, 0};
    complete.backing = (hyle_allocator){io_test_alloc, io_test_free,
                                        io_test_realloc, &state};
    p = hyle_io_proactor_create(&complete);
    CHECK_OPTIONS(p != NULL);
    CHECK_OPTIONS(state.allocations == 1);
    hyle_io_proactor_destroy(p);
    CHECK_OPTIONS(state.frees == 1);
  }

  /* Conversely, a fully present but invalid allocator must not be mistaken
   * for an absent optional tail. */
  {
    hyle_io_proactor_options invalid = HYLE_IO_PROACTOR_OPTIONS_INIT;
    invalid.backing.alloc = io_test_alloc;
    p = hyle_io_proactor_create(&invalid);
    CHECK_OPTIONS(p == NULL);
    CHECK_OPTIONS(strstr(hyle_io_last_error(), "allocator") != NULL);
  }

  /* Future tail bytes are outside this library's known layout and unread. */
  {
    struct future_io_options {
      hyle_io_proactor_options known;
      unsigned char tail[16];
    } future;
    memset(&future, 0xA5, sizeof(future));
    future.known = (hyle_io_proactor_options)HYLE_IO_PROACTOR_OPTIONS_INIT;
    future.known.struct_size = (uint32_t)sizeof(future);
    p = hyle_io_proactor_create(&future.known);
    CHECK_OPTIONS(p != NULL);
    hyle_io_proactor_destroy(p);
  }

  /* memcpy-based reads also accept a physically unaligned FFI byte buffer. */
  {
    hyle_io_proactor_options known = HYLE_IO_PROACTOR_OPTIONS_INIT;
    unsigned char raw[sizeof(known) + 1];
    memcpy(raw + 1, &known, sizeof(known));
    p = hyle_io_proactor_create(
        (const hyle_io_proactor_options*)(const void*)(raw + 1));
    CHECK_OPTIONS(p != NULL);
    hyle_io_proactor_destroy(p);
  }

  return 0;
}

static int reference_entire_surface(void) {
  (void)&hyle_io_last_error;
  (void)&hyle_io_proactor_options_default;
  (void)&hyle_io_proactor_create;
  (void)&hyle_io_proactor_destroy;
  (void)&hyle_io_proactor_valid;
  (void)&hyle_io_proactor_in_flight;
  (void)&hyle_io_proactor_backend;
  (void)&hyle_io_proactor_capabilities;
  (void)&hyle_io_backend_name;
  (void)&hyle_io_has_async_file_data;
  (void)&hyle_io_has_async_file_metadata;
  (void)&hyle_io_submit_read;
  (void)&hyle_io_submit_write;
  (void)&hyle_io_submit_recv;
  (void)&hyle_io_submit_send;
  (void)&hyle_io_submit_accept;
  (void)&hyle_io_submit_accept_multishot;
  (void)&hyle_io_submit_connect;
  (void)&hyle_io_submit_recvfrom;
  (void)&hyle_io_submit_sendto;
  (void)&hyle_io_register_handle;
  (void)&hyle_io_submit_open;
  (void)&hyle_io_submit_close;
  (void)&hyle_io_submit_stat;
  (void)&hyle_io_submit_sync;
  (void)&hyle_io_submit_allocate;
  (void)&hyle_io_submit_cancel;
  (void)&hyle_io_uring_extension_get;
  (void)&hyle_io_uring_register_buffer_ring;
  (void)&hyle_io_uring_register_file_table;
  (void)&hyle_io_uring_assign_fixed_file;
  (void)&hyle_io_uring_release_fixed_file;
  (void)&hyle_io_uring_submit_recv_fixed;
  (void)&hyle_io_uring_submit_send_fixed;
  (void)&hyle_io_uring_submit_accept_direct;
  (void)&hyle_io_uring_submit_close_direct;
  (void)&hyle_io_uring_submit_recv_multishot;
  (void)&hyle_io_uring_buffer_address;
  (void)&hyle_io_uring_buffer_return;
  (void)&hyle_io_reap;
  (void)&hyle_io_post;
  (void)&hyle_io_wake;
  (void)&hyle_io_shared_memory_create;
  (void)&hyle_io_shared_memory_open;
  (void)&hyle_io_shared_memory_create_with_allocator;
  (void)&hyle_io_shared_memory_open_with_allocator;
  (void)&hyle_io_shared_memory_destroy;
  (void)&hyle_io_shared_memory_data;
  (void)&hyle_io_shared_memory_size;
  (void)&hyle_io_shared_memory_flush;
  (void)&hyle_io_shared_memory_unlink;
  (void)&hyle_io_http2_available;
  (void)&hyle_io_http2_session_create;
  (void)&hyle_io_http2_session_destroy;
  (void)&hyle_io_http2_session_valid;
  (void)&hyle_io_http2_feed;
  (void)&hyle_io_http2_next_output;
  (void)&hyle_io_http2_submit_request;
  (void)&hyle_io_http2_submit_response;
  (void)&hyle_io_http2_reset;
  (void)&hyle_io_http2_goaway;
  (void)&hyle_io_http2_next_message;
  (void)&hyle_io_http2_message_header;
  return 1;
}

#if defined(HYLE_IO_HTTP2)
static int shuttle_http2(hyle_io_http2_session* from,
                         hyle_io_http2_session* to) {
  for (;;) {
    const void* bytes = NULL;
    size_t size = hyle_io_http2_next_output(from, &bytes);
    if (size == SIZE_MAX) return 0;
    if (size == 0) return 1;
    if (!hyle_io_http2_feed(to, bytes, size)) return 0;
  }
}
#endif

int main(void) {
  hyle_io_proactor_options options = HYLE_IO_PROACTOR_OPTIONS_INIT;
  hyle_io_proactor* p;
  hyle_io_completion completion = {0, 0, 0, 0};
  int options_result;
  size_t n;

  if (!reference_entire_surface()) return 1;
  options_result = check_proactor_options_contract();
  if (options_result != 0) {
    fprintf(stderr, "proactor options contract failed at line %d: %s\n",
            options_result, hyle_io_last_error());
    return options_result;
  }
  {
    hyle_io_capabilities caps = hyle_io_proactor_capabilities(NULL);
    if (hyle_io_proactor_backend(NULL) != HYLE_IO_BACKEND_NONE ||
        caps.struct_size != sizeof(caps) ||
        caps.backend != HYLE_IO_BACKEND_NONE || caps.bits != 0 ||
        strcmp(hyle_io_backend_name(hyle_io_proactor_backend(NULL)),
               "none") != 0) {
      return 20;
    }
  }
  p = hyle_io_proactor_create(&options);
  if (p == NULL || !hyle_io_proactor_valid(p)) {
    fprintf(stderr, "create: %s\n", hyle_io_last_error());
    return 2;
  }
  if (hyle_io_proactor_backend(p) == HYLE_IO_BACKEND_NONE ||
      strcmp(hyle_io_backend_name(hyle_io_proactor_backend(p)), "none") == 0) {
    hyle_io_proactor_destroy(p);
    return 6;
  }
  {
    hyle_io_capabilities caps = hyle_io_proactor_capabilities(p);
    if (caps.struct_size != sizeof(caps) ||
        caps.backend != hyle_io_proactor_backend(p) ||
        (!!(caps.bits & HYLE_IO_CAP_ASYNC_FILE_METADATA) !=
         !!hyle_io_has_async_file_metadata(p)) ||
        (caps.bits & (HYLE_IO_CAP_NATIVE_MULTISHOT_ACCEPT |
                      HYLE_IO_CAP_EMULATED_MULTISHOT_ACCEPT)) == 0) {
      hyle_io_proactor_destroy(p);
      return 18;
    }
  }
  {
    hyle_io_uring_extension uring = HYLE_IO_URING_EXTENSION_INIT;
    hyle_status status = hyle_io_uring_extension_get(p, &uring);
    if (hyle_io_proactor_backend(p) == HYLE_IO_BACKEND_IO_URING) {
      if (!hyle_status_is_ok(status) || uring.proactor == NULL) {
        hyle_status_free(&status);
        hyle_io_proactor_destroy(p);
        return 19;
      }
    } else {
      if (hyle_status_get_code(status) != HYLE_STATUS_UNIMPLEMENTED ||
          uring.proactor != NULL ||
          hyle_io_uring_release_fixed_file(uring, 0) != 0) {
        hyle_status_free(&status);
        hyle_io_proactor_destroy(p);
        return 19;
      }
    }
    hyle_status_free(&status);
  }
  if (hyle_io_post(p, 0x1234u, 77) != 1) {
    fprintf(stderr, "post: %s\n", hyle_io_last_error());
    hyle_io_proactor_destroy(p);
    return 3;
  }
  n = hyle_io_reap(p, &completion, 1, UINT64_MAX);
  if (n != 1 || completion.user_data != 0x1234u ||
      completion.result != 77 || completion.flags != 0) {
    hyle_io_proactor_destroy(p);
    return 4;
  }
  if (hyle_io_post(p, HYLE_IO_WAKE_TOKEN, 0) != 0 ||
      strstr(hyle_io_last_error(), "reserved") == NULL) {
    hyle_io_proactor_destroy(p);
    return 5;
  }
  if (hyle_io_post(p, 1, -1) != 0 ||
      strstr(hyle_io_last_error(), "32-bit") == NULL ||
      hyle_io_post(p, 1, INT64_MAX) != 0) {
    hyle_io_proactor_destroy(p);
    return 10;
  }
  {
    char byte = 0;
    if (hyle_io_submit_read(p, (intptr_t)-1, &byte, 1, UINT64_MAX, 2) != 0 ||
        strstr(hyle_io_last_error(), "offset") == NULL) {
      hyle_io_proactor_destroy(p);
      return 17;
    }
  }
  hyle_io_proactor_destroy(p);

  {
    const char* name = "hyle_io_c_smoke_20260720";
    hyle_io_shared_memory* owner;
    hyle_io_shared_memory* peer;
    hyle_status cleanup = hyle_io_shared_memory_unlink(name);
    hyle_status_free(&cleanup);
    hyle_allocator backing = {0};
    owner =
        hyle_io_shared_memory_create_with_allocator(name, 4096, backing);
    if (owner == NULL || hyle_io_shared_memory_size(owner) != 4096) return 7;
    memcpy(hyle_io_shared_memory_data(owner), "shared", 7);
    peer = hyle_io_shared_memory_open(name);
    if (peer == NULL ||
        memcmp(hyle_io_shared_memory_data(peer), "shared", 7) != 0) {
      hyle_io_shared_memory_destroy(owner);
      return 8;
    }
    hyle_io_shared_memory_destroy(peer);
    hyle_io_shared_memory_destroy(owner);
    {
      hyle_status unlink_status = hyle_io_shared_memory_unlink(name);
#if defined(_WIN32)
      if (hyle_status_get_code(unlink_status) != HYLE_STATUS_UNIMPLEMENTED) {
        hyle_status_free(&unlink_status);
        return 9;
      }
#else
      if (!hyle_status_is_ok(unlink_status)) {
        hyle_status_free(&unlink_status);
        return 9;
      }
#endif
      hyle_status_free(&unlink_status);
    }
  }
#if defined(HYLE_IO_HTTP2)
  {
    hyle_allocator backing = hyle_system_allocator();
    hyle_io_http2_session* client;
    hyle_io_http2_session* server;
    hyle_io_http2_message message;
    hyle_io_http2_header header = {"x-hyle", 6, "c-abi", 5};
    hyle_io_http2_header decoded;
    int32_t stream_id;

    if (!hyle_io_http2_available()) return 11;
    /* A zero allocator selects HYLE's system allocator, matching the other
     * opaque C factories. */
    client = hyle_io_http2_session_create((hyle_allocator){0},
                                          HYLE_IO_HTTP2_CLIENT);
    server = hyle_io_http2_session_create(backing, HYLE_IO_HTTP2_SERVER);
    if (!hyle_io_http2_session_valid(client) ||
        !hyle_io_http2_session_valid(server))
      return 12;
    if (!shuttle_http2(client, server) ||
        !shuttle_http2(server, client))
      return 13;
    stream_id = hyle_io_http2_submit_request(
        client, "POST", 4, "https", 5, "example.test", 12, "/ffi", 4,
        &header, 1, "request", 7);
    if (stream_id <= 0 || !shuttle_http2(client, server) ||
        hyle_io_http2_next_message(server, &message) != 1 ||
        message.stream_id != stream_id || message.method_size != 4 ||
        memcmp(message.method, "POST", 4) != 0 ||
        message.body_size != 7 ||
        memcmp(message.body, "request", 7) != 0 ||
        !hyle_io_http2_message_header(server, 0, &decoded) ||
        decoded.value_size != 5 ||
        memcmp(decoded.value, "c-abi", 5) != 0)
      return 14;
    if (!hyle_io_http2_submit_response(server, stream_id, 200, NULL, 0,
                                       "response", 8) ||
        !shuttle_http2(server, client) ||
        hyle_io_http2_next_message(client, &message) != 1 ||
        message.stream_id != stream_id || message.status != 200 ||
        message.body_size != 8 ||
        memcmp(message.body, "response", 8) != 0)
      return 15;
    hyle_io_http2_session_destroy(server);
    hyle_io_http2_session_destroy(client);
    if (hyle_io_http2_session_create(backing,
                                     (hyle_io_http2_role)99) != NULL ||
        hyle_io_last_error()[0] == '\0')
      return 16;
  }
#else
  {
    const void* bytes = (const void*)1;
    if (hyle_io_http2_available() ||
        hyle_io_http2_session_create((hyle_allocator){0},
                                     HYLE_IO_HTTP2_CLIENT) != NULL ||
        hyle_io_last_error()[0] == '\0' ||
        hyle_io_http2_next_output(NULL, &bytes) != SIZE_MAX ||
        bytes != NULL)
      return 11;
  }
#endif
  puts("io_c_test: all checks passed");
  return 0;
}
