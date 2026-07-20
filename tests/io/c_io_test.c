#include "hyle/c/io.h"

#include <stdio.h>
#include <string.h>

static int reference_entire_surface(void) {
  (void)&hyle_io_last_error;
  (void)&hyle_io_proactor_options_default;
  (void)&hyle_io_proactor_create;
  (void)&hyle_io_proactor_destroy;
  (void)&hyle_io_proactor_valid;
  (void)&hyle_io_proactor_in_flight;
  (void)&hyle_io_proactor_backend;
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
  (void)&hyle_io_register_buffer_ring;
  (void)&hyle_io_register_file_table;
  (void)&hyle_io_assign_fixed_file;
  (void)&hyle_io_release_fixed_file;
  (void)&hyle_io_submit_recv_fixed;
  (void)&hyle_io_submit_send_fixed;
  (void)&hyle_io_submit_accept_direct;
  (void)&hyle_io_submit_close_direct;
  (void)&hyle_io_submit_recv_multishot;
  (void)&hyle_io_buffer_address;
  (void)&hyle_io_buffer_return;
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
  size_t n;

  if (!reference_entire_surface()) return 1;
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
    (void)hyle_io_shared_memory_unlink(name);
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
    if (!hyle_io_shared_memory_unlink(name)) return 9;
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
