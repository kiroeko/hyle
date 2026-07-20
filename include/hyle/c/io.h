/* SPDX: same as HYLE.
 *
 * Pure C ABI for HYLE completion-based I/O. The stable seam is an opaque
 * proactor plus caller-owned POD completions; C++ coroutines, STL and backend
 * structs never cross this boundary.
 *
 * Docs: docs/guide/c-io.md
 */
#ifndef HYLE_C_IO_H_
#define HYLE_C_IO_H_

#include <stddef.h>
#include <stdint.h>

#include "hyle/c/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_IO_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_IO_BUILD)
#    define HYLE_IO_C_API __declspec(dllexport)
#  else
#    define HYLE_IO_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_IO_C_API __attribute__((visibility("default")))
#else
#  define HYLE_IO_C_API
#endif

typedef struct hyle_io_proactor_o hyle_io_proactor;
typedef struct hyle_io_shared_memory_o hyle_io_shared_memory;
typedef struct hyle_io_http2_session_o hyle_io_http2_session;

typedef enum hyle_io_backend {
  HYLE_IO_BACKEND_NONE = 0,
  HYLE_IO_BACKEND_IOCP = 1,
  HYLE_IO_BACKEND_IO_URING = 2,
  HYLE_IO_BACKEND_EPOLL = 3,
  HYLE_IO_BACKEND_KQUEUE = 4
} hyle_io_backend;

typedef struct hyle_io_proactor_options {
  uint32_t struct_size;
  uint32_t entries; /* 0 = 256 */
  int defer_taskrun;
  hyle_allocator backing;
} hyle_io_proactor_options;

#define HYLE_IO_PROACTOR_OPTIONS_INIT                                         \
  {(uint32_t)sizeof(hyle_io_proactor_options), 256, 0,                        \
   {NULL, NULL, NULL, NULL}}

typedef struct hyle_io_completion {
  uint64_t user_data;
  int64_t result;
  uint32_t flags;
  uint32_t buffer_id;
} hyle_io_completion;

enum {
  HYLE_IO_COMPLETION_MORE = 1u << 0,
  HYLE_IO_COMPLETION_BUFFER = 1u << 1
};

typedef struct hyle_io_file_stat {
  uint64_t size;
  uint64_t mtime_ns;
  uint32_t mode;
  uint8_t is_dir;
  uint8_t reserved[3];
} hyle_io_file_stat;

enum {
  HYLE_IO_OPEN_READ = 1u << 0,
  HYLE_IO_OPEN_WRITE = 1u << 1,
  HYLE_IO_OPEN_CREATE = 1u << 2,
  HYLE_IO_OPEN_TRUNCATE = 1u << 3,
  HYLE_IO_OPEN_APPEND = 1u << 4
};

/* UINT64_MAX is reserved internally for cross-thread wake events and may not be
 * used as application user_data. */
#define HYLE_IO_WAKE_TOKEN UINT64_MAX
#define HYLE_IO_DATAGRAM_SCRATCH_SIZE 128u

HYLE_IO_C_API const char* hyle_io_last_error(void);
HYLE_IO_C_API hyle_io_proactor_options hyle_io_proactor_options_default(void);
HYLE_IO_C_API hyle_io_proactor* hyle_io_proactor_create(
    const hyle_io_proactor_options* options);
HYLE_IO_C_API void hyle_io_proactor_destroy(hyle_io_proactor* proactor);
HYLE_IO_C_API int hyle_io_proactor_valid(const hyle_io_proactor* proactor);
HYLE_IO_C_API int hyle_io_proactor_in_flight(
    const hyle_io_proactor* proactor);
HYLE_IO_C_API hyle_io_backend hyle_io_proactor_backend(
    const hyle_io_proactor* proactor);
HYLE_IO_C_API const char* hyle_io_backend_name(hyle_io_backend backend);
HYLE_IO_C_API int hyle_io_has_async_file_data(
    const hyle_io_proactor* proactor);
HYLE_IO_C_API int hyle_io_has_async_file_metadata(
    const hyle_io_proactor* proactor);

/* Generic descriptor/socket operations. Every successful submit produces a
 * completion with the same user_data. Buffers, paths, addresses, scratch and
 * metadata outputs must remain alive until that completion is reaped. */
HYLE_IO_C_API int hyle_io_submit_read(hyle_io_proactor* proactor, intptr_t fd,
                                      void* buffer, uint32_t length,
                                      uint64_t offset, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_write(hyle_io_proactor* proactor, intptr_t fd,
                                       const void* buffer, uint32_t length,
                                       uint64_t offset, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_recv(hyle_io_proactor* proactor, intptr_t fd,
                                      void* buffer, uint32_t length,
                                      uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_send(hyle_io_proactor* proactor, intptr_t fd,
                                      const void* buffer, uint32_t length,
                                      uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_accept(hyle_io_proactor* proactor,
                                        intptr_t fd, void* address,
                                        void* address_length,
                                        uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_accept_multishot(
    hyle_io_proactor* proactor, intptr_t fd, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_connect(hyle_io_proactor* proactor,
                                         intptr_t fd, const void* address,
                                         uint32_t address_length,
                                         uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_recvfrom(
    hyle_io_proactor* proactor, intptr_t fd, void* buffer, uint32_t length,
    void* address, uint32_t address_capacity, void* scratch,
    uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_sendto(
    hyle_io_proactor* proactor, intptr_t fd, const void* buffer,
    uint32_t length, const void* address, uint32_t address_length,
    void* scratch, uint64_t user_data);
HYLE_IO_C_API int hyle_io_register_handle(hyle_io_proactor* proactor,
                                          intptr_t fd);

/* File operations. hyle_io_submit_stat writes a neutral POD only when its
 * completion reports success. */
HYLE_IO_C_API int hyle_io_submit_open(hyle_io_proactor* proactor,
                                      const char* path, uint32_t flags,
                                      uint32_t mode, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_close(hyle_io_proactor* proactor,
                                       intptr_t fd, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_stat(hyle_io_proactor* proactor, intptr_t fd,
                                      hyle_io_file_stat* output,
                                      uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_sync(hyle_io_proactor* proactor, intptr_t fd,
                                      int data_only, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_allocate(
    hyle_io_proactor* proactor, intptr_t fd, uint32_t mode, uint64_t offset,
    uint64_t length, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_cancel(hyle_io_proactor* proactor,
                                        uint64_t target_user_data,
                                        uint64_t cancel_user_data);

/* Linux io_uring acceleration. Unsupported backends return 0/-1 without
 * changing the portable one-shot path. */
HYLE_IO_C_API int hyle_io_register_buffer_ring(hyle_io_proactor* proactor,
                                               int group_id,
                                               uint32_t entries,
                                               uint32_t buffer_size);
HYLE_IO_C_API int hyle_io_register_file_table(hyle_io_proactor* proactor,
                                              uint32_t entries);
HYLE_IO_C_API int hyle_io_assign_fixed_file(hyle_io_proactor* proactor,
                                            intptr_t fd);
HYLE_IO_C_API void hyle_io_release_fixed_file(hyle_io_proactor* proactor,
                                              int slot);
HYLE_IO_C_API int hyle_io_submit_recv_fixed(
    hyle_io_proactor* proactor, int slot, void* buffer, uint32_t length,
    uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_send_fixed(
    hyle_io_proactor* proactor, int slot, const void* buffer, uint32_t length,
    uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_accept_direct(hyle_io_proactor* proactor,
                                               intptr_t fd,
                                               uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_close_direct(hyle_io_proactor* proactor,
                                              int slot,
                                              uint64_t user_data);
HYLE_IO_C_API int hyle_io_submit_recv_multishot(
    hyle_io_proactor* proactor, intptr_t fd, int group_id,
    uint64_t user_data);
HYLE_IO_C_API void* hyle_io_buffer_address(hyle_io_proactor* proactor,
                                           int group_id,
                                           uint32_t buffer_id);
HYLE_IO_C_API int hyle_io_buffer_return(hyle_io_proactor* proactor,
                                        int group_id, uint32_t buffer_id);

/* Completion loop. timeout_ns: 0=poll, UINT64_MAX=wait indefinitely.
 * hyle_io_post is owner-thread-only; hyle_io_wake is cross-thread safe. */
HYLE_IO_C_API size_t hyle_io_reap(hyle_io_proactor* proactor,
                                  hyle_io_completion* output, size_t capacity,
                                  uint64_t timeout_ns);
HYLE_IO_C_API int hyle_io_post(hyle_io_proactor* proactor,
                               uint64_t user_data, int64_t result);
HYLE_IO_C_API int hyle_io_wake(hyle_io_proactor* proactor);

/* Named cross-process shared memory. create is exclusive; open attaches to an
 * existing object. Existing mappings survive unlink on POSIX. */
HYLE_IO_C_API hyle_io_shared_memory* hyle_io_shared_memory_create(
    const char* name, size_t size);
HYLE_IO_C_API hyle_io_shared_memory* hyle_io_shared_memory_open(
    const char* name);
HYLE_IO_C_API hyle_io_shared_memory* hyle_io_shared_memory_create_with_allocator(
    const char* name, size_t size, hyle_allocator backing);
HYLE_IO_C_API hyle_io_shared_memory* hyle_io_shared_memory_open_with_allocator(
    const char* name, hyle_allocator backing);
HYLE_IO_C_API void hyle_io_shared_memory_destroy(
    hyle_io_shared_memory* memory);
HYLE_IO_C_API void* hyle_io_shared_memory_data(hyle_io_shared_memory* memory);
HYLE_IO_C_API size_t hyle_io_shared_memory_size(
    const hyle_io_shared_memory* memory);
HYLE_IO_C_API int hyle_io_shared_memory_flush(hyle_io_shared_memory* memory);
HYLE_IO_C_API int hyle_io_shared_memory_unlink(const char* name);

/* HTTP/2 protocol session (io-11).
 *
 * This is deliberately transport-neutral: feed bytes received by any socket/TLS
 * stack into hyle_io_http2_feed(), then send every span returned by
 * hyle_io_http2_next_output(). This keeps C callbacks, coroutines and TLS
 * implementation types out of the ABI while retaining HPACK, multiplexing,
 * stream resets and flow control. Returned byte/message views are borrowed and
 * remain valid only until the next call on the same session.
 */
typedef enum hyle_io_http2_role {
  HYLE_IO_HTTP2_CLIENT = 0,
  HYLE_IO_HTTP2_SERVER = 1
} hyle_io_http2_role;

typedef struct hyle_io_http2_header {
  const char* name;
  size_t name_size;
  const char* value;
  size_t value_size;
} hyle_io_http2_header;

typedef struct hyle_io_http2_message {
  int32_t stream_id;
  int32_t status;
  uint32_t stream_error;
  uint32_t header_count;
  const char* method;
  size_t method_size;
  const char* scheme;
  size_t scheme_size;
  const char* authority;
  size_t authority_size;
  const char* path;
  size_t path_size;
  const void* body;
  size_t body_size;
} hyle_io_http2_message;

/* Returns 1 when this build contains the io-11 protocol core. */
HYLE_IO_C_API int hyle_io_http2_available(void);
HYLE_IO_C_API hyle_io_http2_session* hyle_io_http2_session_create(
    hyle_allocator backing, hyle_io_http2_role role);
HYLE_IO_C_API void hyle_io_http2_session_destroy(
    hyle_io_http2_session* session);
HYLE_IO_C_API int hyle_io_http2_session_valid(
    const hyle_io_http2_session* session);

/* Returns 1 on success, 0 on protocol/resource failure. size==0 is a no-op. */
HYLE_IO_C_API int hyle_io_http2_feed(hyle_io_http2_session* session,
                                     const void* data, size_t size);
/* Returns an encoded span length, 0 when currently blocked/no output, or
 * SIZE_MAX on protocol failure. The span must be consumed before another call
 * on this session. Receiving WINDOW_UPDATE through feed() may unlock more DATA. */
HYLE_IO_C_API size_t hyle_io_http2_next_output(
    hyle_io_http2_session* session, const void** data);

HYLE_IO_C_API int32_t hyle_io_http2_submit_request(
    hyle_io_http2_session* session, const char* method, size_t method_size,
    const char* scheme, size_t scheme_size, const char* authority,
    size_t authority_size, const char* path, size_t path_size,
    const hyle_io_http2_header* headers, size_t header_count,
    const void* body, size_t body_size);
HYLE_IO_C_API int hyle_io_http2_submit_response(
    hyle_io_http2_session* session, int32_t stream_id, int status,
    const hyle_io_http2_header* headers, size_t header_count,
    const void* body, size_t body_size);
HYLE_IO_C_API int hyle_io_http2_reset(hyle_io_http2_session* session,
                                      int32_t stream_id,
                                      uint32_t error_code);
HYLE_IO_C_API int hyle_io_http2_goaway(hyle_io_http2_session* session,
                                       uint32_t error_code);

/* Poll a completed request/response. 1=message, 0=none, -1=fatal session.
 * Header fields are obtained by index from the current borrowed message. */
HYLE_IO_C_API int hyle_io_http2_next_message(
    hyle_io_http2_session* session, hyle_io_http2_message* message);
HYLE_IO_C_API int hyle_io_http2_message_header(
    const hyle_io_http2_session* session, size_t index,
    hyle_io_http2_header* header);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HYLE_C_IO_H_ */
