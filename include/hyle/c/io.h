/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
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

#include "hyle/c/error.h"
#include "hyle/c/mem.h"
#include "hyle/c/time_types.h"

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

/* Fixed-width across compilers and -fshort-enums consumers. */
typedef uint32_t hyle_io_backend;
enum {
  HYLE_IO_BACKEND_NONE = 0,
  HYLE_IO_BACKEND_IOCP = 1,
  HYLE_IO_BACKEND_IO_URING = 2,
  HYLE_IO_BACKEND_EPOLL = 3,
  HYLE_IO_BACKEND_KQUEUE = 4
};

/* Capability bits describe the selected backend/extension surface. They do not
 * promise that every operation will succeed on the running kernel: registration
 * can still fail for kernel-version, policy or resource reasons, reported by the
 * operation's result/last_error. */
enum {
  HYLE_IO_CAP_NATIVE_MULTISHOT_ACCEPT = UINT64_C(1) << 0,
  HYLE_IO_CAP_EMULATED_MULTISHOT_ACCEPT = UINT64_C(1) << 1,
  HYLE_IO_CAP_PROVIDED_BUFFER_RING = UINT64_C(1) << 2,
  HYLE_IO_CAP_FIXED_FILES = UINT64_C(1) << 3,
  HYLE_IO_CAP_DIRECT_ACCEPT = UINT64_C(1) << 4,
  HYLE_IO_CAP_ASYNC_FILE_DATA = UINT64_C(1) << 5,
  HYLE_IO_CAP_ASYNC_FILE_METADATA = UINT64_C(1) << 6,
  HYLE_IO_CAP_TLS = UINT64_C(1) << 7,
  HYLE_IO_CAP_HTTP2 = UINT64_C(1) << 8
};

/* Small by-value snapshot with a frozen v1 layout: size 16, backend offset 4,
 * bits offset 8 on the supported 64-bit ABI. struct_size describes the value
 * returned by this library; it is not caller capacity or an extensible input
 * handshake. A future field requires a new query/type rather than appending to
 * this by-value return. */
typedef struct hyle_io_capabilities {
  uint32_t struct_size;
  hyle_io_backend backend;
  uint64_t bits;
} hyle_io_capabilities;

/* Borrowed io_uring-only acceleration handle. Obtain it with
 * hyle_io_uring_extension_get; it becomes invalid when its proactor is
 * destroyed. A zero value is invalid and safe to pass (operations fail). */
typedef struct hyle_io_uring_extension {
  void* proactor;
} hyle_io_uring_extension;

#define HYLE_IO_URING_EXTENSION_INIT {NULL}

typedef struct hyle_io_proactor_options {
  uint32_t struct_size;
  uint32_t reserved; /* must be zero */
  uint32_t entries; /* 0 = 256 */
  uint32_t defer_taskrun; /* 0/1 */
  hyle_allocator backing;
} hyle_io_proactor_options;

#define HYLE_IO_PROACTOR_OPTIONS_REQUIRED_SIZE                                \
  ((uint32_t)offsetof(hyle_io_proactor_options, entries))
#define HYLE_IO_PROACTOR_OPTIONS_INIT                                         \
  {(uint32_t)sizeof(hyle_io_proactor_options), 0, 256, 0,                     \
   {NULL, NULL, NULL, NULL}}

/* `struct_size` is the readable prefix, not an exact-version check. The
 * required prefix is {struct_size,reserved}; omitted entries/defer_taskrun/
 * backing fields use HYLE_IO_PROACTOR_OPTIONS_INIT semantics and unknown
 * future tail bytes are ignored. Optional fields take effect only when the
 * complete field is present. In particular, backing is one atomic allocator
 * value: a prefix ending inside hyle_allocator does not partially override the
 * default allocator. Use the INIT macro or hyle_io_proactor_options_default()
 * for ordinary source code; prefix construction is intended for ABI/FFI
 * bindings. */

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
  hyle_unix_timestamp modified_time;
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
HYLE_IO_C_API hyle_io_capabilities hyle_io_proactor_capabilities(
    const hyle_io_proactor* proactor);
HYLE_IO_C_API const char* hyle_io_backend_name(hyle_io_backend backend);
HYLE_IO_C_API int hyle_io_has_async_file_data(
    const hyle_io_proactor* proactor);
HYLE_IO_C_API int hyle_io_has_async_file_metadata(
    const hyle_io_proactor* proactor);

/* Generic descriptor/socket operations. Every successful submit produces a
 * completion with the same user_data. user_data must be unique among concurrent
 * operations (a multishot operation intentionally reuses its one token).
 * Buffers, paths, addresses, scratch and metadata outputs must remain alive
 * until that completion is reaped. */
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

/* Explicit Linux io_uring acceleration track. Acquisition returns
 * HYLE_STATUS_UNIMPLEMENTED when the selected runtime backend is not io_uring;
 * callers must free a non-OK status. Once acquired, 0/-1 means an actual
 * argument/resource/kernel-version failure, never a silent platform fallback. */
HYLE_IO_C_API hyle_status hyle_io_uring_extension_get(
    hyle_io_proactor* proactor, hyle_io_uring_extension* output);
HYLE_IO_C_API int hyle_io_uring_register_buffer_ring(
    hyle_io_uring_extension extension, int group_id, uint32_t entries,
    uint32_t buffer_size);
HYLE_IO_C_API int hyle_io_uring_register_file_table(
    hyle_io_uring_extension extension, uint32_t entries);
HYLE_IO_C_API int hyle_io_uring_assign_fixed_file(
    hyle_io_uring_extension extension, intptr_t fd);
HYLE_IO_C_API int hyle_io_uring_release_fixed_file(
    hyle_io_uring_extension extension, int slot);
HYLE_IO_C_API int hyle_io_uring_submit_recv_fixed(
    hyle_io_uring_extension extension, int slot, void* buffer,
    uint32_t length, uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_uring_submit_send_fixed(
    hyle_io_uring_extension extension, int slot, const void* buffer,
    uint32_t length, uint32_t flags, uint64_t user_data);
HYLE_IO_C_API int hyle_io_uring_submit_accept_direct(
    hyle_io_uring_extension extension, intptr_t fd, uint64_t user_data);
HYLE_IO_C_API int hyle_io_uring_submit_close_direct(
    hyle_io_uring_extension extension, int slot, uint64_t user_data);
HYLE_IO_C_API int hyle_io_uring_submit_recv_multishot(
    hyle_io_uring_extension extension, intptr_t fd, int group_id,
    uint64_t user_data);
HYLE_IO_C_API void* hyle_io_uring_buffer_address(
    hyle_io_uring_extension extension, int group_id, uint32_t buffer_id);
HYLE_IO_C_API int hyle_io_uring_buffer_return(
    hyle_io_uring_extension extension, int group_id, uint32_t buffer_id);

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
/* POSIX removes the name immediately while existing mappings survive. Windows
 * has no equivalent operation and returns HYLE_STATUS_UNIMPLEMENTED. */
HYLE_IO_C_API hyle_status hyle_io_shared_memory_unlink(const char* name);

/* HTTP/2 protocol session (io-11).
 *
 * This is deliberately transport-neutral: feed bytes received by any socket/TLS
 * stack into hyle_io_http2_feed(), then send every span returned by
 * hyle_io_http2_next_output(). This keeps C callbacks, coroutines and TLS
 * implementation types out of the ABI while retaining HPACK, multiplexing,
 * stream resets and flow control. Returned byte/message views are borrowed and
 * remain valid only until the next call on the same session.
 */
/* Fixed-width for the by-value factory parameter. */
typedef uint32_t hyle_io_http2_role;
enum {
  HYLE_IO_HTTP2_CLIENT = 0,
  HYLE_IO_HTTP2_SERVER = 1
};

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
