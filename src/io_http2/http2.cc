#include "hyle/io/http2.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <coroutine>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

#include <nghttp2/nghttp2.h>

#include "hyle/container/flat_hash_map.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/waker.h"
#include "hyle/util/spin_lock.h"

namespace hyle::io {
namespace detail {

namespace {
constexpr std::size_t kAllocationAlignment = alignof(std::max_align_t);
constexpr std::size_t kAllocationPrefix =
    (sizeof(std::size_t) + kAllocationAlignment - 1) &
    ~(kAllocationAlignment - 1);
constexpr std::size_t kWireBufferBytes = 64 * 1024;
constexpr std::size_t kInputBufferBytes = 64 * 1024;
} // namespace

struct ByteBuffer {
  explicit ByteBuffer(mem::AllocatorRef a) noexcept : allocator(a) {}
  ~ByteBuffer() {
    if (data != nullptr)
      allocator.deallocate(data, capacity, kAllocationAlignment);
  }
  ByteBuffer(const ByteBuffer &) = delete;
  ByteBuffer &operator=(const ByteBuffer &) = delete;

  bool reserve(std::size_t wanted) noexcept {
    if (wanted <= capacity)
      return true;
    std::size_t next = capacity == 0 ? 4096 : capacity;
    while (next < wanted) {
      if (next > (std::numeric_limits<std::size_t>::max)() / 2) {
        next = wanted;
        break;
      }
      next *= 2;
    }
    void *replacement =
        allocator.reallocate(data, capacity, next, kAllocationAlignment);
    if (replacement == nullptr)
      return false;
    data = static_cast<std::uint8_t *>(replacement);
    capacity = next;
    return true;
  }

  bool append(const void *source, std::size_t count,
              std::size_t limit) noexcept {
    if (count > limit || size > limit - count || !reserve(size + count))
      return false;
    if (count != 0)
      std::memcpy(data + size, source, count);
    size += count;
    return true;
  }

  void clear() noexcept { size = 0; }

  mem::AllocatorRef allocator;
  std::uint8_t *data = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;
};

struct FieldRef {
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  bool present = false;
};

struct StoredHeader {
  FieldRef name;
  FieldRef value;
  HttpHeader view{};
};

struct Http2StreamState {
  explicit Http2StreamState(mem::AllocatorRef allocator) noexcept
      : head_bytes(allocator), trailer_bytes(allocator), inbound(allocator) {}

  bool initialize(const Http2Limits &limits) noexcept {
    if (limits.max_header_count == 0)
      return true;
    if (limits.max_header_count >
        (std::numeric_limits<std::size_t>::max)() / sizeof(StoredHeader))
      return false;
    const std::size_t bytes = limits.max_header_count * sizeof(StoredHeader);
    header_slots_bytes = bytes;
    headers = static_cast<StoredHeader *>(
        head_bytes.allocator.allocate(bytes, alignof(StoredHeader)));
    trailers = static_cast<StoredHeader *>(
        head_bytes.allocator.allocate(bytes, alignof(StoredHeader)));
    if (headers == nullptr || trailers == nullptr)
      return false;
    std::uninitialized_value_construct_n(headers, limits.max_header_count);
    std::uninitialized_value_construct_n(trailers, limits.max_header_count);
    return true;
  }

  ~Http2StreamState() {
    if (headers != nullptr) {
      std::destroy_n(headers, header_slots_bytes / sizeof(StoredHeader));
      head_bytes.allocator.deallocate(headers, header_slots_bytes,
                                      alignof(StoredHeader));
    }
    if (trailers != nullptr) {
      std::destroy_n(trailers, header_slots_bytes / sizeof(StoredHeader));
      head_bytes.allocator.deallocate(trailers, header_slots_bytes,
                                      alignof(StoredHeader));
    }
  }

  ByteBuffer head_bytes;
  ByteBuffer trailer_bytes;
  ByteBuffer inbound;
  StoredHeader *headers = nullptr;
  StoredHeader *trailers = nullptr;
  std::size_t header_slots_bytes = 0;
  std::size_t header_count = 0;
  std::size_t trailer_count = 0;
  FieldRef method;
  FieldRef scheme;
  FieldRef authority;
  FieldRef path;

  const std::byte *outbound_data = nullptr;
  std::size_t outbound_size = 0;
  std::size_t outbound_offset = 0;
  std::uint64_t outbound_epoch = 0;
  std::uint64_t encoded_epoch = 0;
  std::uint64_t flushed_epoch = 0;
  std::uint64_t head_wire_epoch = 0;
  std::uint64_t reset_wire_epoch = 0;
  std::uint64_t outbound_total = 0;
  std::optional<std::uint64_t> outbound_expected{};
  nghttp2_nv *trailer_nv = nullptr;
  std::size_t trailer_nv_count = 0;

  Http2StreamState *ready_next = nullptr;
  Http2StreamState *ready_prev = nullptr;
  Http2StreamState *flush_next = nullptr;
  Http2StreamState *garbage_next = nullptr;
  std::uint64_t generation = 1;
  std::uint64_t received_total = 0;
  std::size_t inbound_start = 0;
  std::size_t borrowed_bytes = 0;
  std::int32_t id = 0;
  int status = 0;
  std::uint32_t reset_code = 0;
  bool parsing_trailers = false;
  bool initial_headers_done = false;
  bool head_ready = false;
  bool head_delivered = false;
  bool ready_queued = false;
  bool inbound_end = false;
  bool body_borrowed = false;
  bool reader_active = false;
  bool reader_busy = false;
  bool writer_active = false;
  bool writer_busy = false;
  bool output_submitted = false;
  bool output_finishing = false;
  bool output_done = false;
  bool trailer_submitted = false;
  bool queued_for_flush = false;
  bool garbage_queued = false;
  bool closed = false;
  bool reset_received = false;
  bool local_reset = false;
  std::size_t operation_pins = 0;
  Status failure{};
};

struct Http2SessionState {
  Http2SessionState(mem::AllocatorRef a, HttpTransport byte_transport,
                    Http2Limits configured_limits, bool server_side) noexcept
      : allocator(a), transport(std::move(byte_transport)),
        limits(configured_limits), streams(a), input(a), wire(a),
        goaway_debug(a), server(server_side) {}

  mem::AllocatorRef allocator;
  HttpTransport transport;
  Http2Limits limits;
  container::FlatHashMap<std::int32_t, Http2StreamState *,
                         container::Hash<std::int32_t>, std::equal_to<>,
                         mem::AllocatorRef>
      streams;
  ByteBuffer input;
  ByteBuffer wire;
  ByteBuffer goaway_debug;
  nghttp2_session *session = nullptr;
  nghttp2_mem ngmem{};
  Http2StreamState *ready_head = nullptr;
  Http2StreamState *ready_tail = nullptr;
  Http2StreamState *encoded_head = nullptr;
  Http2StreamState *garbage_head = nullptr;
  std::atomic<std::size_t> references{1};
  util::SpinLock monitor;
  sched::ConditionVariable changed;
  Status fatal_status{};
  std::uint64_t change_epoch = 0;
  std::uint64_t submit_epoch = 0;
  std::uint64_t next_wire_epoch = 0;
  std::uint64_t encoding_wire_epoch = 0;
  std::uint64_t current_wire_epoch = 0;
  std::uint64_t flushed_wire_epoch = 0;
  std::uint64_t goaway_wire_epoch = 0;
  std::size_t wire_offset = 0;
  std::int32_t peer_last_stream_id = 0;
  std::uint32_t peer_error = 0;
  bool server = false;
  bool owner_alive = true;
  bool read_busy = false;
  bool write_busy = false;
  bool peer_closed = false;
  bool peer_sent_goaway = false;
  bool local_draining = false;
  bool closed = false;

  bool initialize() noexcept;
  void destroy() noexcept;
  void retain() noexcept { references.fetch_add(1, std::memory_order_relaxed); }
  void release() noexcept;
  // Protocol state is owner-worker affine. A sibling owner fiber cannot run
  // until the current fiber suspends, by which point ConditionVariable has
  // already published its waiter node. Its own no-waiter fast path therefore
  // supplies all the synchronization needed here.
  void notify_owner() noexcept {
    ++change_epoch;
    changed.notify_all();
  }
  // Cancellation may originate on any thread. Taking the predicate monitor
  // closes the cancel-vs-wait-publication window without charging every normal
  // owner-local protocol transition.
  void notify_cancelled_wait() noexcept {
    std::lock_guard<util::SpinLock> lock(monitor);
    changed.notify_all();
  }
  void fail(Status status) noexcept {
    if (fatal_status.ok())
      fatal_status = std::move(status);
    notify_owner();
  }
  Http2StreamState *find(std::int32_t id) noexcept {
    Http2StreamState **entry = streams.find(id);
    return entry == nullptr ? nullptr : *entry;
  }
  Http2StreamState *create_stream(std::int32_t id) noexcept;
  void queue_ready(Http2StreamState *stream) noexcept;
  Http2StreamState *pop_ready() noexcept;
  Http2StreamState *take_ready(std::int32_t id) noexcept;
  void maybe_retire(Http2StreamState *stream) noexcept;
  void collect_garbage() noexcept;
};

class Http2SessionAccess {
public:
  template <class Awaitable>
  static auto await_on_owner(Awaitable &&operation) noexcept
      -> decltype(operation.await_resume()) {
    if (!operation.await_ready())
      (void)operation.await_suspend(std::noop_coroutine());
    return operation.await_resume();
  }

  template <class T>
  static T run_on_owner(sched::Task<T> task, T allocation_failure) noexcept {
    if (!task.valid())
      return allocation_failure;
    while (!task.done())
      task.resume();
    return task.result();
  }

  static TransferResult read(mem::AllocatorRef allocator,
                             HttpTransport &transport,
                             std::span<std::byte> buffer,
                             OperationOptions options) noexcept {
    if (auto *tcp = std::get_if<TcpStream>(&transport.stream_))
      return await_on_owner(tcp->read_some(buffer, options));
#if defined(HYLE_IO_TLS)
    if (auto *tls = std::get_if<TlsStream>(&transport.stream_)) {
      return run_on_owner(
          tls_read(allocator, *tls, buffer, options),
          TransferResult{0,
                         Status::ResourceExhausted(
                             "HTTP/2 TLS read coroutine allocation failed"),
                         false});
    }
#else
    (void)allocator;
#endif
    return {0, Status::FailedPrecondition("HTTP/2 transport is invalid"),
            false};
  }

  static TransferResult write(mem::AllocatorRef allocator,
                              HttpTransport &transport,
                              std::span<const std::byte> buffer,
                              OperationOptions options) noexcept {
    if (auto *tcp = std::get_if<TcpStream>(&transport.stream_))
      return await_on_owner(tcp->write_some(buffer, options));
#if defined(HYLE_IO_TLS)
    if (auto *tls = std::get_if<TlsStream>(&transport.stream_)) {
      return run_on_owner(
          tls_write(allocator, *tls, buffer, options),
          TransferResult{0,
                         Status::ResourceExhausted(
                             "HTTP/2 TLS write coroutine allocation failed"),
                         false});
    }
#else
    (void)allocator;
#endif
    return {0, Status::FailedPrecondition("HTTP/2 transport is invalid"),
            false};
  }

  static bool supports_immediate_io(const HttpTransport &transport) noexcept {
    return std::holds_alternative<TcpStream>(transport.stream_);
  }

  static Http2SessionState *state(Http2ServerSession &session) noexcept {
    return session.state_;
  }
  static Http2SessionState *state(Http2ClientSession &session) noexcept {
    return session.state_;
  }
  static Http2BodyReader reader(Http2SessionState &session,
                                Http2StreamState &stream) noexcept {
    session.retain();
    stream.reader_active = true;
    return Http2BodyReader(&session, &stream, stream.generation,
                           session.allocator);
  }
  static Http2BodyWriter writer(Http2SessionState &session,
                                Http2StreamState &stream) noexcept {
    session.retain();
    stream.writer_active = true;
    return Http2BodyWriter(&session, &stream, stream.generation,
                           session.allocator);
  }
  static sched::Task<Result<HttpBodyChunk>>
  source_next(mem::AllocatorRef allocator, HttpBodySource source,
              OperationOptions options) noexcept {
    return source.next_(allocator, source.context_, options);
  }
  static Result<HttpBodyChunkView> read_body(mem::AllocatorRef,
                                             Http2BodyReader &, std::size_t,
                                             OperationOptions) noexcept;
  static void release(Http2BodyReader &reader) noexcept;
  static void release(Http2BodyWriter &writer) noexcept;
};

namespace {

void destroy_stream(Http2SessionState &state,
                    Http2StreamState *stream) noexcept {
  if (stream == nullptr)
    return;
  if (stream->trailer_nv != nullptr) {
    state.allocator.deallocate(stream->trailer_nv,
                               stream->trailer_nv_count * sizeof(nghttp2_nv),
                               alignof(nghttp2_nv));
  }
  stream->~Http2StreamState();
  state.allocator.deallocate(stream, sizeof(Http2StreamState),
                             alignof(Http2StreamState));
}

void *h2_malloc(std::size_t size, void *user) noexcept {
  auto *state = static_cast<Http2SessionState *>(user);
  if (size > (std::numeric_limits<std::size_t>::max)() - kAllocationPrefix)
    return nullptr;
  const std::size_t total = size + kAllocationPrefix;
  void *base = state->allocator.allocate(total, kAllocationAlignment);
  if (base == nullptr)
    return nullptr;
  *static_cast<std::size_t *>(base) = total;
  return static_cast<std::uint8_t *>(base) + kAllocationPrefix;
}

void h2_free(void *pointer, void *user) noexcept {
  if (pointer == nullptr)
    return;
  auto *state = static_cast<Http2SessionState *>(user);
  void *base = static_cast<std::uint8_t *>(pointer) - kAllocationPrefix;
  state->allocator.deallocate(base, *static_cast<std::size_t *>(base),
                              kAllocationAlignment);
}

void *h2_calloc(std::size_t count, std::size_t size, void *user) noexcept {
  if (size != 0 && count > (std::numeric_limits<std::size_t>::max)() / size)
    return nullptr;
  const std::size_t bytes = count * size;
  void *result = h2_malloc(bytes, user);
  if (result != nullptr)
    std::memset(result, 0, bytes);
  return result;
}

void *h2_realloc(void *pointer, std::size_t size, void *user) noexcept {
  if (pointer == nullptr)
    return h2_malloc(size, user);
  if (size == 0) {
    h2_free(pointer, user);
    return nullptr;
  }
  if (size > (std::numeric_limits<std::size_t>::max)() - kAllocationPrefix)
    return nullptr;
  auto *state = static_cast<Http2SessionState *>(user);
  void *base = static_cast<std::uint8_t *>(pointer) - kAllocationPrefix;
  const std::size_t old_total = *static_cast<std::size_t *>(base);
  const std::size_t new_total = size + kAllocationPrefix;
  void *replacement = state->allocator.reallocate(base, old_total, new_total,
                                                  kAllocationAlignment);
  if (replacement == nullptr)
    return nullptr;
  *static_cast<std::size_t *>(replacement) = new_total;
  return static_cast<std::uint8_t *>(replacement) + kAllocationPrefix;
}

bool bytes_equal(const std::uint8_t *data, std::size_t size,
                 const char *literal) noexcept {
  const std::size_t literal_size = std::strlen(literal);
  return size == literal_size && std::memcmp(data, literal, size) == 0;
}

FieldRef append_field(ByteBuffer &buffer, const void *data, std::size_t size,
                      std::size_t limit) noexcept {
  FieldRef result;
  if (size > (std::numeric_limits<std::uint32_t>::max)() ||
      buffer.size > (std::numeric_limits<std::uint32_t>::max)() - size)
    return result;
  result.offset = static_cast<std::uint32_t>(buffer.size);
  if (!buffer.append(data, size, limit))
    return {};
  result.length = static_cast<std::uint32_t>(size);
  result.present = true;
  return result;
}

StringView field_view(const ByteBuffer &buffer, FieldRef field) noexcept {
  if (!field.present)
    return {};
  return StringView(reinterpret_cast<const char *>(buffer.data + field.offset),
                    field.length);
}

void materialize_headers(ByteBuffer &bytes, StoredHeader *headers,
                         std::size_t count) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    headers[index].view = HttpHeader{field_view(bytes, headers[index].name),
                                     field_view(bytes, headers[index].value)};
  }
}

Status status_from_nghttp2(int error, const char *operation) noexcept {
  if (error == NGHTTP2_ERR_NOMEM)
    return Status::ResourceExhausted(operation);
  if (error == NGHTTP2_ERR_STREAM_CLOSED)
    return Status::FailedPrecondition(operation);
  if (error == NGHTTP2_ERR_INVALID_ARGUMENT)
    return Status::InvalidArgument(operation);
  return Status::DataLoss(operation);
}

Status stream_reset_status(std::uint32_t code) noexcept {
  switch (static_cast<Http2Error>(code)) {
  case Http2Error::kCancel:
    return Status::Cancelled("HTTP/2 stream reset by peer");
  case Http2Error::kRefusedStream:
    return Status::Unavailable("HTTP/2 stream refused by peer");
  case Http2Error::kEnhanceYourCalm:
    return Status::ResourceExhausted("HTTP/2 peer rejected stream load");
  case Http2Error::kNoError:
    return Status::Unavailable("HTTP/2 stream closed by peer");
  default:
    return Status::DataLoss("HTTP/2 stream reset by peer");
  }
}

void fail_stream(Http2SessionState &state, Http2StreamState &stream,
                 Status status, Http2Error protocol_error) noexcept {
  if (stream.failure.ok())
    stream.failure = std::move(status);
  stream.reset_code = static_cast<std::uint32_t>(protocol_error);
  (void)nghttp2_submit_rst_stream(state.session, NGHTTP2_FLAG_NONE, stream.id,
                                  stream.reset_code);
}

int consume_rejected_data(Http2SessionState &state,
                          std::size_t length) noexcept {
  if (length == 0)
    return 0;
  const int consumed =
      nghttp2_session_consume_connection(state.session, length);
  if (consumed == 0) {
    ++state.submit_epoch;
    state.notify_owner();
    return 0;
  }
  state.fail(status_from_nghttp2(
      consumed, "nghttp2 rejected DATA connection consumption"));
  return NGHTTP2_ERR_CALLBACK_FAILURE;
}

int reject_data(Http2SessionState &state, Http2StreamState &stream,
                std::size_t length, Status status,
                Http2Error protocol_error) noexcept {
  fail_stream(state, stream, std::move(status), protocol_error);
  // Unlike header callbacks, nghttp2 treats every nonzero DATA callback
  // result as a fatal session error. Submit the stream reset ourselves,
  // return connection credit for bytes the application will not retain, and
  // report callback success so an oversized/OOM stream stays isolated.
  return consume_rejected_data(state, length);
}

int on_begin_headers(nghttp2_session *session, const nghttp2_frame *frame,
                     void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  if (frame->hd.type != NGHTTP2_HEADERS)
    return 0;

  const std::int32_t id = frame->hd.stream_id;
  Http2StreamState *stream = state.find(id);
  if (frame->headers.cat == NGHTTP2_HCAT_REQUEST) {
    if (!state.server || stream != nullptr)
      return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    stream = state.create_stream(id);
    if (stream == nullptr)
      return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    nghttp2_session_set_stream_user_data(session, id, stream);
  } else if (stream == nullptr) {
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }

  if (frame->headers.cat == NGHTTP2_HCAT_HEADERS &&
      stream->initial_headers_done) {
    stream->parsing_trailers = true;
    stream->trailer_bytes.clear();
    stream->trailer_count = 0;
  } else {
    stream->parsing_trailers = false;
    if (!state.server && stream->status >= 100 && stream->status < 200) {
      stream->head_bytes.clear();
      stream->header_count = 0;
      stream->status = 0;
    }
  }
  return 0;
}

int on_header(nghttp2_session *, const nghttp2_frame *frame,
              const std::uint8_t *name, std::size_t name_length,
              const std::uint8_t *value, std::size_t value_length, std::uint8_t,
              void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  Http2StreamState *stream = state.find(frame->hd.stream_id);
  if (stream == nullptr)
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  ByteBuffer &bytes =
      stream->parsing_trailers ? stream->trailer_bytes : stream->head_bytes;
  if (name_length > state.limits.max_header_bytes ||
      value_length > state.limits.max_header_bytes - name_length ||
      bytes.size > state.limits.max_header_bytes - name_length - value_length) {
    fail_stream(state, *stream,
                Status::ResourceExhausted("HTTP/2 header bytes exceed limit"),
                Http2Error::kEnhanceYourCalm);
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }

  if (name_length != 0 && name[0] == ':') {
    if (stream->parsing_trailers) {
      fail_stream(state, *stream,
                  Status::DataLoss("HTTP/2 trailers contain pseudo-header"),
                  Http2Error::kProtocol);
      return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    }
    FieldRef field =
        append_field(bytes, value, value_length, state.limits.max_header_bytes);
    if (!field.present) {
      fail_stream(state, *stream,
                  Status::ResourceExhausted("HTTP/2 pseudo-header storage"),
                  Http2Error::kInternal);
      return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    }
    if (bytes_equal(name, name_length, ":method"))
      stream->method = field;
    else if (bytes_equal(name, name_length, ":scheme"))
      stream->scheme = field;
    else if (bytes_equal(name, name_length, ":authority"))
      stream->authority = field;
    else if (bytes_equal(name, name_length, ":path"))
      stream->path = field;
    else if (bytes_equal(name, name_length, ":status")) {
      int status = 0;
      if (value_length != 3)
        return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
      for (std::size_t index = 0; index < value_length; ++index) {
        if (value[index] < '0' || value[index] > '9')
          return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
        status = status * 10 + static_cast<int>(value[index] - '0');
      }
      stream->status = status;
    }
    return 0;
  }

  std::size_t &count =
      stream->parsing_trailers ? stream->trailer_count : stream->header_count;
  StoredHeader *slots =
      stream->parsing_trailers ? stream->trailers : stream->headers;
  if (count >= state.limits.max_header_count) {
    fail_stream(state, *stream,
                Status::ResourceExhausted("HTTP/2 header count exceeds limit"),
                Http2Error::kEnhanceYourCalm);
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }
  const FieldRef stored_name =
      append_field(bytes, name, name_length, state.limits.max_header_bytes);
  const FieldRef stored_value =
      append_field(bytes, value, value_length, state.limits.max_header_bytes);
  if (!stored_name.present || !stored_value.present) {
    fail_stream(state, *stream,
                Status::ResourceExhausted("HTTP/2 header storage"),
                Http2Error::kInternal);
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }
  slots[count].name = stored_name;
  slots[count].value = stored_value;
  ++count;
  return 0;
}

int on_data_chunk(nghttp2_session *, std::uint8_t, std::int32_t stream_id,
                  const std::uint8_t *data, std::size_t length,
                  void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  Http2StreamState *stream = state.find(stream_id);
  if (stream == nullptr)
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  if (!stream->failure.ok() || stream->local_reset)
    return consume_rejected_data(state, length);
  if (length > state.limits.max_body_bytes ||
      stream->received_total > state.limits.max_body_bytes - length) {
    return reject_data(
        state, *stream, length,
        Status::ResourceExhausted("HTTP/2 body exceeds stream limit"),
        Http2Error::kEnhanceYourCalm);
  }
  if (stream->inbound.capacity == 0 &&
      !stream->inbound.reserve(static_cast<std::size_t>(
          (std::min)(state.limits.max_body_bytes,
                     static_cast<std::uint64_t>(
                         (std::max)(state.limits.stream_receive_window,
                                    static_cast<std::uint32_t>(
                                        NGHTTP2_INITIAL_WINDOW_SIZE))))))) {
    return reject_data(
        state, *stream, length,
        Status::ResourceExhausted("HTTP/2 receive window allocation"),
        Http2Error::kInternal);
  }
  if (length > stream->inbound.capacity - stream->inbound.size &&
      stream->inbound_start != 0) {
    const std::size_t remaining = stream->inbound.size - stream->inbound_start;
    if (remaining != 0)
      std::memmove(stream->inbound.data,
                   stream->inbound.data + stream->inbound_start, remaining);
    stream->inbound_start = 0;
    stream->inbound.size = remaining;
  }
  if (length > stream->inbound.capacity - stream->inbound.size) {
    return reject_data(
        state, *stream, length,
        Status::ResourceExhausted("HTTP/2 receive window exhausted"),
        Http2Error::kFlowControl);
  }
  if (length != 0)
    std::memcpy(stream->inbound.data + stream->inbound.size, data, length);
  stream->inbound.size += length;
  stream->received_total += length;
  // The transport read publishes the whole nghttp2 callback batch once after
  // nghttp2_session_mem_recv2() returns.  Notifying here would take the session
  // monitor once per wire fragment even though no observer can run between
  // callbacks on the owner worker.
  return 0;
}

void queue_completed_head(Http2SessionState &state,
                          Http2StreamState &stream) noexcept {
  if (stream.parsing_trailers) {
    materialize_headers(stream.trailer_bytes, stream.trailers,
                        stream.trailer_count);
    return;
  }
  if (!state.server && stream.status >= 100 && stream.status < 200) {
    stream.head_bytes.clear();
    stream.header_count = 0;
    stream.status = 0;
    return;
  }
  materialize_headers(stream.head_bytes, stream.headers, stream.header_count);
  stream.initial_headers_done = true;
  stream.head_ready = true;
  state.queue_ready(&stream);
}

int on_frame_recv(nghttp2_session *, const nghttp2_frame *frame,
                  void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  if (frame->hd.type == NGHTTP2_HEADERS) {
    if (Http2StreamState *stream = state.find(frame->hd.stream_id)) {
      queue_completed_head(state, *stream);
      if ((frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0)
        stream->inbound_end = true;
    }
  } else if (frame->hd.type == NGHTTP2_DATA &&
             (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0) {
    if (Http2StreamState *stream = state.find(frame->hd.stream_id))
      stream->inbound_end = true;
  } else if (frame->hd.type == NGHTTP2_GOAWAY) {
    state.peer_sent_goaway = true;
    state.peer_last_stream_id = frame->goaway.last_stream_id;
    state.peer_error = frame->goaway.error_code;
    state.goaway_debug.clear();
    (void)state.goaway_debug.append(frame->goaway.opaque_data,
                                    frame->goaway.opaque_data_len,
                                    state.limits.max_header_bytes);
  }
  // process_received() publishes this callback batch after nghttp2 returns.
  return 0;
}

int on_frame_send(nghttp2_session *, const nghttp2_frame *frame,
                  void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  const std::uint64_t epoch = state.encoding_wire_epoch;
  if (epoch == 0)
    return 0;

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_REQUEST ||
        frame->headers.cat == NGHTTP2_HCAT_RESPONSE) {
      if (Http2StreamState *stream = state.find(frame->hd.stream_id))
        stream->head_wire_epoch = epoch;
    }
    break;
  case NGHTTP2_RST_STREAM:
    if (Http2StreamState *stream = state.find(frame->hd.stream_id))
      stream->reset_wire_epoch = epoch;
    break;
  case NGHTTP2_GOAWAY:
    state.goaway_wire_epoch = epoch;
    break;
  default:
    break;
  }
  return 0;
}

int on_stream_close(nghttp2_session *, std::int32_t stream_id,
                    std::uint32_t error_code, void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  if (Http2StreamState *stream = state.find(stream_id)) {
    stream->closed = true;
    if (error_code != 0) {
      stream->reset_received = true;
      stream->reset_code = error_code;
      if (stream->failure.ok())
        stream->failure = stream_reset_status(error_code);
    } else if (!stream->inbound_end && stream->failure.ok()) {
      stream->failure =
          Status::Unavailable("HTTP/2 stream closed before END_STREAM");
    }
    if (stream->reset_received || !stream->failure.ok()) {
      // DATA already accepted by nghttp2 still occupies the connection window
      // after a stream is reset. The application can no longer consume those
      // bytes through this reader, so return connection credit here; otherwise
      // one abandoned/reset stream can stall every later stream indefinitely.
      const std::size_t unread = stream->inbound.size - stream->inbound_start;
      if (unread != 0) {
        const int consumed =
            nghttp2_session_consume_connection(state.session, unread);
        if (consumed != 0)
          state.fail(status_from_nghttp2(
              consumed, "nghttp2 reset connection consumption"));
        else
          ++state.submit_epoch;
      }
      stream->inbound_start = 0;
      stream->inbound.size = 0;
      stream->borrowed_bytes = 0;
      stream->body_borrowed = false;
    }
    if (!stream->head_ready) {
      stream->head_ready = true;
      state.queue_ready(stream);
    }
    state.maybe_retire(stream);
  }
  state.notify_owner();
  return 0;
}

nghttp2_ssize read_outbound(nghttp2_session *session, std::int32_t stream_id,
                            std::uint8_t *buffer, std::size_t length,
                            std::uint32_t *data_flags,
                            nghttp2_data_source *source, void *user) noexcept {
  auto &state = *static_cast<Http2SessionState *>(user);
  auto *stream = static_cast<Http2StreamState *>(source->ptr);
  if (stream == nullptr || stream->id != stream_id)
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;

  const std::size_t remaining = stream->outbound_size - stream->outbound_offset;
  const std::size_t take = (std::min)(remaining, length);
  if (take != 0) {
    std::memcpy(buffer, stream->outbound_data + stream->outbound_offset, take);
    stream->outbound_offset += take;
  }

  if (stream->outbound_offset == stream->outbound_size) {
    if (stream->outbound_epoch != 0 &&
        stream->encoded_epoch < stream->outbound_epoch) {
      stream->encoded_epoch = stream->outbound_epoch;
      if (!stream->queued_for_flush) {
        stream->queued_for_flush = true;
        stream->flush_next = state.encoded_head;
        state.encoded_head = stream;
      }
    }
    if (stream->output_finishing) {
      *data_flags |= NGHTTP2_DATA_FLAG_EOF;
      if (stream->trailer_nv_count != 0) {
        *data_flags |= NGHTTP2_DATA_FLAG_NO_END_STREAM;
        if (!stream->trailer_submitted) {
          const int result = nghttp2_submit_trailer(
              session, stream_id, stream->trailer_nv, stream->trailer_nv_count);
          if (result != 0)
            return result;
          stream->trailer_submitted = true;
        }
      }
      stream->output_done = true;
    } else if (take == 0) {
      return NGHTTP2_ERR_DEFERRED;
    }
  }
  return static_cast<nghttp2_ssize>(take);
}

} // namespace

bool Http2SessionState::initialize() noexcept {
  constexpr std::uint32_t kMaximumWindow = 0x7fffffffu;
  if (!allocator.valid() || !transport.valid() ||
      (transport.encrypted() && transport.negotiated_protocol() != "h2") ||
      limits.max_header_bytes == 0 ||
      limits.max_header_bytes > (std::numeric_limits<std::uint32_t>::max)() ||
      limits.max_body_bytes == 0 || limits.max_concurrent_streams == 0 ||
      limits.stream_receive_window == 0 ||
      limits.stream_receive_window > kMaximumWindow ||
      limits.connection_receive_window <
          NGHTTP2_INITIAL_CONNECTION_WINDOW_SIZE ||
      limits.connection_receive_window > kMaximumWindow ||
      limits.read_chunk_bytes == 0 ||
      limits.read_chunk_bytes > limits.stream_receive_window)
    return false;

  if (!streams.try_reserve(
          static_cast<std::size_t>(limits.max_concurrent_streams) * 2) ||
      !input.reserve(kInputBufferBytes) || !wire.reserve(kWireBufferBytes))
    return false;

  nghttp2_session_callbacks *callbacks = nullptr;
  nghttp2_option *option = nullptr;
  if (nghttp2_session_callbacks_new(&callbacks) != 0 ||
      nghttp2_option_new(&option) != 0) {
    if (callbacks != nullptr)
      nghttp2_session_callbacks_del(callbacks);
    if (option != nullptr)
      nghttp2_option_del(option);
    return false;
  }
  nghttp2_session_callbacks_set_on_begin_headers_callback(callbacks,
                                                          on_begin_headers);
  nghttp2_session_callbacks_set_on_header_callback(callbacks, on_header);
  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(callbacks,
                                                            on_data_chunk);
  nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks,
                                                       on_frame_recv);
  nghttp2_session_callbacks_set_on_frame_send_callback(callbacks,
                                                       on_frame_send);
  nghttp2_session_callbacks_set_on_stream_close_callback(callbacks,
                                                         on_stream_close);
  nghttp2_option_set_no_auto_window_update(option, 1);
  nghttp2_option_set_peer_max_concurrent_streams(option,
                                                 limits.max_concurrent_streams);
  nghttp2_option_set_max_send_header_block_length(option,
                                                  limits.max_header_bytes);
  nghttp2_option_set_max_settings(option, 32);
  nghttp2_option_set_max_continuations(option, 16);

  ngmem = nghttp2_mem{this, h2_malloc, h2_free, h2_calloc, h2_realloc};
  const int created = server ? nghttp2_session_server_new3(&session, callbacks,
                                                           this, option, &ngmem)
                             : nghttp2_session_client_new3(
                                   &session, callbacks, this, option, &ngmem);
  nghttp2_option_del(option);
  nghttp2_session_callbacks_del(callbacks);
  if (created != 0)
    return false;

  nghttp2_settings_entry settings[4] = {
      {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, limits.max_concurrent_streams},
      {NGHTTP2_SETTINGS_MAX_HEADER_LIST_SIZE,
       static_cast<std::uint32_t>(limits.max_header_bytes)},
      {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, limits.stream_receive_window},
      {NGHTTP2_SETTINGS_ENABLE_PUSH, 0},
  };
  const std::size_t setting_count = server ? 3 : 4;
  if (nghttp2_submit_settings(session, NGHTTP2_FLAG_NONE, settings,
                              setting_count) != 0 ||
      nghttp2_session_set_local_window_size(
          session, NGHTTP2_FLAG_NONE, 0,
          static_cast<std::int32_t>(limits.connection_receive_window)) != 0) {
    nghttp2_session_del(session);
    session = nullptr;
    return false;
  }
  ++submit_epoch;
  return true;
}

void Http2SessionState::destroy() noexcept {
  if (session != nullptr) {
    nghttp2_session_del(session);
    session = nullptr;
  }
  streams.for_each([&](const std::int32_t &, Http2StreamState *&stream) {
    destroy_stream(*this, stream);
  });
  streams.clear();
  ready_head = ready_tail = encoded_head = garbage_head = nullptr;
}

void Http2SessionState::release() noexcept {
  if (references.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  mem::AllocatorRef saved_allocator = allocator;
  destroy();
  this->~Http2SessionState();
  saved_allocator.deallocate(this, sizeof(Http2SessionState),
                             alignof(Http2SessionState));
}

Http2StreamState *Http2SessionState::create_stream(std::int32_t id) noexcept {
  collect_garbage();
  if (streams.size() >= limits.max_concurrent_streams)
    return nullptr;
  void *storage =
      allocator.allocate(sizeof(Http2StreamState), alignof(Http2StreamState));
  if (storage == nullptr)
    return nullptr;
  auto *stream = new (storage) Http2StreamState(allocator);
  if (!stream->initialize(limits)) {
    destroy_stream(*this, stream);
    return nullptr;
  }
  stream->id = id;
  if (id != 0) {
    const auto inserted = streams.try_insert_or_assign(id, stream);
    if (inserted == decltype(streams)::InsertResult::kOutOfMemory) {
      destroy_stream(*this, stream);
      return nullptr;
    }
  }
  return stream;
}

void Http2SessionState::queue_ready(Http2StreamState *stream) noexcept {
  if (stream == nullptr || stream->ready_queued || stream->head_delivered)
    return;
  stream->ready_queued = true;
  stream->ready_next = nullptr;
  stream->ready_prev = ready_tail;
  if (ready_tail != nullptr)
    ready_tail->ready_next = stream;
  else
    ready_head = stream;
  ready_tail = stream;
}

Http2StreamState *Http2SessionState::pop_ready() noexcept {
  Http2StreamState *stream = ready_head;
  if (stream == nullptr)
    return nullptr;
  ready_head = stream->ready_next;
  if (ready_head == nullptr)
    ready_tail = nullptr;
  else
    ready_head->ready_prev = nullptr;
  stream->ready_next = nullptr;
  stream->ready_prev = nullptr;
  stream->ready_queued = false;
  stream->head_delivered = true;
  return stream;
}

Http2StreamState *Http2SessionState::take_ready(std::int32_t id) noexcept {
  Http2StreamState *current = find(id);
  if (current == nullptr || !current->ready_queued)
    return nullptr;
  if (current->ready_prev != nullptr)
    current->ready_prev->ready_next = current->ready_next;
  else
    ready_head = current->ready_next;
  if (current->ready_next != nullptr)
    current->ready_next->ready_prev = current->ready_prev;
  if (ready_tail == current)
    ready_tail = current->ready_prev;
  current->ready_next = nullptr;
  current->ready_prev = nullptr;
  current->ready_queued = false;
  current->head_delivered = true;
  return current;
}

void Http2SessionState::maybe_retire(Http2StreamState *stream) noexcept {
  if (stream == nullptr || stream->garbage_queued || !stream->closed ||
      stream->reader_active || stream->writer_active || stream->ready_queued ||
      stream->queued_for_flush || stream->operation_pins != 0)
    return;
  stream->garbage_queued = true;
  stream->garbage_next = garbage_head;
  garbage_head = stream;
}

void Http2SessionState::collect_garbage() noexcept {
  while (garbage_head != nullptr) {
    Http2StreamState *stream = garbage_head;
    garbage_head = stream->garbage_next;
    stream->garbage_next = nullptr;
    streams.erase(stream->id);
    destroy_stream(*this, stream);
  }
}

namespace {

Http2SessionState *allocate_session(mem::AllocatorRef allocator,
                                    HttpTransport transport, Http2Limits limits,
                                    bool server) noexcept {
  void *storage =
      allocator.allocate(sizeof(Http2SessionState), alignof(Http2SessionState));
  if (storage == nullptr)
    return nullptr;
  auto *state = new (storage)
      Http2SessionState(allocator, std::move(transport), limits, server);
  if (!state->initialize()) {
    state->destroy();
    state->~Http2SessionState();
    allocator.deallocate(storage, sizeof(Http2SessionState),
                         alignof(Http2SessionState));
    return nullptr;
  }
  return state;
}

Status validate_owner(const Http2SessionState &state) noexcept {
  if (!state.owner_alive || state.closed || state.session == nullptr)
    return Status::FailedPrecondition("HTTP/2 session is closed");
  if (sched::current_worker_index() !=
      static_cast<int>(state.transport.owner_worker()))
    return Status::FailedPrecondition(
        "HTTP/2 operation must run on transport owner worker");
  return Status::Ok();
}

Status check_operation(const Http2SessionState &state,
                       OperationOptions options) noexcept {
  Status owner = validate_owner(state);
  if (!owner.ok())
    return owner;
  if (options.cancel != nullptr && options.cancel->is_cancelled())
    return Status::Cancelled("HTTP/2 operation cancelled");
  if (options.deadline.has_value() &&
      time::remaining(*options.deadline) == time::Duration::zero())
    return Status::DeadlineExceeded("HTTP/2 operation deadline expired");
  return Status::Ok();
}

struct CancelWakeContext {
  Http2SessionState *state = nullptr;
};

void wake_cancelled_wait(void *opaque) noexcept {
  auto *context = static_cast<CancelWakeContext *>(opaque);
  context->state->notify_cancelled_wait();
}

Status wait_for_change(Http2SessionState &state, OperationOptions options,
                       std::uint64_t observed) noexcept {
  auto wait_locked = [&](std::unique_lock<util::SpinLock> &lock) noexcept {
    auto changed = [&]() noexcept {
      return state.change_epoch != observed || !state.fatal_status.ok() ||
             state.peer_closed ||
             (options.cancel != nullptr && options.cancel->is_cancelled());
    };
    if (options.deadline.has_value())
      (void)state.changed.wait_until(lock, *options.deadline, changed);
    else
      state.changed.wait(lock, changed);
  };

  if (options.cancel != nullptr) {
    CancelWakeContext context{&state};
    sched::CancelCallback callback(*options.cancel, wake_cancelled_wait,
                                   &context);
    std::unique_lock<util::SpinLock> lock(state.monitor);
    if (state.change_epoch == observed && state.fatal_status.ok() &&
        !state.peer_closed && !options.cancel->is_cancelled())
      wait_locked(lock);
  } else {
    std::unique_lock<util::SpinLock> lock(state.monitor);
    if (state.change_epoch == observed && state.fatal_status.ok() &&
        !state.peer_closed)
      wait_locked(lock);
  }
  return check_operation(state, options);
}

bool claim_direction(bool &busy) noexcept {
  if (busy)
    return false;
  busy = true;
  return true;
}

void release_direction(Http2SessionState &state, bool &busy,
                       bool notify_waiters = true) noexcept {
  busy = false;
  if (notify_waiters)
    state.notify_owner();
}

struct ProgressResult {
  Status status{};
  bool progressed = false;
};

ProgressResult process_received(Http2SessionState &state,
                                TransferResult received) noexcept {
  bool progressed = false;
  if (received.bytes > state.input.capacity) {
    Status failure =
        Status::Internal("HTTP/2 transport over-reported read progress");
    state.fail(failure);
    return {failure, false};
  }
  if (received.bytes != 0) {
    std::size_t offset = 0;
    while (offset < received.bytes) {
      const nghttp2_ssize used = nghttp2_session_mem_recv2(
          state.session, state.input.data + offset, received.bytes - offset);
      if (used <= 0) {
        Status failure = status_from_nghttp2(static_cast<int>(used),
                                             "nghttp2_session_mem_recv2");
        state.fail(failure);
        return {failure, progressed};
      }
      offset += static_cast<std::size_t>(used);
    }
    state.collect_garbage();
    // One socket read may produce several header/data/frame callbacks.  They
    // execute atomically on the session owner, so expose the resulting state
    // transition and wake parked sibling operations once per read rather than
    // once per callback.
    state.notify_owner();
    progressed = true;
  }
  if (received.eof) {
    state.peer_closed = true;
    state.notify_owner();
  }
  if (!received.ok()) {
    const Status failure = received.status;
    if (!failure.IsCancelled() && !failure.IsDeadlineExceeded())
      state.fail(failure.WithContext("HTTP/2 transport read"));
    return {failure, progressed};
  }
  return {Status::Ok(), progressed || received.eof};
}

ProgressResult try_read_once(Http2SessionState &state) noexcept {
  if (!claim_direction(state.read_busy))
    return {};
  const ImmediateTransferResult attempted = detail::http_transport_try_read(
      state.transport,
      std::span<std::byte>(reinterpret_cast<std::byte *>(state.input.data),
                           state.input.capacity));
  ProgressResult result;
  if (attempted.state == ImmediateTransferState::kReady)
    result = process_received(state, attempted.transfer);
  // The immediate attempt cannot suspend, so no same-owner fiber can observe
  // the temporary claim. If it found no protocol change there is no waiter to
  // wake and no epoch to publish.
  release_direction(state, state.read_busy, result.progressed);
  return result;
}

void complete_wire_frame(Http2SessionState &state) noexcept {
  state.flushed_wire_epoch =
      (std::max)(state.flushed_wire_epoch, state.current_wire_epoch);
  while (state.encoded_head != nullptr) {
    Http2StreamState *stream = state.encoded_head;
    state.encoded_head = stream->flush_next;
    stream->flush_next = nullptr;
    stream->queued_for_flush = false;
    stream->flushed_epoch =
        (std::max)(stream->flushed_epoch, stream->encoded_epoch);
    state.maybe_retire(stream);
  }
  state.collect_garbage();
  state.wire.clear();
  state.wire_offset = 0;
  state.current_wire_epoch = 0;
}

ProgressResult flush_once(mem::AllocatorRef allocator, Http2SessionState &state,
                          OperationOptions options) noexcept {
  if (!claim_direction(state.write_busy))
    return {};

  if (state.wire_offset == state.wire.size) {
    state.wire.clear();
    state.wire_offset = 0;
    state.encoded_head = nullptr;
    const std::uint8_t *bytes = nullptr;
    const std::uint64_t candidate_epoch = state.next_wire_epoch + 1;
    state.encoding_wire_epoch = candidate_epoch;
    const nghttp2_ssize encoded =
        nghttp2_session_mem_send2(state.session, &bytes);
    state.encoding_wire_epoch = 0;
    if (encoded < 0) {
      Status failure = status_from_nghttp2(static_cast<int>(encoded),
                                           "nghttp2_session_mem_send2");
      state.fail(failure);
      release_direction(state, state.write_busy);
      return {failure, false};
    }
    if (encoded == 0) {
      release_direction(state, state.write_busy);
      return {};
    }
    state.next_wire_epoch = candidate_epoch;
    state.current_wire_epoch = candidate_epoch;
    if (!state.wire.append(bytes, static_cast<std::size_t>(encoded),
                           (std::numeric_limits<std::size_t>::max)())) {
      Status failure =
          Status::ResourceExhausted("HTTP/2 wire buffer allocation");
      state.fail(failure);
      release_direction(state, state.write_busy);
      return {failure, false};
    }
  }

  while (state.wire_offset < state.wire.size) {
    auto bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(state.wire.data) +
            state.wire_offset,
        state.wire.size - state.wire_offset);
    TransferResult sent;
    const ImmediateTransferResult attempted =
        detail::http_transport_try_write(state.transport, bytes);
    if (attempted.state == ImmediateTransferState::kReady) {
      sent = attempted.transfer;
    } else if (attempted.state == ImmediateTransferState::kWouldBlock) {
      // Both peers may be backpressured on writes. Drain one already-arrived
      // input chunk before parking this direction.
      ProgressResult received = try_read_once(state);
      if (!received.status.ok()) {
        release_direction(state, state.write_busy);
        return received;
      }
      if (received.progressed) {
        release_direction(state, state.write_busy);
        return {Status::Ok(), true};
      }
      // Give another runnable connection on this owner one bounded chance to
      // make the socket writable before registering a kernel readiness wait.
      // This is an adaptive fiber yield, not a spin: after one retry the normal
      // proactor path remains authoritative.
      sched::Fiber::yield();
      Status after_yield = check_operation(state, options);
      if (!after_yield.ok()) {
        release_direction(state, state.write_busy);
        return {std::move(after_yield), false};
      }
      received = try_read_once(state);
      if (!received.status.ok()) {
        release_direction(state, state.write_busy);
        return received;
      }
      if (received.progressed) {
        release_direction(state, state.write_busy);
        return {Status::Ok(), true};
      }
      const ImmediateTransferResult retried =
          detail::http_transport_try_write(state.transport, bytes);
      sent = retried.state == ImmediateTransferState::kReady
                 ? retried.transfer
                 : Http2SessionAccess::write(allocator, state.transport, bytes,
                                             options);
    } else {
      sent =
          Http2SessionAccess::write(allocator, state.transport, bytes, options);
    }
    if (sent.bytes > bytes.size()) {
      Status failure =
          Status::Internal("HTTP/2 transport over-reported write progress");
      state.fail(failure);
      release_direction(state, state.write_busy);
      return {failure, false};
    }
    state.wire_offset += sent.bytes;
    if (!sent.ok()) {
      const Status failure = sent.status;
      if (state.wire_offset == state.wire.size)
        complete_wire_frame(state);
      if (!failure.IsCancelled() && !failure.IsDeadlineExceeded())
        state.fail(failure.WithContext("HTTP/2 transport write"));
      release_direction(state, state.write_busy);
      return {failure, sent.bytes != 0};
    }
    if (sent.bytes == 0) {
      Status failure = Status::Unavailable("HTTP/2 transport made no progress");
      state.fail(failure);
      release_direction(state, state.write_busy);
      return {failure, false};
    }
  }

  complete_wire_frame(state);
  release_direction(state, state.write_busy);
  return {Status::Ok(), true};
}

ProgressResult read_once(mem::AllocatorRef allocator, Http2SessionState &state,
                         OperationOptions options) noexcept {
  if (!claim_direction(state.read_busy))
    return {};

  TransferResult received = Http2SessionAccess::read(
      allocator, state.transport,
      std::span<std::byte>(reinterpret_cast<std::byte *>(state.input.data),
                           state.input.capacity),
      options);
  ProgressResult result = process_received(state, received);
  release_direction(state, state.read_busy);
  return result;
}

template <class Ready>
Status drive_until(mem::AllocatorRef allocator, Http2SessionState &state,
                   Ready ready, OperationOptions options) noexcept {
  for (;;) {
    if (ready())
      return Status::Ok();
    Status operation = check_operation(state, options);
    if (!operation.ok())
      return operation;
    if (!state.fatal_status.ok())
      return state.fatal_status;
    if (state.peer_closed)
      return Status::Unavailable("HTTP/2 peer closed connection");

    ProgressResult flushed = flush_once(allocator, state, options);
    if (ready())
      return Status::Ok();
    if (!flushed.status.ok())
      return flushed.status;
    if (flushed.progressed)
      continue;

    // Most protocol turns arrive while the owner fiber is already running
    // (for example a peer WINDOW_UPDATE following the write above).  Consume
    // queued bytes with the allocation-free nonblocking seam before paying for
    // a proactor submission and fiber suspension.  A miss still falls through
    // to the ordinary readiness-backed asynchronous read below.
    ProgressResult immediate = try_read_once(state);
    if (ready())
      return Status::Ok();
    if (!immediate.status.ok())
      return immediate.status;
    if (immediate.progressed)
      continue;

    // As above, one cooperative turn lets other ready connections publish
    // bytes without paying a proactor submit/reap cycle.  A second immediate
    // miss falls through to the real readiness wait, so an idle worker cannot
    // busy-loop.
    if (Http2SessionAccess::supports_immediate_io(state.transport)) {
      sched::Fiber::yield();
      if (ready())
        return Status::Ok();
      Status after_yield = check_operation(state, options);
      if (!after_yield.ok())
        return after_yield;
      immediate = try_read_once(state);
      if (ready())
        return Status::Ok();
      if (!immediate.status.ok())
        return immediate.status;
      if (immediate.progressed)
        continue;
    }

    ProgressResult received = read_once(allocator, state, options);
    if (ready())
      return Status::Ok();
    if (!received.status.ok())
      return received.status;
    if (received.progressed)
      continue;

    const std::uint64_t observed = state.change_epoch;
    Status waited = wait_for_change(state, options, observed);
    if (!waited.ok())
      return waited;
  }
}

bool valid_string(StringView value) noexcept {
  return value.empty() || value.data() != nullptr;
}

bool ascii_iequal(StringView left, const char *right) noexcept {
  const std::size_t right_size = std::strlen(right);
  if (left.size() != right_size)
    return false;
  for (std::size_t index = 0; index < right_size; ++index) {
    unsigned char character =
        static_cast<unsigned char>(left[static_cast<std::uint32_t>(index)]);
    if (character >= 'A' && character <= 'Z')
      character = static_cast<unsigned char>(character + ('a' - 'A'));
    if (character != static_cast<unsigned char>(right[index]))
      return false;
  }
  return true;
}

Status validate_headers(const Http2SessionState &state,
                        std::span<const HttpHeader> headers,
                        bool generated_content_length) noexcept {
  if (headers.size() > state.limits.max_header_count)
    return Status::ResourceExhausted("HTTP/2 header count exceeds limit");
  std::size_t bytes = 0;
  for (const HttpHeader &header : headers) {
    if (!valid_string(header.name) || !valid_string(header.value) ||
        header.name.empty() || header.name[0] == ':')
      return Status::InvalidArgument("invalid HTTP/2 regular header");
    if (header.name.size() >
            state.limits.max_header_bytes -
                (std::min)(bytes, state.limits.max_header_bytes) ||
        header.value.size() >
            state.limits.max_header_bytes - bytes - header.name.size())
      return Status::ResourceExhausted("HTTP/2 header bytes exceed limit");
    bytes += header.name.size() + header.value.size();
    if (ascii_iequal(header.name, "connection") ||
        ascii_iequal(header.name, "proxy-connection") ||
        ascii_iequal(header.name, "keep-alive") ||
        ascii_iequal(header.name, "upgrade") ||
        ascii_iequal(header.name, "transfer-encoding"))
      return Status::InvalidArgument(
          "connection-specific header is forbidden in HTTP/2");
    if (ascii_iequal(header.name, "te") &&
        !ascii_iequal(header.value, "trailers"))
      return Status::InvalidArgument(
          "HTTP/2 TE header may contain only trailers");
    if (ascii_iequal(header.name, "content-length"))
      return Status::InvalidArgument(
          generated_content_length
              ? "duplicate HTTP/2 content-length owner"
              : "use the typed HTTP/2 content-length parameter");
  }
  return Status::Ok();
}

nghttp2_nv make_nv(StringView name, StringView value) noexcept {
  return nghttp2_nv{
      reinterpret_cast<std::uint8_t *>(const_cast<char *>(name.data())),
      reinterpret_cast<std::uint8_t *>(const_cast<char *>(value.data())),
      name.size(), value.size(), NGHTTP2_NV_FLAG_NONE};
}

struct NvArray {
  mem::AllocatorRef allocator{};
  nghttp2_nv *data = nullptr;
  std::size_t count = 0;
  NvArray() noexcept = default;
  NvArray(NvArray &&other) noexcept
      : allocator(other.allocator), data(std::exchange(other.data, nullptr)),
        count(std::exchange(other.count, 0)) {}
  NvArray &operator=(NvArray &&other) noexcept {
    if (this == &other)
      return *this;
    if (data != nullptr)
      allocator.deallocate(data, count * sizeof(nghttp2_nv),
                           alignof(nghttp2_nv));
    allocator = other.allocator;
    data = std::exchange(other.data, nullptr);
    count = std::exchange(other.count, 0);
    return *this;
  }
  NvArray(const NvArray &) = delete;
  NvArray &operator=(const NvArray &) = delete;
  ~NvArray() {
    if (data != nullptr)
      allocator.deallocate(data, count * sizeof(nghttp2_nv),
                           alignof(nghttp2_nv));
  }
};

Result<NvArray> allocate_nv(mem::AllocatorRef allocator,
                            std::size_t count) noexcept {
  NvArray result;
  result.allocator = allocator;
  result.count = count;
  if (count == 0)
    return result;
  if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(nghttp2_nv))
    return make_unexpected(Status::ResourceExhausted("HTTP/2 header array"));
  result.data = static_cast<nghttp2_nv *>(
      allocator.allocate(count * sizeof(nghttp2_nv), alignof(nghttp2_nv)));
  if (result.data == nullptr)
    return make_unexpected(Status::ResourceExhausted("HTTP/2 header array"));
  return result;
}

Status prepare_trailers(Http2SessionState &state, Http2StreamState &stream,
                        std::span<const HttpHeader> trailers) noexcept {
  Status valid = validate_headers(state, trailers, false);
  if (!valid.ok())
    return valid;
  if (trailers.empty())
    return Status::Ok();
  const std::size_t bytes = trailers.size() * sizeof(nghttp2_nv);
  stream.trailer_nv = static_cast<nghttp2_nv *>(
      state.allocator.allocate(bytes, alignof(nghttp2_nv)));
  if (stream.trailer_nv == nullptr)
    return Status::ResourceExhausted("HTTP/2 trailer array");
  stream.trailer_nv_count = trailers.size();
  for (std::size_t index = 0; index < trailers.size(); ++index)
    stream.trailer_nv[index] =
        make_nv(trailers[index].name, trailers[index].value);
  return Status::Ok();
}

} // namespace

namespace {

Result<std::uint64_t>
prepare_content_length(std::optional<std::uint64_t> length, char (&buffer)[32],
                       StringView *view) noexcept {
  if (!length.has_value()) {
    *view = {};
    return 0;
  }
  auto converted = std::to_chars(buffer, buffer + sizeof(buffer), *length);
  if (converted.ec != std::errc{})
    return make_unexpected(
        Status::Internal("failed to format HTTP/2 content length"));
  *view =
      StringView(buffer, static_cast<std::uint32_t>(converted.ptr - buffer));
  return *length;
}

Result<Http2StreamState *>
submit_request_head(Http2SessionState &state, const Http2RequestHead &head,
                    std::optional<std::uint64_t> content_length) noexcept {
  if (state.server)
    return make_unexpected(
        Status::FailedPrecondition("request submitted on server session"));
  if (state.local_draining || state.peer_sent_goaway)
    return make_unexpected(Status::Unavailable("HTTP/2 session is draining"));
  if (!valid_string(head.method) || !valid_string(head.scheme) ||
      !valid_string(head.authority) || !valid_string(head.path) ||
      head.method.empty() || head.scheme.empty() || head.authority.empty() ||
      head.path.empty())
    return make_unexpected(Status::InvalidArgument(
        "HTTP/2 request requires method, scheme, authority and path"));
  if (content_length.has_value() &&
      *content_length > state.limits.max_body_bytes)
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 request body exceeds limit"));
  Status headers =
      validate_headers(state, head.headers, content_length.has_value());
  if (!headers.ok())
    return make_unexpected(std::move(headers));
  if (state.streams.size() >= state.limits.max_concurrent_streams ||
      nghttp2_session_check_request_allowed(state.session) == 0)
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 concurrent stream limit"));

  Http2StreamState *stream = state.create_stream(0);
  if (stream == nullptr)
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 stream allocation"));

  const std::size_t field_count =
      4 + head.headers.size() + (content_length.has_value() ? 1 : 0);
  Result<NvArray> allocated = allocate_nv(state.allocator, field_count);
  if (!allocated.has_value()) {
    destroy_stream(state, stream);
    return make_unexpected(std::move(allocated).error());
  }
  NvArray fields = std::move(*allocated);
  fields.data[0] = make_nv(StringView(":method", 7), head.method);
  fields.data[1] = make_nv(StringView(":scheme", 7), head.scheme);
  fields.data[2] = make_nv(StringView(":authority", 10), head.authority);
  fields.data[3] = make_nv(StringView(":path", 5), head.path);
  std::size_t next = 4;
  for (const HttpHeader &header : head.headers)
    fields.data[next++] = make_nv(header.name, header.value);
  char length_text[32]{};
  StringView length_view;
  Result<std::uint64_t> prepared =
      prepare_content_length(content_length, length_text, &length_view);
  if (!prepared.has_value()) {
    destroy_stream(state, stream);
    return make_unexpected(std::move(prepared).error());
  }
  if (content_length.has_value())
    fields.data[next++] =
        make_nv(StringView("content-length", 14), length_view);

  nghttp2_data_provider2 provider{};
  provider.source.ptr = stream;
  provider.read_callback = read_outbound;
  const std::int32_t id = nghttp2_submit_request2(
      state.session, nullptr, fields.data, fields.count, &provider, stream);
  if (id < 0) {
    destroy_stream(state, stream);
    return make_unexpected(status_from_nghttp2(id, "nghttp2_submit_request2"));
  }
  stream->id = id;
  stream->output_submitted = true;
  stream->outbound_expected = content_length;
  const auto inserted = state.streams.try_insert_or_assign(id, stream);
  if (inserted == decltype(state.streams)::InsertResult::kOutOfMemory) {
    (void)nghttp2_submit_rst_stream(state.session, NGHTTP2_FLAG_NONE, id,
                                    NGHTTP2_INTERNAL_ERROR);
    nghttp2_session_set_stream_user_data(state.session, id, nullptr);
    destroy_stream(state, stream);
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 stream index allocation"));
  }
  ++state.submit_epoch;
  state.notify_owner();
  return stream;
}

Status
submit_response_head(Http2SessionState &state, Http2StreamState &stream,
                     const Http2ResponseHead &head,
                     std::optional<std::uint64_t> content_length) noexcept {
  if (!state.server)
    return Status::FailedPrecondition("response submitted on client session");
  if (stream.output_submitted)
    return Status::FailedPrecondition("HTTP/2 response already started");
  if (head.status < 100 || head.status > 999)
    return Status::InvalidArgument("invalid HTTP/2 response status");
  if (content_length.has_value() &&
      *content_length > state.limits.max_body_bytes)
    return Status::ResourceExhausted("HTTP/2 response body exceeds limit");
  Status headers =
      validate_headers(state, head.headers, content_length.has_value());
  if (!headers.ok())
    return headers;

  const std::size_t field_count =
      1 + head.headers.size() + (content_length.has_value() ? 1 : 0);
  Result<NvArray> allocated = allocate_nv(state.allocator, field_count);
  if (!allocated.has_value())
    return std::move(allocated).error();
  NvArray fields = std::move(*allocated);
  char status_text[4] = {static_cast<char>('0' + head.status / 100),
                         static_cast<char>('0' + (head.status / 10) % 10),
                         static_cast<char>('0' + head.status % 10), '\0'};
  fields.data[0] =
      make_nv(StringView(":status", 7), StringView(status_text, 3));
  std::size_t next = 1;
  for (const HttpHeader &header : head.headers)
    fields.data[next++] = make_nv(header.name, header.value);
  char length_text[32]{};
  StringView length_view;
  Result<std::uint64_t> prepared =
      prepare_content_length(content_length, length_text, &length_view);
  if (!prepared.has_value())
    return std::move(prepared).error();
  if (content_length.has_value())
    fields.data[next++] =
        make_nv(StringView("content-length", 14), length_view);

  nghttp2_data_provider2 provider{};
  provider.source.ptr = &stream;
  provider.read_callback = read_outbound;
  const int submitted = nghttp2_submit_response2(
      state.session, stream.id, fields.data, fields.count, &provider);
  if (submitted != 0)
    return status_from_nghttp2(submitted, "nghttp2_submit_response2");
  stream.output_submitted = true;
  stream.outbound_expected = content_length;
  ++state.submit_epoch;
  state.notify_owner();
  return Status::Ok();
}

HttpHeadersView regular_headers(Http2StreamState &stream) noexcept {
  return HttpHeadersView(std::span<const HttpHeader>(
      stream.header_count == 0 ? nullptr : &stream.headers[0].view,
      stream.header_count));
}

HttpHeadersView trailer_headers(Http2StreamState &stream) noexcept {
  return HttpHeadersView(std::span<const HttpHeader>(
      stream.trailer_count == 0 ? nullptr : &stream.trailers[0].view,
      stream.trailer_count));
}

std::optional<Http2Error> reset_error(const Http2StreamState &stream) noexcept {
  return stream.reset_received || stream.local_reset
             ? std::optional<Http2Error>(
                   static_cast<Http2Error>(stream.reset_code))
             : std::nullopt;
}

Http2IncomingRequest make_request(Http2SessionState &state,
                                  Http2StreamState &stream) noexcept {
  const auto id = *Http2StreamId::from_value(stream.id);
  return Http2IncomingRequest{
      id,
      Http2RequestView(field_view(stream.head_bytes, stream.method),
                       field_view(stream.head_bytes, stream.scheme),
                       field_view(stream.head_bytes, stream.authority),
                       field_view(stream.head_bytes, stream.path),
                       regular_headers(stream)),
      reset_error(stream), Http2SessionAccess::reader(state, stream)};
}

Http2IncomingResponse make_response(Http2SessionState &state,
                                    Http2StreamState &stream) noexcept {
  const auto id = *Http2StreamId::from_value(stream.id);
  return Http2IncomingResponse{
      id, Http2ResponseView(stream.status, regular_headers(stream)),
      reset_error(stream), Http2SessionAccess::reader(state, stream)};
}

} // namespace

void Http2SessionAccess::release(Http2BodyReader &reader) noexcept {
  if (reader.session_ == nullptr)
    return;
  Http2SessionState *state = reader.session_;
  Http2StreamState *stream = reader.stream_;
  if (stream != nullptr && stream->generation == reader.generation_) {
    const std::size_t unread = stream->inbound.size - stream->inbound_start;
    const std::size_t delivered =
        stream->body_borrowed ? stream->borrowed_bytes : 0;
    const bool body_fully_delivered =
        stream->inbound_end &&
        stream->inbound_start + delivered == stream->inbound.size;
    if (unread != 0 && state->session != nullptr) {
      if (!stream->closed)
        (void)nghttp2_session_consume_stream(state->session, stream->id,
                                             unread);
      (void)nghttp2_session_consume_connection(state->session, unread);
      ++state->submit_epoch;
    }
    stream->inbound_start = 0;
    stream->inbound.size = 0;
    if (!body_fully_delivered && !stream->closed && state->session != nullptr) {
      stream->local_reset = true;
      stream->reset_code = static_cast<std::uint32_t>(Http2Error::kCancel);
      (void)nghttp2_submit_rst_stream(state->session, NGHTTP2_FLAG_NONE,
                                      stream->id, stream->reset_code);
      ++state->submit_epoch;
    }
    stream->reader_active = false;
    stream->reader_busy = false;
    stream->body_borrowed = false;
    stream->borrowed_bytes = 0;
    state->maybe_retire(stream);
    state->collect_garbage();
  }
  reader.session_ = nullptr;
  reader.stream_ = nullptr;
  state->notify_owner();
  state->release();
}

void Http2SessionAccess::release(Http2BodyWriter &writer) noexcept {
  if (writer.session_ == nullptr)
    return;
  Http2SessionState *state = writer.session_;
  Http2StreamState *stream = writer.stream_;
  if (stream != nullptr && stream->generation == writer.generation_) {
    if (!stream->output_done && !stream->closed && state->session != nullptr) {
      stream->local_reset = true;
      stream->reset_code = static_cast<std::uint32_t>(Http2Error::kCancel);
      (void)nghttp2_submit_rst_stream(state->session, NGHTTP2_FLAG_NONE,
                                      stream->id, stream->reset_code);
      ++state->submit_epoch;
    }
    stream->outbound_data = nullptr;
    stream->outbound_size = stream->outbound_offset = 0;
    stream->writer_active = false;
    stream->writer_busy = false;
    state->maybe_retire(stream);
    state->collect_garbage();
  }
  writer.session_ = nullptr;
  writer.stream_ = nullptr;
  state->notify_owner();
  state->release();
}

namespace {

Status acknowledge_body(Http2SessionState &state,
                        Http2StreamState &stream) noexcept {
  if (!stream.body_borrowed)
    return Status::Ok();
  const std::size_t consumed = stream.borrowed_bytes;
  if (consumed != 0) {
    const int stream_result =
        nghttp2_session_consume_stream(state.session, stream.id, consumed);
    const int connection_result =
        nghttp2_session_consume_connection(state.session, consumed);
    if (stream_result != 0 || connection_result != 0)
      return status_from_nghttp2(stream_result != 0 ? stream_result
                                                    : connection_result,
                                 "nghttp2 receive consumption");
    ++state.submit_epoch;
    stream.inbound_start += consumed;
    if (stream.inbound_start == stream.inbound.size) {
      stream.inbound_start = 0;
      stream.inbound.size = 0;
    }
  }
  stream.borrowed_bytes = 0;
  stream.body_borrowed = false;
  state.notify_owner();
  return Status::Ok();
}

Status reset_stream(Http2SessionState &state, Http2StreamState &stream,
                    Http2Error error) noexcept {
  if (stream.closed)
    return Status::Ok();
  const int submitted =
      nghttp2_submit_rst_stream(state.session, NGHTTP2_FLAG_NONE, stream.id,
                                static_cast<std::uint32_t>(error));
  if (submitted != 0)
    return status_from_nghttp2(submitted, "nghttp2_submit_rst_stream");
  stream.local_reset = true;
  stream.reset_code = static_cast<std::uint32_t>(error);
  stream.outbound_data = nullptr;
  stream.outbound_size = stream.outbound_offset = 0;
  ++state.submit_epoch;
  state.notify_owner();
  return Status::Ok();
}

} // namespace

} // namespace detail

Http2BodyReader::~Http2BodyReader() { abandon_(); }

Http2BodyReader::Http2BodyReader(Http2BodyReader &&other) noexcept
    : session_(std::exchange(other.session_, nullptr)),
      stream_(std::exchange(other.stream_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      allocator_(other.allocator_) {}

Http2BodyReader &Http2BodyReader::operator=(Http2BodyReader &&other) noexcept {
  if (this == &other)
    return *this;
  abandon_();
  session_ = std::exchange(other.session_, nullptr);
  stream_ = std::exchange(other.stream_, nullptr);
  generation_ = std::exchange(other.generation_, 0);
  allocator_ = other.allocator_;
  return *this;
}

bool Http2BodyReader::valid() const noexcept {
  return session_ != nullptr && session_->owner_alive && stream_ != nullptr &&
         stream_->generation == generation_ && stream_->reader_active;
}

bool Http2BodyReader::done() const noexcept {
  if (!valid() || !stream_->inbound_end)
    return false;
  const std::size_t exposed =
      stream_->body_borrowed ? stream_->borrowed_bytes : 0;
  return stream_->inbound_start + exposed == stream_->inbound.size;
}

std::optional<Http2Error> Http2BodyReader::stream_error() const noexcept {
  return valid() ? detail::reset_error(*stream_) : std::nullopt;
}

void Http2BodyReader::abandon_() noexcept {
  detail::Http2SessionAccess::release(*this);
}

Result<HttpBodyChunkView> detail::Http2SessionAccess::read_body(
    mem::AllocatorRef allocator, Http2BodyReader &reader, std::size_t max_bytes,
    OperationOptions options) noexcept {
  if (!reader.valid())
    return make_unexpected(
        Status::FailedPrecondition("invalid HTTP/2 body reader"));
  auto &state = *reader.session_;
  auto &stream = *reader.stream_;
  Status operation = detail::check_operation(state, options);
  if (!operation.ok())
    return make_unexpected(std::move(operation));
  if (stream.reader_busy)
    return make_unexpected(Status::FailedPrecondition(
        "HTTP/2 body reader already has an operation in flight"));
  stream.reader_busy = true;
  Status acknowledged = detail::acknowledge_body(state, stream);
  if (!acknowledged.ok()) {
    stream.reader_busy = false;
    return make_unexpected(std::move(acknowledged));
  }
  const std::size_t limit =
      max_bytes == 0 ? state.limits.read_chunk_bytes : max_bytes;
  if (limit == 0) {
    stream.reader_busy = false;
    return make_unexpected(
        Status::InvalidArgument("HTTP/2 read size must be nonzero"));
  }

  auto ready = [&]() noexcept {
    return stream.inbound_start != stream.inbound.size || stream.inbound_end ||
           stream.closed || stream.reset_received || stream.local_reset ||
           !stream.failure.ok();
  };
  Status driven = detail::drive_until(allocator, state, ready, options);
  if (!driven.ok()) {
    stream.reader_busy = false;
    return make_unexpected(std::move(driven));
  }
  if (!stream.failure.ok()) {
    stream.reader_busy = false;
    return make_unexpected(stream.failure);
  }
  if (stream.reset_received) {
    stream.reader_busy = false;
    return make_unexpected(detail::stream_reset_status(stream.reset_code));
  }
  if (stream.local_reset || (stream.closed && !stream.inbound_end)) {
    stream.reader_busy = false;
    return make_unexpected(detail::stream_reset_status(stream.reset_code));
  }

  const std::size_t available = stream.inbound.size - stream.inbound_start;
  if (available == 0) {
    stream.reader_busy = false;
    return HttpBodyChunkView{{}, detail::trailer_headers(stream), true};
  }
  const std::size_t count = (std::min)(available, limit);
  stream.body_borrowed = true;
  stream.borrowed_bytes = count;
  stream.reader_busy = false;
  return HttpBodyChunkView{
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(stream.inbound.data) +
              stream.inbound_start,
          count),
      stream.inbound_end && count == available ? detail::trailer_headers(stream)
                                               : HttpHeadersView{},
      stream.inbound_end && count == available};
}

sched::Task<Result<HttpBodyChunkView>>
http2_body_read(mem::AllocatorRef allocator, Http2BodyReader &reader,
                std::size_t max_bytes, OperationOptions options) noexcept {
  co_return detail::Http2SessionAccess::read_body(allocator, reader, max_bytes,
                                                  options);
}

sched::Task<Status>
detail::http2_body_consume(mem::AllocatorRef allocator, Http2BodyReader &reader,
                           void *sink_state, Http2BodySinkFn sink,
                           std::size_t max_bytes,
                           OperationOptions options) noexcept {
  if (sink_state == nullptr || sink == nullptr)
    co_return Status::InvalidArgument("HTTP/2 body sink is invalid");
  for (;;) {
    Result<HttpBodyChunkView> chunk =
        Http2SessionAccess::read_body(allocator, reader, max_bytes, options);
    if (!chunk.has_value())
      co_return std::move(chunk).error();
    Status accepted = sink(sink_state, *chunk);
    if (!accepted.ok()) {
      // The sink rejected bytes that cannot be offered again after the next
      // reader operation acknowledges the borrowed view.  End this stream
      // instead of leaving a seemingly retryable reader that would skip them.
      Http2SessionAccess::release(reader);
      co_return accepted;
    }
    if (chunk->end_stream) {
      co_return acknowledge_body(*reader.session_, *reader.stream_);
    }
  }
}

sched::Task<Status> http2_body_discard(mem::AllocatorRef allocator,
                                       Http2BodyReader &reader,
                                       OperationOptions options) noexcept {
  for (;;) {
    Result<HttpBodyChunkView> chunk =
        co_await http2_body_read(allocator, reader, 0, options);
    if (!chunk.has_value())
      co_return std::move(chunk).error();
    if (chunk->end_stream) {
      Status acknowledged =
          detail::acknowledge_body(*reader.session_, *reader.stream_);
      co_return acknowledged;
    }
  }
}

Http2BodyWriter::~Http2BodyWriter() { abandon_(); }

Http2BodyWriter::Http2BodyWriter(Http2BodyWriter &&other) noexcept
    : session_(std::exchange(other.session_, nullptr)),
      stream_(std::exchange(other.stream_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      allocator_(other.allocator_) {}

Http2BodyWriter &Http2BodyWriter::operator=(Http2BodyWriter &&other) noexcept {
  if (this == &other)
    return *this;
  abandon_();
  session_ = std::exchange(other.session_, nullptr);
  stream_ = std::exchange(other.stream_, nullptr);
  generation_ = std::exchange(other.generation_, 0);
  allocator_ = other.allocator_;
  return *this;
}

bool Http2BodyWriter::valid() const noexcept {
  return session_ != nullptr && session_->owner_alive && stream_ != nullptr &&
         stream_->generation == generation_ && stream_->writer_active;
}

bool Http2BodyWriter::done() const noexcept {
  return valid() && stream_->output_done;
}

void Http2BodyWriter::abandon_() noexcept {
  detail::Http2SessionAccess::release(*this);
}

sched::Task<Status> http2_body_write(mem::AllocatorRef allocator,
                                     Http2BodyWriter &writer,
                                     std::span<const std::byte> bytes,
                                     OperationOptions options) noexcept {
  if (!writer.valid())
    co_return Status::FailedPrecondition("invalid HTTP/2 body writer");
  auto &state = *writer.session_;
  auto &stream = *writer.stream_;
  Status operation = detail::check_operation(state, options);
  if (!operation.ok())
    co_return operation;
  if (stream.writer_busy || stream.output_finishing || stream.output_done)
    co_return Status::FailedPrecondition(
        "HTTP/2 body writer is busy or finished");
  if (bytes.empty())
    co_return Status::Ok();
  if (bytes.size() > state.limits.max_body_bytes ||
      stream.outbound_total > state.limits.max_body_bytes - bytes.size())
    co_return Status::ResourceExhausted("HTTP/2 body exceeds stream limit");
  if (stream.outbound_expected.has_value() &&
      stream.outbound_total + bytes.size() > *stream.outbound_expected)
    co_return Status::OutOfRange("HTTP/2 body exceeds content-length");

  stream.writer_busy = true;
  stream.outbound_data = bytes.data();
  stream.outbound_size = bytes.size();
  stream.outbound_offset = 0;
  stream.outbound_total += bytes.size();
  const std::uint64_t epoch = ++stream.outbound_epoch;
  const int resumed = nghttp2_session_resume_data(state.session, stream.id);
  if (resumed != 0 && resumed != NGHTTP2_ERR_INVALID_ARGUMENT) {
    stream.writer_busy = false;
    co_return detail::status_from_nghttp2(resumed,
                                          "nghttp2_session_resume_data");
  }
  state.notify_owner();
  auto complete = [&]() noexcept {
    return stream.flushed_epoch >= epoch || stream.closed ||
           stream.reset_received || stream.local_reset || !stream.failure.ok();
  };
  Status driven = detail::drive_until(allocator, state, complete, options);
  const bool delivered = stream.flushed_epoch >= epoch;
  if (!driven.ok() ||
      (!delivered && (stream.closed || stream.reset_received ||
                      stream.local_reset || !stream.failure.ok()))) {
    Status failure = !driven.ok() ? driven
                     : !stream.failure.ok()
                         ? stream.failure
                         : detail::stream_reset_status(stream.reset_code);
    (void)detail::reset_stream(state, stream, Http2Error::kCancel);
    stream.writer_busy = false;
    co_return failure;
  }
  stream.outbound_data = nullptr;
  stream.outbound_size = stream.outbound_offset = 0;
  stream.writer_busy = false;
  co_return Status::Ok();
}

sched::Task<Status> http2_body_finish(mem::AllocatorRef allocator,
                                      Http2BodyWriter &writer,
                                      std::span<const HttpHeader> trailers,
                                      OperationOptions options) noexcept {
  if (!writer.valid())
    co_return Status::FailedPrecondition("invalid HTTP/2 body writer");
  auto &state = *writer.session_;
  auto &stream = *writer.stream_;
  Status operation = detail::check_operation(state, options);
  if (!operation.ok())
    co_return operation;
  if (stream.writer_busy || stream.output_finishing || stream.output_done)
    co_return Status::FailedPrecondition(
        "HTTP/2 body writer is busy or finished");
  if (stream.outbound_expected.has_value() &&
      stream.outbound_total != *stream.outbound_expected)
    co_return Status::OutOfRange("HTTP/2 body does not match content-length");
  Status trailer_status = detail::prepare_trailers(state, stream, trailers);
  if (!trailer_status.ok())
    co_return trailer_status;

  stream.writer_busy = true;
  stream.output_finishing = true;
  stream.outbound_data = nullptr;
  stream.outbound_size = stream.outbound_offset = 0;
  const std::uint64_t epoch = ++stream.outbound_epoch;
  const int resumed = nghttp2_session_resume_data(state.session, stream.id);
  if (resumed != 0 && resumed != NGHTTP2_ERR_INVALID_ARGUMENT) {
    stream.writer_busy = false;
    co_return detail::status_from_nghttp2(resumed,
                                          "nghttp2_session_resume_data");
  }
  state.notify_owner();
  auto complete = [&]() noexcept {
    return (stream.output_done && stream.flushed_epoch >= epoch) ||
           stream.reset_received || stream.local_reset || !stream.failure.ok();
  };
  Status driven = detail::drive_until(allocator, state, complete, options);
  const bool delivered = stream.output_done && stream.flushed_epoch >= epoch;
  if (!driven.ok() ||
      (!delivered &&
       (stream.reset_received || stream.local_reset || !stream.failure.ok()))) {
    Status failure = !driven.ok() ? driven
                     : !stream.failure.ok()
                         ? stream.failure
                         : detail::stream_reset_status(stream.reset_code);
    (void)detail::reset_stream(state, stream, Http2Error::kCancel);
    stream.writer_busy = false;
    co_return failure;
  }
  if (stream.trailer_nv != nullptr) {
    state.allocator.deallocate(stream.trailer_nv,
                               stream.trailer_nv_count * sizeof(nghttp2_nv),
                               alignof(nghttp2_nv));
    stream.trailer_nv = nullptr;
    stream.trailer_nv_count = 0;
  }
  stream.writer_busy = false;
  co_return Status::Ok();
}

namespace {

Status validate_session_arguments(mem::AllocatorRef allocator,
                                  const HttpTransport &transport,
                                  const Http2Limits &limits) noexcept {
  constexpr std::uint32_t kMaximumWindow = 0x7fffffffu;
  if (!allocator.valid())
    return Status::InvalidArgument("HTTP/2 requires a complete allocator");
  if (!transport.valid())
    return Status::InvalidArgument("HTTP/2 requires a connected transport");
  if (transport.encrypted() && transport.negotiated_protocol() != "h2")
    return Status::FailedPrecondition(
        "TLS transport did not negotiate HTTP/2 with ALPN");
  if (limits.max_header_bytes == 0 || limits.max_body_bytes == 0 ||
      limits.max_concurrent_streams == 0 || limits.stream_receive_window == 0 ||
      limits.stream_receive_window > kMaximumWindow ||
      limits.connection_receive_window <
          NGHTTP2_INITIAL_CONNECTION_WINDOW_SIZE ||
      limits.connection_receive_window > kMaximumWindow ||
      limits.read_chunk_bytes == 0 ||
      limits.read_chunk_bytes > limits.stream_receive_window ||
      limits.max_header_bytes > (std::numeric_limits<std::uint32_t>::max)())
    return Status::InvalidArgument("invalid HTTP/2 limits");
  return Status::Ok();
}

void release_session_owner(detail::Http2SessionState *&state) noexcept {
  if (state == nullptr)
    return;
  state->owner_alive = false;
  state->notify_owner();
  detail::Http2SessionState *saved = std::exchange(state, nullptr);
  saved->release();
}

std::size_t active_protocol_streams(detail::Http2SessionState &state) noexcept {
  std::size_t active = 0;
  state.streams.for_each(
      [&](const std::int32_t &, detail::Http2StreamState *&stream) {
        if (!stream->closed)
          ++active;
      });
  return active;
}

sched::Task<Status> session_reset(mem::AllocatorRef allocator,
                                  detail::Http2SessionState *state,
                                  Http2StreamId id, Http2Error error,
                                  OperationOptions options) noexcept {
  if (state == nullptr || !id.valid())
    co_return Status::InvalidArgument("invalid HTTP/2 stream reset");
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return operation;
  detail::Http2StreamState *stream = state->find(id.value());
  if (stream == nullptr)
    co_return Status::NotFound("HTTP/2 stream does not exist");
  if (stream->closed)
    co_return Status::Ok();
  Status submitted = detail::reset_stream(*state, *stream, error);
  if (!submitted.ok())
    co_return submitted;
  ++stream->operation_pins;
  auto complete = [&]() noexcept {
    return stream->closed ||
           (stream->reset_wire_epoch != 0 &&
            state->flushed_wire_epoch >= stream->reset_wire_epoch);
  };
  Status driven = detail::drive_until(allocator, *state, complete, options);
  --stream->operation_pins;
  state->maybe_retire(stream);
  state->collect_garbage();
  co_return driven;
}

sched::Task<Status> session_drain(mem::AllocatorRef allocator,
                                  detail::Http2SessionState *state,
                                  Http2Error error,
                                  OperationOptions options) noexcept {
  if (state == nullptr)
    co_return Status::FailedPrecondition("invalid HTTP/2 session");
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return operation;
  if (!state->local_draining) {
    const std::int32_t last =
        nghttp2_session_get_last_proc_stream_id(state->session);
    const int submitted =
        nghttp2_submit_goaway(state->session, NGHTTP2_FLAG_NONE, last,
                              static_cast<std::uint32_t>(error), nullptr, 0);
    if (submitted != 0)
      co_return detail::status_from_nghttp2(submitted, "nghttp2_submit_goaway");
    state->local_draining = true;
    ++state->submit_epoch;
    state->notify_owner();
  }
  auto complete = [&]() noexcept {
    return state->goaway_wire_epoch != 0 &&
           state->flushed_wire_epoch >= state->goaway_wire_epoch &&
           active_protocol_streams(*state) == 0;
  };
  co_return detail::drive_until(allocator, *state, complete, options);
}

sched::Task<Status> session_close(mem::AllocatorRef allocator,
                                  detail::Http2SessionState *state,
                                  OperationOptions options) noexcept {
  if (state == nullptr)
    co_return Status::Ok();
  Status drained =
      co_await session_drain(allocator, state, Http2Error::kNoError, options);
  Status shutdown = co_await state->transport.shutdown(allocator, options);
  state->closed = true;
  state->notify_owner();
  co_return drained.ok() ? shutdown : drained;
}

sched::Task<Status> write_source(mem::AllocatorRef allocator,
                                 Http2BodyWriter &writer, HttpBodySource source,
                                 OperationOptions options) noexcept {
  if (!source.valid())
    co_return Status::InvalidArgument("HTTP/2 body source is empty");
  for (;;) {
    auto task =
        detail::Http2SessionAccess::source_next(allocator, source, options);
    if (!task.valid())
      co_return Status::ResourceExhausted(
          "HTTP/2 body source task allocation failed");
    Result<HttpBodyChunk> next = co_await task;
    if (!next.has_value())
      co_return std::move(next).error();
    if (!next->end_stream && !next->trailers.empty())
      co_return Status::InvalidArgument("HTTP/2 trailers require end_stream");
    if (!next->bytes.empty()) {
      Status written = co_await writer.write(next->bytes, options);
      if (!written.ok())
        co_return written;
    } else if (!next->end_stream) {
      co_return Status::DataLoss(
          "HTTP/2 body source returned empty nonterminal chunk");
    }
    if (next->end_stream)
      co_return co_await writer.finish(next->trailers, options);
  }
}

sched::Task<Status> send_submitted_fixed(mem::AllocatorRef allocator,
                                         detail::Http2SessionState &state,
                                         detail::Http2StreamState &stream,
                                         std::span<const std::byte> body,
                                         OperationOptions options) noexcept {
  // A fixed body is one logical submission. Feed its borrowed bytes and EOF to
  // nghttp2 before the first drive so HEADERS, DATA and END_STREAM share one
  // protocol lifecycle rather than constructing a public streaming writer and
  // driving the same session three times.
  ++stream.operation_pins;
  stream.writer_busy = true;
  stream.output_finishing = true;
  stream.outbound_data = body.data();
  stream.outbound_size = body.size();
  stream.outbound_offset = 0;
  stream.outbound_total = body.size();
  const std::uint64_t epoch = ++stream.outbound_epoch;
  const int resumed = nghttp2_session_resume_data(state.session, stream.id);
  if (resumed != 0 && resumed != NGHTTP2_ERR_INVALID_ARGUMENT) {
    const Status failure =
        detail::status_from_nghttp2(resumed, "nghttp2_session_resume_data");
    (void)detail::reset_stream(state, stream, Http2Error::kCancel);
    stream.outbound_data = nullptr;
    stream.outbound_size = stream.outbound_offset = 0;
    stream.writer_busy = false;
    --stream.operation_pins;
    state.maybe_retire(&stream);
    state.collect_garbage();
    co_return failure;
  }
  state.notify_owner();
  auto complete = [&]() noexcept {
    return (stream.output_done && stream.flushed_epoch >= epoch) ||
           stream.reset_received || stream.local_reset || !stream.failure.ok();
  };
  Status driven = detail::drive_until(allocator, state, complete, options);
  const bool delivered = stream.output_done && stream.flushed_epoch >= epoch;
  Status result = Status::Ok();
  if (!driven.ok() ||
      (!delivered &&
       (stream.reset_received || stream.local_reset || !stream.failure.ok()))) {
    result = !driven.ok() ? std::move(driven)
             : !stream.failure.ok()
                 ? stream.failure
                 : detail::stream_reset_status(stream.reset_code);
    (void)detail::reset_stream(state, stream, Http2Error::kCancel);
  }
  stream.outbound_data = nullptr;
  stream.outbound_size = stream.outbound_offset = 0;
  stream.writer_busy = false;
  --stream.operation_pins;
  state.maybe_retire(&stream);
  state.collect_garbage();
  co_return result;
}

} // namespace

Http2ServerSession::~Http2ServerSession() { release_session_owner(state_); }

Http2ServerSession::Http2ServerSession(Http2ServerSession &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      allocator_(other.allocator_) {}

Http2ServerSession &
Http2ServerSession::operator=(Http2ServerSession &&other) noexcept {
  if (this == &other)
    return *this;
  release_session_owner(state_);
  state_ = std::exchange(other.state_, nullptr);
  allocator_ = other.allocator_;
  return *this;
}

Result<Http2ServerSession>
Http2ServerSession::create(mem::AllocatorRef allocator, HttpTransport transport,
                           Http2Limits limits) noexcept {
  Status valid = validate_session_arguments(allocator, transport, limits);
  if (!valid.ok())
    return make_unexpected(std::move(valid));
  detail::Http2SessionState *state =
      detail::allocate_session(allocator, std::move(transport), limits, true);
  if (state == nullptr)
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 server session allocation"));
  Http2ServerSession result;
  result.state_ = state;
  result.allocator_ = allocator;
  return result;
}

bool Http2ServerSession::valid() const noexcept {
  return state_ != nullptr && state_->owner_alive && !state_->closed &&
         state_->session != nullptr && state_->fatal_status.ok();
}

unsigned Http2ServerSession::owner_worker() const noexcept {
  return state_ == nullptr ? 0 : state_->transport.owner_worker();
}

const Http2Limits &Http2ServerSession::limits() const noexcept {
  static const Http2Limits empty{};
  return state_ == nullptr ? empty : state_->limits;
}

std::optional<Http2PeerGoaway>
Http2ServerSession::peer_goaway() const noexcept {
  if (state_ == nullptr || !state_->peer_sent_goaway)
    return std::nullopt;
  auto last = Http2StreamId::from_value(state_->peer_last_stream_id);
  return Http2PeerGoaway{
      last.value_or(Http2StreamId{}),
      static_cast<Http2Error>(state_->peer_error),
      StringView(reinterpret_cast<const char *>(state_->goaway_debug.data),
                 static_cast<std::uint32_t>(state_->goaway_debug.size))};
}

sched::Task<Result<std::optional<Http2IncomingRequest>>>
http2_server_accept(mem::AllocatorRef allocator, Http2ServerSession &session,
                    OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || !state->server)
    co_return make_unexpected(
        Status::FailedPrecondition("invalid HTTP/2 server session"));
  auto ready = [&]() noexcept {
    return state->ready_head != nullptr || state->peer_closed ||
           state->peer_sent_goaway;
  };
  Status driven = detail::drive_until(allocator, *state, ready, options);
  if (!driven.ok() && state->ready_head == nullptr)
    co_return make_unexpected(std::move(driven));
  detail::Http2StreamState *stream = state->pop_ready();
  if (stream == nullptr)
    co_return std::optional<Http2IncomingRequest>{};
  co_return std::optional<Http2IncomingRequest>(
      detail::make_request(*state, *stream));
}

sched::Task<Result<Http2BodyWriter>> http2_server_start_response(
    mem::AllocatorRef allocator, Http2ServerSession &session, Http2StreamId id,
    const Http2ResponseHead &head, std::optional<std::uint64_t> content_length,
    OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || !id.valid())
    co_return make_unexpected(
        Status::InvalidArgument("invalid HTTP/2 response stream"));
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return make_unexpected(std::move(operation));
  detail::Http2StreamState *stream = state->find(id.value());
  if (stream == nullptr || !stream->head_delivered)
    co_return make_unexpected(
        Status::NotFound("HTTP/2 request stream is not accepted"));
  Status submitted =
      detail::submit_response_head(*state, *stream, head, content_length);
  if (!submitted.ok())
    co_return make_unexpected(std::move(submitted));
  Http2BodyWriter writer = detail::Http2SessionAccess::writer(*state, *stream);
  auto complete = [&]() noexcept {
    return (stream->head_wire_epoch != 0 &&
            state->flushed_wire_epoch >= stream->head_wire_epoch) ||
           stream->closed || stream->reset_received || !stream->failure.ok();
  };
  Status flushed = detail::drive_until(allocator, *state, complete, options);
  if (!flushed.ok())
    co_return make_unexpected(std::move(flushed));
  if (!stream->failure.ok())
    co_return make_unexpected(stream->failure);
  if (stream->reset_received || stream->closed)
    co_return make_unexpected(detail::stream_reset_status(stream->reset_code));
  co_return std::move(writer);
}

sched::Task<Status> http2_server_send_fixed(mem::AllocatorRef allocator,
                                            Http2ServerSession &session,
                                            Http2StreamId id,
                                            const Http2ResponseHead &head,
                                            std::span<const std::byte> body,
                                            OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || !id.valid())
    co_return Status::InvalidArgument("invalid HTTP/2 response stream");
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return operation;
  detail::Http2StreamState *stream = state->find(id.value());
  if (stream == nullptr || !stream->head_delivered)
    co_return Status::NotFound("HTTP/2 request stream is not accepted");
  Status submitted =
      detail::submit_response_head(*state, *stream, head, body.size());
  if (!submitted.ok())
    co_return submitted;
  co_return co_await send_submitted_fixed(allocator, *state, *stream, body,
                                          options);
}

sched::Task<Status>
http2_server_send_source(mem::AllocatorRef allocator,
                         Http2ServerSession &session, Http2StreamId id,
                         const Http2ResponseHead &head, HttpBodySource source,
                         OperationOptions options) noexcept {
  Result<Http2BodyWriter> started = co_await http2_server_start_response(
      allocator, session, id, head, source.content_length(), options);
  if (!started.has_value())
    co_return std::move(started).error();
  Http2BodyWriter writer = std::move(*started);
  co_return co_await write_source(allocator, writer, source, options);
}

sched::Task<Status>
Http2ServerSession::reset(Http2StreamId id, Http2Error error,
                          OperationOptions options) noexcept {
  return session_reset(allocator_, state_, id, error, options);
}

sched::Task<Status>
Http2ServerSession::drain(Http2Error error, OperationOptions options) noexcept {
  return session_drain(allocator_, state_, error, options);
}

sched::Task<Status>
Http2ServerSession::close(OperationOptions options) noexcept {
  return session_close(allocator_, state_, options);
}

Http2ClientSession::~Http2ClientSession() { release_session_owner(state_); }

Http2ClientSession::Http2ClientSession(Http2ClientSession &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      allocator_(other.allocator_) {}

Http2ClientSession &
Http2ClientSession::operator=(Http2ClientSession &&other) noexcept {
  if (this == &other)
    return *this;
  release_session_owner(state_);
  state_ = std::exchange(other.state_, nullptr);
  allocator_ = other.allocator_;
  return *this;
}

Result<Http2ClientSession>
Http2ClientSession::create(mem::AllocatorRef allocator, HttpTransport transport,
                           Http2Limits limits) noexcept {
  Status valid = validate_session_arguments(allocator, transport, limits);
  if (!valid.ok())
    return make_unexpected(std::move(valid));
  detail::Http2SessionState *state =
      detail::allocate_session(allocator, std::move(transport), limits, false);
  if (state == nullptr)
    return make_unexpected(
        Status::ResourceExhausted("HTTP/2 client session allocation"));
  Http2ClientSession result;
  result.state_ = state;
  result.allocator_ = allocator;
  return result;
}

bool Http2ClientSession::valid() const noexcept {
  return state_ != nullptr && state_->owner_alive && !state_->closed &&
         state_->session != nullptr && state_->fatal_status.ok();
}

unsigned Http2ClientSession::owner_worker() const noexcept {
  return state_ == nullptr ? 0 : state_->transport.owner_worker();
}

const Http2Limits &Http2ClientSession::limits() const noexcept {
  static const Http2Limits empty{};
  return state_ == nullptr ? empty : state_->limits;
}

std::optional<Http2PeerGoaway>
Http2ClientSession::peer_goaway() const noexcept {
  if (state_ == nullptr || !state_->peer_sent_goaway)
    return std::nullopt;
  auto last = Http2StreamId::from_value(state_->peer_last_stream_id);
  return Http2PeerGoaway{
      last.value_or(Http2StreamId{}),
      static_cast<Http2Error>(state_->peer_error),
      StringView(reinterpret_cast<const char *>(state_->goaway_debug.data),
                 static_cast<std::uint32_t>(state_->goaway_debug.size))};
}

sched::Task<Result<Http2OutgoingRequest>> http2_client_start_request(
    mem::AllocatorRef allocator, Http2ClientSession &session,
    const Http2RequestHead &head, std::optional<std::uint64_t> content_length,
    OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || state->server)
    co_return make_unexpected(
        Status::FailedPrecondition("invalid HTTP/2 client session"));
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return make_unexpected(std::move(operation));
  Result<detail::Http2StreamState *> submitted =
      detail::submit_request_head(*state, head, content_length);
  if (!submitted.has_value())
    co_return make_unexpected(std::move(submitted).error());
  detail::Http2StreamState *stream = *submitted;
  Http2BodyWriter writer = detail::Http2SessionAccess::writer(*state, *stream);
  auto complete = [&]() noexcept {
    return (stream->head_wire_epoch != 0 &&
            state->flushed_wire_epoch >= stream->head_wire_epoch) ||
           stream->closed || stream->reset_received || !stream->failure.ok();
  };
  Status flushed = detail::drive_until(allocator, *state, complete, options);
  if (!flushed.ok())
    co_return make_unexpected(std::move(flushed));
  if (!stream->failure.ok())
    co_return make_unexpected(stream->failure);
  if (stream->reset_received || stream->closed)
    co_return make_unexpected(detail::stream_reset_status(stream->reset_code));
  co_return Http2OutgoingRequest{*Http2StreamId::from_value(stream->id),
                                 std::move(writer)};
}

sched::Task<Result<Http2StreamId>> http2_client_send_fixed(
    mem::AllocatorRef allocator, Http2ClientSession &session,
    const Http2RequestHead &head, std::span<const std::byte> body,
    OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || state->server)
    co_return make_unexpected(
        Status::FailedPrecondition("invalid HTTP/2 client session"));
  Status operation = detail::check_operation(*state, options);
  if (!operation.ok())
    co_return make_unexpected(std::move(operation));
  Result<detail::Http2StreamState *> submitted =
      detail::submit_request_head(*state, head, body.size());
  if (!submitted.has_value())
    co_return make_unexpected(std::move(submitted).error());
  detail::Http2StreamState *stream = *submitted;
  const Http2StreamId id = *Http2StreamId::from_value(stream->id);
  Status sent =
      co_await send_submitted_fixed(allocator, *state, *stream, body, options);
  if (!sent.ok())
    co_return make_unexpected(std::move(sent));
  co_return id;
}

sched::Task<Result<Http2StreamId>>
http2_client_send_source(mem::AllocatorRef allocator,
                         Http2ClientSession &session,
                         const Http2RequestHead &head, HttpBodySource source,
                         OperationOptions options) noexcept {
  Result<Http2OutgoingRequest> started = co_await http2_client_start_request(
      allocator, session, head, source.content_length(), options);
  if (!started.has_value())
    co_return make_unexpected(std::move(started).error());
  Http2StreamId id = started->stream_id;
  Status written =
      co_await write_source(allocator, started->body, source, options);
  if (!written.ok())
    co_return make_unexpected(std::move(written));
  co_return id;
}

sched::Task<Result<std::optional<Http2IncomingResponse>>>
http2_client_next_response(mem::AllocatorRef allocator,
                           Http2ClientSession &session,
                           OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || state->server)
    co_return make_unexpected(
        Status::FailedPrecondition("invalid HTTP/2 client session"));
  auto ready = [&]() noexcept {
    return state->ready_head != nullptr || state->peer_closed ||
           (state->peer_sent_goaway && active_protocol_streams(*state) == 0);
  };
  Status driven = detail::drive_until(allocator, *state, ready, options);
  if (!driven.ok() && state->ready_head == nullptr)
    co_return make_unexpected(std::move(driven));
  detail::Http2StreamState *stream = state->pop_ready();
  if (stream == nullptr)
    co_return std::optional<Http2IncomingResponse>{};
  co_return std::optional<Http2IncomingResponse>(
      detail::make_response(*state, *stream));
}

sched::Task<Result<Http2IncomingResponse>>
http2_client_read_response(mem::AllocatorRef allocator,
                           Http2ClientSession &session, Http2StreamId id,
                           OperationOptions options) noexcept {
  detail::Http2SessionState *state = session.state_;
  if (state == nullptr || !id.valid() || state->server)
    co_return make_unexpected(
        Status::InvalidArgument("invalid HTTP/2 response stream"));
  detail::Http2StreamState *stream = state->find(id.value());
  if (stream == nullptr)
    co_return make_unexpected(Status::NotFound("HTTP/2 stream does not exist"));
  if (stream->head_delivered)
    co_return make_unexpected(
        Status::FailedPrecondition("HTTP/2 response already consumed"));
  auto ready = [&]() noexcept {
    return stream->head_ready || state->peer_closed;
  };
  Status driven = detail::drive_until(allocator, *state, ready, options);
  if (!driven.ok() && !stream->head_ready)
    co_return make_unexpected(std::move(driven));
  detail::Http2StreamState *taken = state->take_ready(id.value());
  if (taken == nullptr)
    co_return make_unexpected(
        Status::Unavailable("HTTP/2 response did not complete"));
  co_return detail::make_response(*state, *taken);
}

sched::Task<Status>
Http2ClientSession::reset(Http2StreamId id, Http2Error error,
                          OperationOptions options) noexcept {
  return session_reset(allocator_, state_, id, error, options);
}

sched::Task<Status>
Http2ClientSession::drain(Http2Error error, OperationOptions options) noexcept {
  return session_drain(allocator_, state_, error, options);
}

sched::Task<Status>
Http2ClientSession::close(OperationOptions options) noexcept {
  return session_close(allocator_, state_, options);
}

} // namespace hyle::io
