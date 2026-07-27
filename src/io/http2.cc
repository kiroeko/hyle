#include "hyle/io/http2.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include <nghttp2/nghttp2.h>

#include "hyle/c/io.h"
#include "hyle/container/swiss_table.h"
#include "c_io_internal.h"

namespace hyle::io {
namespace {

constexpr std::size_t kAllocAlign = alignof(std::max_align_t);
constexpr std::size_t kAllocPrefix =
    (sizeof(std::size_t) + kAllocAlign - 1) & ~(kAllocAlign - 1);

struct ByteBuf {
  mem::Allocator alloc;
  std::uint8_t* data = nullptr;
  std::uint32_t size = 0;
  std::uint32_t cap = 0;

  explicit ByteBuf(mem::Allocator a) noexcept : alloc(a) {}
  ~ByteBuf() {
    if (data != nullptr) alloc.deallocate(data, cap, kAllocAlign);
  }
  HYLE_DISABLE_COPY_AND_MOVE(ByteBuf);

  bool reserve(std::uint32_t want) noexcept {
    if (want <= cap) return true;
    std::uint32_t next = cap == 0 ? 4096u : cap;
    while (next < want) {
      if (next > (std::numeric_limits<std::uint32_t>::max)() / 2) {
        next = want;
        break;
      }
      next *= 2;
    }
    void* p = alloc.reallocate(data, cap, next, kAllocAlign);
    if (p == nullptr) return false;
    data = static_cast<std::uint8_t*>(p);
    cap = next;
    return true;
  }

  bool append(const void* src, std::size_t n,
              std::uint32_t limit) noexcept {
    if (n > limit || size > limit - n) return false;
    const auto total = size + static_cast<std::uint32_t>(n);
    if (!reserve(total)) return false;
    if (n != 0) std::memcpy(data + size, src, n);
    size = total;
    return true;
  }
};

struct FieldRef {
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  bool present = false;
};

struct HeaderSlot {
  FieldRef name;
  FieldRef value;
};

struct Stream {
  explicit Stream(mem::Allocator alloc) noexcept
      : header_bytes(alloc), body(alloc), outbound(alloc) {}

  ByteBuf header_bytes;
  ByteBuf body;
  ByteBuf outbound;
  HeaderSlot headers[kHttp2MaxHeaders]{};
  FieldRef method;
  FieldRef scheme;
  FieldRef authority;
  FieldRef path;
  std::size_t header_count = 0;
  std::uint32_t outbound_sent = 0;
  std::uint32_t error_code = 0;
  std::int32_t id = 0;
  int status = 0;
  bool queued = false;
  bool delivered = false;
  bool borrowed = false;
  bool closed = false;
  bool failed = false;
  bool response_submitted = false;
  Stream* ready_next = nullptr;
  Stream* garbage_next = nullptr;
};

struct SessionState {
  explicit SessionState(mem::Allocator a, bool is_server) noexcept
      : alloc(a), streams(a), input(a), server(is_server) {}

  mem::Allocator alloc;
  container::SwissTable<std::int32_t, Stream*,
                        container::Hash<std::int32_t>,
                        std::equal_to<>, mem::Allocator>
      streams;
  ByteBuf input;
  nghttp2_session* session = nullptr;
  nghttp2_mem ngmem{};
  Stream* ready_head = nullptr;
  Stream* ready_tail = nullptr;
  Stream* garbage = nullptr;
  bool server = false;
  bool fatal = false;

  bool initialize() noexcept;
  void destroy() noexcept;
  Stream* create_stream(std::int32_t id) noexcept;
  Stream* find_stream(std::int32_t id) noexcept {
    Stream** value = streams.find(id);
    return value == nullptr ? nullptr : *value;
  }
  void queue_ready(Stream* stream) noexcept {
    if (stream->queued || stream->failed) return;
    stream->queued = true;
    stream->ready_next = nullptr;
    if (ready_tail != nullptr)
      ready_tail->ready_next = stream;
    else
      ready_head = stream;
    ready_tail = stream;
  }
  Stream* pop_ready() noexcept {
    Stream* stream = ready_head;
    if (stream == nullptr) return nullptr;
    ready_head = stream->ready_next;
    if (ready_head == nullptr) ready_tail = nullptr;
    stream->ready_next = nullptr;
    stream->delivered = true;
    stream->borrowed = true;
    return stream;
  }
  void retire(Stream* stream) noexcept {
    stream->garbage_next = garbage;
    garbage = stream;
  }
  void collect_garbage() noexcept;
  void release_current(const void*& current) noexcept;
};

Stream* allocate_stream(mem::Allocator alloc) noexcept {
  void* p = alloc.allocate(sizeof(Stream), alignof(Stream));
  return p == nullptr ? nullptr : new (p) Stream(alloc);
}

void destroy_stream(mem::Allocator alloc, Stream* stream) noexcept {
  if (stream == nullptr) return;
  stream->~Stream();
  alloc.deallocate(stream, sizeof(Stream), alignof(Stream));
}

void* h2_malloc(std::size_t size, void* user) noexcept {
  auto* state = static_cast<SessionState*>(user);
  if (size > (std::numeric_limits<std::size_t>::max)() - kAllocPrefix)
    return nullptr;
  const std::size_t total = size + kAllocPrefix;
  void* base = state->alloc.allocate(total, kAllocAlign);
  if (base == nullptr) return nullptr;
  *static_cast<std::size_t*>(base) = total;
  return static_cast<std::uint8_t*>(base) + kAllocPrefix;
}

void h2_free(void* ptr, void* user) noexcept {
  if (ptr == nullptr) return;
  auto* state = static_cast<SessionState*>(user);
  void* base = static_cast<std::uint8_t*>(ptr) - kAllocPrefix;
  const std::size_t total = *static_cast<std::size_t*>(base);
  state->alloc.deallocate(base, total, kAllocAlign);
}

void* h2_calloc(std::size_t count, std::size_t size, void* user) noexcept {
  if (size != 0 &&
      count > (std::numeric_limits<std::size_t>::max)() / size)
    return nullptr;
  const std::size_t bytes = count * size;
  void* p = h2_malloc(bytes, user);
  if (p != nullptr) std::memset(p, 0, bytes);
  return p;
}

void* h2_realloc(void* ptr, std::size_t size, void* user) noexcept {
  if (ptr == nullptr) return h2_malloc(size, user);
  if (size == 0) {
    h2_free(ptr, user);
    return nullptr;
  }
  if (size > (std::numeric_limits<std::size_t>::max)() - kAllocPrefix)
    return nullptr;
  auto* state = static_cast<SessionState*>(user);
  void* base = static_cast<std::uint8_t*>(ptr) - kAllocPrefix;
  const std::size_t old_total = *static_cast<std::size_t*>(base);
  const std::size_t new_total = size + kAllocPrefix;
  void* next =
      state->alloc.reallocate(base, old_total, new_total, kAllocAlign);
  if (next == nullptr) return nullptr;
  *static_cast<std::size_t*>(next) = new_total;
  return static_cast<std::uint8_t*>(next) + kAllocPrefix;
}

FieldRef append_field_bytes(Stream& stream, const void* data,
                            std::size_t size) noexcept {
  FieldRef ref;
  ref.offset = stream.header_bytes.size;
  if (size > (std::numeric_limits<std::uint32_t>::max)() ||
      !stream.header_bytes.append(data, size, kHttp2MaxHeaderBytes))
    return ref;
  ref.length = static_cast<std::uint32_t>(size);
  ref.present = true;
  return ref;
}

bool bytes_equal(const std::uint8_t* data, std::size_t size,
                 const char* literal) noexcept {
  const std::size_t n = std::strlen(literal);
  return size == n && std::memcmp(data, literal, n) == 0;
}

int on_begin_headers(nghttp2_session* session, const nghttp2_frame* frame,
                     void* user) noexcept {
  auto& state = *static_cast<SessionState*>(user);
  if (frame->hd.type != NGHTTP2_HEADERS) return 0;
  const std::int32_t id = frame->hd.stream_id;
  Stream* stream = state.find_stream(id);
  if (stream == nullptr) {
    stream = state.create_stream(id);
    if (stream == nullptr) return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    nghttp2_session_set_stream_user_data(session, id, stream);
  } else if (!state.server && stream->status >= 100 &&
             stream->status < 200) {
    // A final response follows any informational response. Do not expose the
    // interim pseudo/header block as the completed response.
    stream->header_bytes.size = 0;
    stream->header_count = 0;
    stream->status = 0;
  }
  return 0;
}

int on_header(nghttp2_session*, const nghttp2_frame* frame,
              const std::uint8_t* name, std::size_t name_len,
              const std::uint8_t* value, std::size_t value_len, std::uint8_t,
              void* user) noexcept {
  auto& state = *static_cast<SessionState*>(user);
  Stream* stream = state.find_stream(frame->hd.stream_id);
  if (stream == nullptr) return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  if (name_len > kHttp2MaxHeaderBytes ||
      value_len > kHttp2MaxHeaderBytes - name_len ||
      stream->header_bytes.size >
          kHttp2MaxHeaderBytes - name_len - value_len) {
    stream->failed = true;
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }

  if (name_len != 0 && name[0] == ':') {
    FieldRef ref = append_field_bytes(*stream, value, value_len);
    if (!ref.present) {
      stream->failed = true;
      return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
    }
    if (bytes_equal(name, name_len, ":method"))
      stream->method = ref;
    else if (bytes_equal(name, name_len, ":scheme"))
      stream->scheme = ref;
    else if (bytes_equal(name, name_len, ":authority"))
      stream->authority = ref;
    else if (bytes_equal(name, name_len, ":path"))
      stream->path = ref;
    else if (bytes_equal(name, name_len, ":status")) {
      int status = 0;
      if (value_len != 3) {
        stream->failed = true;
        return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
      }
      for (std::size_t i = 0; i < value_len; ++i) {
        if (value[i] < '0' || value[i] > '9') {
          stream->failed = true;
          return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
        }
        status = status * 10 + static_cast<int>(value[i] - '0');
      }
      stream->status = status;
    }
    return 0;
  }

  if (stream->header_count >= kHttp2MaxHeaders) {
    stream->failed = true;
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }
  HeaderSlot& slot = stream->headers[stream->header_count];
  slot.name = append_field_bytes(*stream, name, name_len);
  slot.value = append_field_bytes(*stream, value, value_len);
  if (!slot.name.present || !slot.value.present) {
    stream->failed = true;
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }
  ++stream->header_count;
  return 0;
}

int on_data_chunk(nghttp2_session*, std::uint8_t, std::int32_t stream_id,
                  const std::uint8_t* data, std::size_t len,
                  void* user) noexcept {
  auto& state = *static_cast<SessionState*>(user);
  Stream* stream = state.find_stream(stream_id);
  if (stream == nullptr) return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  if (!stream->body.append(data, len,
                           static_cast<std::uint32_t>(kHttp2MaxBodyBytes))) {
    stream->failed = true;
    return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
  }
  return 0;
}

int on_frame_recv(nghttp2_session*, const nghttp2_frame* frame,
                  void* user) noexcept {
  auto& state = *static_cast<SessionState*>(user);
  if ((frame->hd.type == NGHTTP2_HEADERS ||
       frame->hd.type == NGHTTP2_DATA) &&
      (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0) {
    Stream* stream = state.find_stream(frame->hd.stream_id);
    if (stream != nullptr && !stream->failed &&
        (state.server || stream->status < 100 || stream->status >= 200))
      state.queue_ready(stream);
  }
  return 0;
}

int on_stream_close(nghttp2_session*, std::int32_t stream_id,
                    std::uint32_t error_code, void* user) noexcept {
  auto& state = *static_cast<SessionState*>(user);
  Stream* stream = state.find_stream(stream_id);
  if (stream == nullptr) return 0;
  stream->closed = true;
  stream->error_code = error_code;
  if (!state.server && !stream->queued && !stream->delivered) {
    // Surface reset/refusal to the client instead of making next_response hang.
    stream->failed = false;
    state.queue_ready(stream);
  } else if ((stream->failed && !stream->queued && !stream->delivered) ||
             (stream->delivered && !stream->borrowed)) {
    state.retire(stream);
  }
  return 0;
}

nghttp2_ssize read_outbound(nghttp2_session*, std::int32_t, std::uint8_t* buf,
                            std::size_t length, std::uint32_t* data_flags,
                            nghttp2_data_source* source, void*) noexcept {
  auto* stream = static_cast<Stream*>(source->ptr);
  const std::uint32_t remain = stream->outbound.size - stream->outbound_sent;
  const std::size_t take = std::min<std::size_t>(length, remain);
  if (take != 0) {
    std::memcpy(buf, stream->outbound.data + stream->outbound_sent, take);
    stream->outbound_sent += static_cast<std::uint32_t>(take);
  }
  if (stream->outbound_sent == stream->outbound.size)
    *data_flags |= NGHTTP2_DATA_FLAG_EOF;
  return static_cast<nghttp2_ssize>(take);
}

bool SessionState::initialize() noexcept {
  if (!streams.reserve(kHttp2DefaultMaxConcurrentStreams * 2) ||
      !input.reserve(16 * 1024))
    return false;

  nghttp2_session_callbacks* callbacks = nullptr;
  nghttp2_option* option = nullptr;
  if (nghttp2_session_callbacks_new(&callbacks) != 0 ||
      nghttp2_option_new(&option) != 0) {
    if (callbacks != nullptr) nghttp2_session_callbacks_del(callbacks);
    if (option != nullptr) nghttp2_option_del(option);
    return false;
  }
  nghttp2_session_callbacks_set_on_begin_headers_callback(callbacks,
                                                          on_begin_headers);
  nghttp2_session_callbacks_set_on_header_callback(callbacks, on_header);
  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(callbacks,
                                                             on_data_chunk);
  nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks, on_frame_recv);
  nghttp2_session_callbacks_set_on_stream_close_callback(callbacks,
                                                          on_stream_close);
  nghttp2_option_set_peer_max_concurrent_streams(
      option, kHttp2DefaultMaxConcurrentStreams);
  nghttp2_option_set_max_send_header_block_length(option,
                                                   kHttp2MaxHeaderBytes);
  nghttp2_option_set_max_settings(option, 32);
  nghttp2_option_set_max_continuations(option, 16);

  ngmem = nghttp2_mem{this, h2_malloc, h2_free, h2_calloc, h2_realloc};
  const int rv =
      server ? nghttp2_session_server_new3(&session, callbacks, this, option,
                                           &ngmem)
             : nghttp2_session_client_new3(&session, callbacks, this, option,
                                           &ngmem);
  nghttp2_option_del(option);
  nghttp2_session_callbacks_del(callbacks);
  if (rv != 0) return false;

  nghttp2_settings_entry settings[3] = {
      {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS,
       kHttp2DefaultMaxConcurrentStreams},
      {NGHTTP2_SETTINGS_MAX_HEADER_LIST_SIZE, kHttp2MaxHeaderBytes},
      {NGHTTP2_SETTINGS_ENABLE_PUSH, 0},
  };
  // SETTINGS_ENABLE_PUSH is a client-to-server setting; a server sending it is
  // a connection-level protocol error under RFC 9113.
  const std::size_t settings_count = server ? 2 : 3;
  if (nghttp2_submit_settings(session, NGHTTP2_FLAG_NONE, settings,
                              settings_count) != 0) {
    nghttp2_session_del(session);
    session = nullptr;
    return false;
  }
  return true;
}

void SessionState::destroy() noexcept {
  if (session != nullptr) {
    nghttp2_session_del(session);
    session = nullptr;
  }
  streams.for_each(
      [&](const std::int32_t&, Stream*& stream) { destroy_stream(alloc, stream); });
  streams.clear();
  ready_head = ready_tail = garbage = nullptr;
}

Stream* SessionState::create_stream(std::int32_t id) noexcept {
  if (streams.size() >= kHttp2DefaultMaxConcurrentStreams * 2) return nullptr;
  Stream* stream = allocate_stream(alloc);
  if (stream == nullptr) return nullptr;
  stream->id = id;
  streams.insert_or_assign(id, stream);
  return stream;
}

void SessionState::collect_garbage() noexcept {
  while (garbage != nullptr) {
    Stream* stream = garbage;
    garbage = stream->garbage_next;
    streams.erase(stream->id);
    destroy_stream(alloc, stream);
  }
}

void SessionState::release_current(const void*& current) noexcept {
  if (current == nullptr) return;
  auto* stream = static_cast<Stream*>(const_cast<void*>(current));
  current = nullptr;
  stream->borrowed = false;
  if (stream->closed) {
    streams.erase(stream->id);
    destroy_stream(alloc, stream);
  }
}

string_ref field_view(const Stream* stream, FieldRef ref) noexcept {
  if (stream == nullptr || !ref.present) return {};
  return string_ref{
      reinterpret_cast<const char*>(stream->header_bytes.data + ref.offset),
      ref.length};
}

bool ascii_iequal(string_ref a, string_ref b) noexcept {
  if (a.size != b.size) return false;
  for (std::uint32_t i = 0; i < a.size; ++i) {
    unsigned char x = static_cast<unsigned char>(a.data[i]);
    unsigned char y = static_cast<unsigned char>(b.data[i]);
    if (x >= 'A' && x <= 'Z') x = static_cast<unsigned char>(x + 32);
    if (y >= 'A' && y <= 'Z') y = static_cast<unsigned char>(y + 32);
    if (x != y) return false;
  }
  return true;
}

bool valid_view(string_ref value) noexcept {
  return value.size == 0 || value.data != nullptr;
}

nghttp2_nv make_nv(string_ref name, string_ref value) noexcept {
  return nghttp2_nv{
      reinterpret_cast<std::uint8_t*>(const_cast<char*>(name.data)),
      reinterpret_cast<std::uint8_t*>(const_cast<char*>(value.data)), name.size,
      value.size, NGHTTP2_NV_FLAG_NONE};
}

bool append_outbound(Stream& stream, string_ref body) noexcept {
  stream.outbound.size = 0;
  stream.outbound_sent = 0;
  return valid_view(body) &&
         stream.outbound.append(body.data, body.size,
                                static_cast<std::uint32_t>(kHttp2MaxBodyBytes));
}

bool valid_headers(const Http2Header* headers,
                   std::size_t header_count) noexcept {
  if (header_count > kHttp2MaxHeaders ||
      (header_count != 0 && headers == nullptr))
    return false;
  for (std::size_t i = 0; i < header_count; ++i) {
    if (!valid_view(headers[i].name) || !valid_view(headers[i].value) ||
        headers[i].name.size == 0 || headers[i].name.data[0] == ':')
      return false;
  }
  return true;
}

std::int32_t submit_request_state(
    SessionState& state, string_ref method, string_ref scheme,
    string_ref authority, string_ref path, const Http2Header* headers,
    std::size_t header_count, string_ref body) noexcept {
  if (state.server || state.fatal || !valid_view(method) ||
      !valid_view(scheme) || !valid_view(authority) || !valid_view(path) ||
      method.size == 0 || scheme.size == 0 || authority.size == 0 ||
      path.size == 0 || !valid_headers(headers, header_count) ||
      !valid_view(body) ||
      state.streams.size() >= kHttp2DefaultMaxConcurrentStreams * 2)
    return -1;

  Stream* stream = allocate_stream(state.alloc);
  if (stream == nullptr || !append_outbound(*stream, body)) {
    destroy_stream(state.alloc, stream);
    return -1;
  }
  nghttp2_nv fields[4 + kHttp2MaxHeaders];
  fields[0] = make_nv(string_ref{":method", 7}, method);
  fields[1] = make_nv(string_ref{":scheme", 7}, scheme);
  fields[2] = make_nv(string_ref{":authority", 10}, authority);
  fields[3] = make_nv(string_ref{":path", 5}, path);
  for (std::size_t i = 0; i < header_count; ++i)
    fields[i + 4] = make_nv(headers[i].name, headers[i].value);

  nghttp2_data_provider2 provider{};
  nghttp2_data_provider2* provider_ptr = nullptr;
  if (stream->outbound.size != 0) {
    provider.source.ptr = stream;
    provider.read_callback = read_outbound;
    provider_ptr = &provider;
  }
  const std::int32_t id = nghttp2_submit_request2(
      state.session, nullptr, fields, header_count + 4, provider_ptr, stream);
  if (id < 0) {
    destroy_stream(state.alloc, stream);
    return id;
  }
  stream->id = id;
  state.streams.insert_or_assign(id, stream);
  return id;
}

bool submit_response_state(SessionState& state, std::int32_t stream_id,
                           int status, const Http2Header* headers,
                           std::size_t header_count,
                           string_ref body) noexcept {
  if (!state.server || state.fatal || status < 100 || status > 999 ||
      !valid_headers(headers, header_count) || !valid_view(body))
    return false;
  Stream* stream = state.find_stream(stream_id);
  if (stream == nullptr || stream->response_submitted ||
      !append_outbound(*stream, body))
    return false;

  char status_text[4] = {
      static_cast<char>('0' + status / 100),
      static_cast<char>('0' + (status / 10) % 10),
      static_cast<char>('0' + status % 10), '\0'};
  nghttp2_nv fields[1 + kHttp2MaxHeaders];
  fields[0] =
      make_nv(string_ref{":status", 7}, string_ref{status_text, 3});
  for (std::size_t i = 0; i < header_count; ++i)
    fields[i + 1] = make_nv(headers[i].name, headers[i].value);

  nghttp2_data_provider2 provider{};
  nghttp2_data_provider2* provider_ptr = nullptr;
  if (stream->outbound.size != 0) {
    provider.source.ptr = stream;
    provider.read_callback = read_outbound;
    provider_ptr = &provider;
  }
  if (nghttp2_submit_response2(state.session, stream_id, fields,
                               header_count + 1, provider_ptr) != 0)
    return false;
  stream->response_submitted = true;
  return true;
}

sched::Task<bool> write_all_h2(mem::Allocator alloc,
                               HttpTransport& transport,
                               const std::uint8_t* data,
                               std::size_t size) noexcept {
  std::size_t offset = 0;
  while (offset < size) {
    const auto chunk = static_cast<std::uint32_t>(
        std::min<std::size_t>(size - offset,
                              (std::numeric_limits<std::uint32_t>::max)()));
    const std::int64_t n = co_await http_transport_write(
        alloc, transport, data + offset, chunk);
    if (n <= 0) co_return false;
    offset += static_cast<std::size_t>(n);
  }
  co_return true;
}

sched::Task<bool> flush_session(mem::Allocator alloc, SessionState& state,
                                HttpTransport& transport) noexcept {
  for (;;) {
    const std::uint8_t* data = nullptr;
    const nghttp2_ssize size =
        nghttp2_session_mem_send2(state.session, &data);
    if (size < 0) {
      state.fatal = true;
      co_return false;
    }
    if (size == 0) break;
    if (!(co_await write_all_h2(alloc, transport, data,
                                static_cast<std::size_t>(size)))) {
      state.fatal = true;
      co_return false;
    }
  }
  co_return true;
}

bool feed_session(SessionState& state, const std::uint8_t* data,
                  std::size_t size) noexcept {
  std::size_t offset = 0;
  while (offset < size) {
    const nghttp2_ssize used =
        nghttp2_session_mem_recv2(state.session, data + offset, size - offset);
    if (used <= 0) {
      state.fatal = true;
      return false;
    }
    offset += static_cast<std::size_t>(used);
  }
  state.collect_garbage();
  return true;
}

sched::Task<bool> pump_once(mem::Allocator alloc, SessionState& state,
                            HttpTransport& transport) noexcept {
  if (!(co_await flush_session(alloc, state, transport))) co_return false;
  const std::int64_t n = co_await http_transport_read(
      alloc, transport, state.input.data, state.input.cap);
  if (n <= 0) {
    state.fatal = true;
    co_return false;
  }
  if (!feed_session(state, state.input.data, static_cast<std::size_t>(n)))
    co_return false;
  co_return co_await flush_session(alloc, state, transport);
}

sched::Task<bool> flush_stream(mem::Allocator alloc, SessionState& state,
                               HttpTransport& transport,
                               Stream& stream) noexcept {
  if (!(co_await flush_session(alloc, state, transport))) co_return false;
  // DATA may stop at the peer's connection/stream window. Keep driving inbound
  // WINDOW_UPDATE frames until the fixed body has been handed to the transport.
  while (stream.outbound_sent < stream.outbound.size) {
    if (stream.closed || !(co_await pump_once(alloc, state, transport)))
      co_return false;
  }
  co_return true;
}

bool submit_reset(SessionState& state, std::int32_t id,
                  Http2Error error) noexcept {
  return nghttp2_submit_rst_stream(state.session, NGHTTP2_FLAG_NONE, id,
                                   static_cast<std::uint32_t>(error)) == 0;
}

bool submit_goaway(SessionState& state, Http2Error error) noexcept {
  const std::int32_t last =
      nghttp2_session_get_last_proc_stream_id(state.session);
  return nghttp2_submit_goaway(state.session, NGHTTP2_FLAG_NONE, last,
                               static_cast<std::uint32_t>(error), nullptr,
                               0) == 0;
}

}  // namespace

struct Http2ServerConn::Impl : SessionState {
  explicit Impl(mem::Allocator alloc) noexcept : SessionState(alloc, true) {}
};

struct Http2ClientConn::Impl : SessionState {
  explicit Impl(mem::Allocator alloc) noexcept : SessionState(alloc, false) {}
};

std::int32_t Http2MessageView::stream_id() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return stream == nullptr ? 0 : stream->id;
}
string_ref Http2MessageView::method() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return field_view(stream, stream == nullptr ? FieldRef{} : stream->method);
}
string_ref Http2MessageView::scheme() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return field_view(stream, stream == nullptr ? FieldRef{} : stream->scheme);
}
string_ref Http2MessageView::authority() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return field_view(stream, stream == nullptr ? FieldRef{} : stream->authority);
}
string_ref Http2MessageView::path() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return field_view(stream, stream == nullptr ? FieldRef{} : stream->path);
}
int Http2MessageView::status() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return stream == nullptr ? 0 : stream->status;
}
Http2Error Http2MessageView::stream_error() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return static_cast<Http2Error>(stream == nullptr ? 0 : stream->error_code);
}
std::size_t Http2MessageView::header_count() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return stream == nullptr ? 0 : stream->header_count;
}
string_ref Http2MessageView::header_name(std::size_t index) const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  if (stream == nullptr || index >= stream->header_count) return {};
  return field_view(stream, stream->headers[index].name);
}
string_ref Http2MessageView::header_value(std::size_t index) const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  if (stream == nullptr || index >= stream->header_count) return {};
  return field_view(stream, stream->headers[index].value);
}
string_ref Http2MessageView::header(string_ref name) const noexcept {
  for (std::size_t i = 0; i < header_count(); ++i)
    if (ascii_iequal(header_name(i), name)) return header_value(i);
  return {};
}
string_ref Http2MessageView::body() const noexcept {
  const auto* stream = static_cast<const Stream*>(stream_);
  return stream == nullptr
             ? string_ref{}
             : string_ref{reinterpret_cast<const char*>(stream->body.data),
                          stream->body.size};
}

namespace {

template <class Impl>
Impl* make_impl(mem::Allocator alloc) noexcept {
  void* p = alloc.allocate(sizeof(Impl), alignof(Impl));
  if (p == nullptr) return nullptr;
  auto* impl = new (p) Impl(alloc);
  if (!impl->initialize()) {
    impl->destroy();
    impl->~Impl();
    alloc.deallocate(impl, sizeof(Impl), alignof(Impl));
    return nullptr;
  }
  return impl;
}

template <class Impl>
void free_impl(mem::Allocator alloc, Impl* impl) noexcept {
  if (impl == nullptr) return;
  impl->destroy();
  impl->~Impl();
  alloc.deallocate(impl, sizeof(Impl), alignof(Impl));
}

}  // namespace

Http2ServerConn::Http2ServerConn(mem::Allocator alloc,
                                 TcpStream stream) noexcept
    : impl_(make_impl<Impl>(alloc)),
      transport_(std::move(stream)),
      alloc_(alloc) {}
#if defined(HYLE_IO_TLS)
Http2ServerConn::Http2ServerConn(mem::Allocator alloc,
                                 TlsStream stream) noexcept
    : impl_(stream.negotiated_protocol() == "h2" ? make_impl<Impl>(alloc)
                                                  : nullptr),
      transport_(std::move(stream)),
      alloc_(alloc) {}
#endif
Http2ServerConn::~Http2ServerConn() { free_impl(alloc_, impl_); }
Http2ServerConn::Http2ServerConn(Http2ServerConn&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)),
      transport_(std::move(other.transport_)),
      current_(other.current_),
      alloc_(other.alloc_) {
  other.current_.stream_ = nullptr;
}
Http2ServerConn& Http2ServerConn::operator=(Http2ServerConn&& other) noexcept {
  if (this == &other) return *this;
  free_impl(alloc_, impl_);
  impl_ = std::exchange(other.impl_, nullptr);
  transport_ = std::move(other.transport_);
  current_ = other.current_;
  alloc_ = other.alloc_;
  other.current_.stream_ = nullptr;
  return *this;
}

sched::Task<Http2MessageView*> http2_server_next(
    mem::Allocator alloc, Http2ServerConn& conn) noexcept {
  if (!conn.valid() || conn.impl_->fatal) co_return nullptr;
  conn.impl_->release_current(conn.current_.stream_);
  for (;;) {
    if (Stream* stream = conn.impl_->pop_ready()) {
      conn.current_.stream_ = stream;
      co_return &conn.current_;
    }
    if (!(co_await pump_once(alloc, *conn.impl_, conn.transport_)))
      co_return nullptr;
  }
}

sched::Task<Http2MessageView*> Http2ServerConn::next_request() noexcept {
  return http2_server_next(alloc_, *this);
}

sched::Task<bool> http2_server_respond(
    mem::Allocator alloc, Http2ServerConn& conn, std::int32_t stream_id,
    int status, const Http2Header* headers, std::size_t header_count,
    string_ref body) noexcept {
  if (!conn.valid() ||
      !submit_response_state(*conn.impl_, stream_id, status, headers,
                             header_count, body))
    co_return false;
  Stream* stream = conn.impl_->find_stream(stream_id);
  if (stream == nullptr) co_return false;
  co_return co_await flush_stream(alloc, *conn.impl_, conn.transport_, *stream);
}

sched::Task<bool> Http2ServerConn::respond(
    std::int32_t stream_id, int status, const Http2Header* headers,
    std::size_t header_count, string_ref body) noexcept {
  return http2_server_respond(alloc_, *this, stream_id, status, headers,
                              header_count, body);
}

sched::Task<bool> http2_server_reset(mem::Allocator alloc,
                                     Http2ServerConn& conn,
                                     std::int32_t stream_id,
                                     Http2Error error) noexcept {
  if (!conn.valid() || conn.impl_->fatal ||
      !submit_reset(*conn.impl_, stream_id, error))
    co_return false;
  co_return co_await flush_session(alloc, *conn.impl_, conn.transport_);
}
sched::Task<bool> Http2ServerConn::reset(std::int32_t stream_id,
                                         Http2Error error) noexcept {
  return http2_server_reset(alloc_, *this, stream_id, error);
}
sched::Task<bool> http2_server_shutdown(mem::Allocator alloc,
                                        Http2ServerConn& conn,
                                        Http2Error error) noexcept {
  if (!conn.valid() || conn.impl_->fatal ||
      !submit_goaway(*conn.impl_, error))
    co_return false;
  if (!(co_await flush_session(alloc, *conn.impl_, conn.transport_)))
    co_return false;
  co_return (co_await http_transport_shutdown(alloc, conn.transport_)) == 0;
}
sched::Task<bool> Http2ServerConn::shutdown(Http2Error error) noexcept {
  return http2_server_shutdown(alloc_, *this, error);
}

Http2ClientConn::Http2ClientConn(mem::Allocator alloc,
                                 TcpStream stream) noexcept
    : impl_(make_impl<Impl>(alloc)),
      transport_(std::move(stream)),
      alloc_(alloc) {}
#if defined(HYLE_IO_TLS)
Http2ClientConn::Http2ClientConn(mem::Allocator alloc,
                                 TlsStream stream) noexcept
    : impl_(stream.negotiated_protocol() == "h2" ? make_impl<Impl>(alloc)
                                                  : nullptr),
      transport_(std::move(stream)),
      alloc_(alloc) {}
#endif
Http2ClientConn::~Http2ClientConn() { free_impl(alloc_, impl_); }
Http2ClientConn::Http2ClientConn(Http2ClientConn&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)),
      transport_(std::move(other.transport_)),
      current_(other.current_),
      alloc_(other.alloc_) {
  other.current_.stream_ = nullptr;
}
Http2ClientConn& Http2ClientConn::operator=(Http2ClientConn&& other) noexcept {
  if (this == &other) return *this;
  free_impl(alloc_, impl_);
  impl_ = std::exchange(other.impl_, nullptr);
  transport_ = std::move(other.transport_);
  current_ = other.current_;
  alloc_ = other.alloc_;
  other.current_.stream_ = nullptr;
  return *this;
}

sched::Task<std::int32_t> http2_client_submit(
    mem::Allocator alloc, Http2ClientConn& conn, string_ref method,
    string_ref scheme, string_ref authority, string_ref path,
    const Http2Header* headers, std::size_t header_count,
    string_ref body) noexcept {
  if (!conn.valid()) co_return -1;
  const std::int32_t id =
      submit_request_state(*conn.impl_, method, scheme, authority, path,
                           headers, header_count, body);
  if (id < 0) co_return id;
  Stream* stream = conn.impl_->find_stream(id);
  if (stream == nullptr) co_return -1;
  if (!(co_await flush_stream(alloc, *conn.impl_, conn.transport_, *stream)))
    co_return -1;
  co_return id;
}

sched::Task<std::int32_t> Http2ClientConn::submit_request(
    string_ref method, string_ref scheme, string_ref authority,
    string_ref path, const Http2Header* headers, std::size_t header_count,
    string_ref body) noexcept {
  return http2_client_submit(alloc_, *this, method, scheme, authority, path,
                             headers, header_count, body);
}

sched::Task<Http2MessageView*> http2_client_next(
    mem::Allocator alloc, Http2ClientConn& conn) noexcept {
  if (!conn.valid() || conn.impl_->fatal) co_return nullptr;
  conn.impl_->release_current(conn.current_.stream_);
  for (;;) {
    if (Stream* stream = conn.impl_->pop_ready()) {
      conn.current_.stream_ = stream;
      co_return &conn.current_;
    }
    if (!(co_await pump_once(alloc, *conn.impl_, conn.transport_)))
      co_return nullptr;
  }
}
sched::Task<Http2MessageView*> Http2ClientConn::next_response() noexcept {
  return http2_client_next(alloc_, *this);
}

sched::Task<bool> http2_client_reset(mem::Allocator alloc,
                                     Http2ClientConn& conn,
                                     std::int32_t stream_id,
                                     Http2Error error) noexcept {
  if (!conn.valid() || conn.impl_->fatal ||
      !submit_reset(*conn.impl_, stream_id, error))
    co_return false;
  co_return co_await flush_session(alloc, *conn.impl_, conn.transport_);
}
sched::Task<bool> Http2ClientConn::reset(std::int32_t stream_id,
                                         Http2Error error) noexcept {
  return http2_client_reset(alloc_, *this, stream_id, error);
}
sched::Task<bool> http2_client_shutdown(mem::Allocator alloc,
                                        Http2ClientConn& conn,
                                        Http2Error error) noexcept {
  if (!conn.valid() || conn.impl_->fatal ||
      !submit_goaway(*conn.impl_, error))
    co_return false;
  if (!(co_await flush_session(alloc, *conn.impl_, conn.transport_)))
    co_return false;
  co_return (co_await http_transport_shutdown(alloc, conn.transport_)) == 0;
}
sched::Task<bool> Http2ClientConn::shutdown(Http2Error error) noexcept {
  return http2_client_shutdown(alloc_, *this, error);
}

namespace {

void* c_alloc(void* state, std::size_t size,
              std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  return backing->alloc(backing->state, size, alignment);
}
void c_free(void* state, void* pointer, std::size_t size,
            std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  backing->free(backing->state, pointer, size, alignment);
}
void* c_realloc(void* state, void* pointer, std::size_t old_size,
                std::size_t new_size, std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  return backing->realloc(backing->state, pointer, old_size, new_size,
                          alignment);
}

struct CHttp2Session {
  explicit CHttp2Session(hyle_allocator a, bool server) noexcept
      : backing(a),
        alloc(c_alloc, c_free, c_realloc, &backing),
        state(alloc, server) {}

  hyle_allocator backing;
  mem::Allocator alloc;
  SessionState state;
  const void* current = nullptr;
};

bool c_view(const void* data, std::size_t size, string_ref& out) noexcept {
  if ((size != 0 && data == nullptr) ||
      size > (std::numeric_limits<std::uint32_t>::max)())
    return false;
  out = string_ref{static_cast<const char*>(data),
                   static_cast<std::uint32_t>(size)};
  return true;
}

bool c_headers(const hyle_io_http2_header* input, std::size_t count,
               Http2Header* output) noexcept {
  if (count > kHttp2MaxHeaders || (count != 0 && input == nullptr))
    return false;
  for (std::size_t i = 0; i < count; ++i) {
    if (!c_view(input[i].name, input[i].name_size, output[i].name) ||
        !c_view(input[i].value, input[i].value_size, output[i].value))
      return false;
  }
  return true;
}

CHttp2Session* c_http2_create(hyle_allocator backing, bool server) noexcept {
  if (backing.alloc == nullptr || backing.free == nullptr ||
      backing.realloc == nullptr)
    return nullptr;
  void* p = backing.alloc(backing.state, sizeof(CHttp2Session),
                          alignof(CHttp2Session));
  if (p == nullptr) return nullptr;
  auto* session = new (p) CHttp2Session(backing, server);
  if (!session->state.initialize()) {
    session->state.destroy();
    session->~CHttp2Session();
    backing.free(backing.state, p, sizeof(CHttp2Session),
                 alignof(CHttp2Session));
    return nullptr;
  }
  return session;
}

void c_http2_destroy(CHttp2Session* session) noexcept {
  if (session == nullptr) return;
  hyle_allocator backing = session->backing;
  session->state.destroy();
  session->~CHttp2Session();
  backing.free(backing.state, session, sizeof(CHttp2Session),
               alignof(CHttp2Session));
}

}  // namespace

}  // namespace hyle::io

struct hyle_io_http2_session_o {};

namespace {

hyle::io::CHttp2Session* c_session(hyle_io_http2_session* session) noexcept {
  return reinterpret_cast<hyle::io::CHttp2Session*>(session);
}
const hyle::io::CHttp2Session* c_session(
    const hyle_io_http2_session* session) noexcept {
  return reinterpret_cast<const hyle::io::CHttp2Session*>(session);
}

}  // namespace

extern "C" {

int hyle_io_http2_available(void) {
  hyle::io::c_detail::clear_error();
  return 1;
}

hyle_io_http2_session* hyle_io_http2_session_create(
    hyle_allocator backing, hyle_io_http2_role role) {
  if (role != HYLE_IO_HTTP2_CLIENT && role != HYLE_IO_HTTP2_SERVER) {
    hyle::io::c_detail::set_error("invalid HTTP/2 session role");
    return nullptr;
  }
  if (!hyle::io::c_detail::normalize_allocator(&backing)) return nullptr;
  auto* out = reinterpret_cast<hyle_io_http2_session*>(
      hyle::io::c_http2_create(backing, role == HYLE_IO_HTTP2_SERVER));
  if (out == nullptr) {
    hyle::io::c_detail::set_error("HTTP/2 session allocation or initialization failed");
  } else {
    hyle::io::c_detail::clear_error();
  }
  return out;
}

void hyle_io_http2_session_destroy(hyle_io_http2_session* session) {
  hyle::io::c_http2_destroy(c_session(session));
}

int hyle_io_http2_session_valid(const hyle_io_http2_session* session) {
  const auto* value = c_session(session);
  return value != nullptr && value->state.session != nullptr &&
         !value->state.fatal;
}

int hyle_io_http2_feed(hyle_io_http2_session* session, const void* data,
                       size_t size) {
  auto* value = c_session(session);
  if (value == nullptr || value->state.fatal ||
      (size != 0 && data == nullptr)) {
    hyle::io::c_detail::set_error("invalid or failed HTTP/2 session/input");
    return 0;
  }
  if (size != 0 &&
      !hyle::io::feed_session(
          value->state, static_cast<const std::uint8_t*>(data), size)) {
    hyle::io::c_detail::set_error("HTTP/2 protocol or resource failure while feeding input");
    return 0;
  }
  hyle::io::c_detail::clear_error();
  return 1;
}

size_t hyle_io_http2_next_output(hyle_io_http2_session* session,
                                 const void** data) {
  auto* value = c_session(session);
  if (data == nullptr) {
    hyle::io::c_detail::set_error("HTTP/2 output pointer is null");
    return SIZE_MAX;
  }
  *data = nullptr;
  if (value == nullptr || value->state.fatal) {
    hyle::io::c_detail::set_error("invalid or failed HTTP/2 session");
    return SIZE_MAX;
  }
  const std::uint8_t* encoded = nullptr;
  const nghttp2_ssize size =
      nghttp2_session_mem_send2(value->state.session, &encoded);
  if (size < 0) {
    value->state.fatal = true;
    hyle::io::c_detail::set_error("HTTP/2 encoder failure");
    return SIZE_MAX;
  }
  *data = encoded;
  hyle::io::c_detail::clear_error();
  return static_cast<size_t>(size);
}

int32_t hyle_io_http2_submit_request(
    hyle_io_http2_session* session, const char* method, size_t method_size,
    const char* scheme, size_t scheme_size, const char* authority,
    size_t authority_size, const char* path, size_t path_size,
    const hyle_io_http2_header* headers, size_t header_count,
    const void* body, size_t body_size) {
  auto* value = c_session(session);
  if (value == nullptr || value->state.fatal) {
    hyle::io::c_detail::set_error("invalid or failed HTTP/2 session");
    return -1;
  }
  hyle::io::Http2Header converted[hyle::io::kHttp2MaxHeaders];
  hyle::container::string_ref m, s, a, p, b;
  if (!hyle::io::c_view(method, method_size, m) ||
      !hyle::io::c_view(scheme, scheme_size, s) ||
      !hyle::io::c_view(authority, authority_size, a) ||
      !hyle::io::c_view(path, path_size, p) ||
      !hyle::io::c_view(body, body_size, b) ||
      !hyle::io::c_headers(headers, header_count, converted)) {
    hyle::io::c_detail::set_error("invalid HTTP/2 request fields or resource limits");
    return -1;
  }
  const int32_t id = hyle::io::submit_request_state(
      value->state, m, s, a, p, converted, header_count, b);
  if (id < 0)
    hyle::io::c_detail::set_error("HTTP/2 request submission failed");
  else
    hyle::io::c_detail::clear_error();
  return id;
}

int hyle_io_http2_submit_response(
    hyle_io_http2_session* session, int32_t stream_id, int status,
    const hyle_io_http2_header* headers, size_t header_count,
    const void* body, size_t body_size) {
  auto* value = c_session(session);
  if (value == nullptr || value->state.fatal) {
    hyle::io::c_detail::set_error("invalid or failed HTTP/2 session");
    return 0;
  }
  hyle::io::Http2Header converted[hyle::io::kHttp2MaxHeaders];
  hyle::container::string_ref b;
  if (!hyle::io::c_view(body, body_size, b) ||
      !hyle::io::c_headers(headers, header_count, converted)) {
    hyle::io::c_detail::set_error("invalid HTTP/2 response fields or resource limits");
    return 0;
  }
  const bool ok = hyle::io::submit_response_state(
      value->state, stream_id, status, converted, header_count, b);
  if (ok)
    hyle::io::c_detail::clear_error();
  else
    hyle::io::c_detail::set_error("HTTP/2 response submission failed");
  return ok ? 1 : 0;
}

int hyle_io_http2_reset(hyle_io_http2_session* session, int32_t stream_id,
                        uint32_t error_code) {
  auto* value = c_session(session);
  const bool ok = value != nullptr && !value->state.fatal &&
                  hyle::io::submit_reset(
                      value->state, stream_id,
                      static_cast<hyle::io::Http2Error>(error_code));
  if (ok)
    hyle::io::c_detail::clear_error();
  else
    hyle::io::c_detail::set_error("HTTP/2 stream reset submission failed");
  return ok ? 1 : 0;
}

int hyle_io_http2_goaway(hyle_io_http2_session* session,
                         uint32_t error_code) {
  auto* value = c_session(session);
  const bool ok = value != nullptr && !value->state.fatal &&
                  hyle::io::submit_goaway(
                      value->state,
                      static_cast<hyle::io::Http2Error>(error_code));
  if (ok)
    hyle::io::c_detail::clear_error();
  else
    hyle::io::c_detail::set_error("HTTP/2 GOAWAY submission failed");
  return ok ? 1 : 0;
}

int hyle_io_http2_next_message(hyle_io_http2_session* session,
                               hyle_io_http2_message* message) {
  auto* value = c_session(session);
  if (value == nullptr || message == nullptr) {
    hyle::io::c_detail::set_error("HTTP/2 message poll requires session and output");
    return -1;
  }
  value->state.release_current(value->current);
  hyle::io::Stream* stream = value->state.pop_ready();
  if (stream == nullptr) {
    if (value->state.fatal) {
      hyle::io::c_detail::set_error("HTTP/2 session failed");
      return -1;
    }
    hyle::io::c_detail::clear_error();
    return 0;
  }
  value->current = stream;
  const auto method = hyle::io::field_view(stream, stream->method);
  const auto scheme = hyle::io::field_view(stream, stream->scheme);
  const auto authority = hyle::io::field_view(stream, stream->authority);
  const auto path = hyle::io::field_view(stream, stream->path);
  *message = hyle_io_http2_message{
      stream->id,
      stream->status,
      stream->error_code,
      static_cast<uint32_t>(stream->header_count),
      method.data,
      method.size,
      scheme.data,
      scheme.size,
      authority.data,
      authority.size,
      path.data,
      path.size,
      stream->body.data,
      stream->body.size};
  hyle::io::c_detail::clear_error();
  return 1;
}

int hyle_io_http2_message_header(const hyle_io_http2_session* session,
                                 size_t index,
                                 hyle_io_http2_header* header) {
  const auto* value = c_session(session);
  if (value == nullptr || value->current == nullptr || header == nullptr) {
    hyle::io::c_detail::set_error("HTTP/2 header lookup requires a current message and output");
    return 0;
  }
  const auto* stream =
      static_cast<const hyle::io::Stream*>(value->current);
  if (index >= stream->header_count) {
    hyle::io::c_detail::set_error("HTTP/2 header index is out of range");
    return 0;
  }
  const auto name =
      hyle::io::field_view(stream, stream->headers[index].name);
  const auto field_value =
      hyle::io::field_view(stream, stream->headers[index].value);
  *header = hyle_io_http2_header{name.data, name.size, field_value.data,
                                 field_value.size};
  hyle::io::c_detail::clear_error();
  return 1;
}

}  // extern "C"
