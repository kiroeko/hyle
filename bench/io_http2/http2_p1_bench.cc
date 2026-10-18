// HTTP/2 protocol/payload comparison for multiplexed inference streams.
//
// Both routes own one fresh nonblocking Unix socket pair and execute client and
// server HTTP/2 state on one persistent worker. They submit two concurrent
// POST streams at a time, retain and validate every request byte, complete the
// responses in reverse stream order, incrementally consume every response
// byte, update both flow-control windows, and retire all connection state. The
// HYLE route uses only public HYLE APIs. The peer uses the pinned public
// nghttp2 session API directly; it does not replace HPACK or frame machinery.
// By default the direct peer advances both endpoints in one manual loop;
// --paired gives both routes two endpoint Tasks on the same Scheduler and
// aligns allocator and DATA-copy choices. --paired-self runs HYLE on both
// arms to observe comparison noise. Neither mode isolates wrapper overhead:
// HYLE retains generic public-API responsibilities absent from the fixed peer.

#include <nanobench.h>
#include <nghttp2/nghttp2.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include "hyle/io/http2.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::Status;
using hyle::io::Http2ClientSession;
using hyle::io::Http2IncomingRequest;
using hyle::io::Http2IncomingResponse;
using hyle::io::Http2RequestHead;
using hyle::io::Http2ResponseHead;
using hyle::io::Http2ServerSession;
using hyle::io::IoContext;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::size_t kWireBytes = 64 * 1024;
constexpr std::size_t kBodyReadBytes = 16 * 1024;
constexpr std::size_t kStreamWindowBytes = 64 * 1024;
constexpr std::size_t kNativeAllocationAlignment = alignof(std::max_align_t);
constexpr std::size_t kNativeAllocationPrefix =
    (sizeof(std::size_t) + kNativeAllocationAlignment - 1) &
    ~(kNativeAllocationAlignment - 1);

void close_descriptor(int descriptor) noexcept {
  if (descriptor >= 0)
    (void)::close(descriptor);
}

class SocketPair {
public:
  SocketPair() noexcept = default;
  SocketPair(int client, int server) noexcept
      : client_(client), server_(server) {}
  ~SocketPair() {
    close_descriptor(client_);
    close_descriptor(server_);
  }
  SocketPair(SocketPair &&other) noexcept
      : client_(std::exchange(other.client_, -1)),
        server_(std::exchange(other.server_, -1)) {}
  SocketPair &operator=(SocketPair &&other) noexcept {
    if (this != &other) {
      close_descriptor(client_);
      close_descriptor(server_);
      client_ = std::exchange(other.client_, -1);
      server_ = std::exchange(other.server_, -1);
    }
    return *this;
  }
  SocketPair(const SocketPair &) = delete;
  SocketPair &operator=(const SocketPair &) = delete;

  bool valid() const noexcept { return client_ >= 0 && server_ >= 0; }
  int release_client() noexcept { return std::exchange(client_, -1); }
  int release_server() noexcept { return std::exchange(server_, -1); }

private:
  int client_ = -1;
  int server_ = -1;
};

SocketPair make_socket_pair() noexcept {
  int descriptors[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, descriptors) != 0)
    return {};
  constexpr int kSocketBufferBytes = 4 * 1024;
  for (int descriptor : descriptors) {
    const int flags = ::fcntl(descriptor, F_GETFL, 0);
    if (flags < 0 ||
        ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0 ||
        ::fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) {
      close_descriptor(descriptors[0]);
      close_descriptor(descriptors[1]);
      return {};
    }
    if (::setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF,
                     &kSocketBufferBytes, sizeof(kSocketBufferBytes)) != 0 ||
        ::setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF,
                     &kSocketBufferBytes, sizeof(kSocketBufferBytes)) != 0) {
      close_descriptor(descriptors[0]);
      close_descriptor(descriptors[1]);
      return {};
    }
  }
  return {descriptors[0], descriptors[1]};
}

std::span<const std::byte> as_bytes(std::string_view value) noexcept {
  return {reinterpret_cast<const std::byte *>(value.data()), value.size()};
}

struct Workload {
  const char *title = nullptr;
  int streams = 0;
  std::size_t request_bytes = 0;
  std::size_t response_bytes = 0;
  std::uint32_t connection_window = 1024 * 1024;
  std::string request_body;
  std::string response_body;
};

struct HyleRunState {
  explicit HyleRunState(const Workload &selected) noexcept
      : workload(&selected) {}

  const Workload *workload = nullptr;
  hyle::sched::Event client_consumed;
};

Workload make_workload(const char *title, int streams,
                       std::size_t request_bytes,
                       std::size_t response_bytes,
                       std::uint32_t connection_window = 1024 * 1024) {
  Workload result;
  result.title = title;
  result.streams = streams;
  result.request_bytes = request_bytes;
  result.response_bytes = response_bytes;
  result.connection_window = connection_window;
  result.request_body.assign(request_bytes, 'Q');
  result.response_body.assign(response_bytes, 'T');
  return result;
}

Task<int> hyle_server(AllocatorRef allocator, TcpStream stream,
                      HyleRunState *run) noexcept {
  const Workload *workload = run->workload;
  hyle::io::Http2Limits limits;
  limits.connection_receive_window = workload->connection_window;
  auto made = Http2ServerSession::create(allocator, std::move(stream), limits);
  if (!made.has_value())
    co_return -1;
  Http2ServerSession session = std::move(*made);
  for (int base = 0; base < workload->streams; base += 2) {
    std::array<Http2IncomingRequest, 2> incoming;
    for (auto &request : incoming) {
      auto accepted = co_await session.accept();
      if (!accepted.has_value() || !accepted->has_value())
        co_return -1;
      request = std::move(**accepted);
      if (request.head.method() != "POST" ||
          request.head.path() != "/v1/completions")
        co_return -1;
      std::size_t received = 0;
      auto verify =
          [&](const hyle::io::HttpBodyChunkView &next) noexcept -> Status {
        if (received + next.bytes.size() > workload->request_body.size() ||
            std::memcmp(next.bytes.data(),
                        workload->request_body.data() + received,
                        next.bytes.size()) != 0)
          return Status::DataLoss("HTTP/2 benchmark body mismatch");
        received += next.bytes.size();
        return Status::Ok();
      };
      Status read = co_await request.body.consume(verify, kBodyReadBytes);
      if (!read.ok() || received != workload->request_body.size())
        co_return -1;
    }
    for (int index = 1; index >= 0; --index) {
      Status sent = co_await session.send_response(
          incoming[static_cast<std::size_t>(index)].stream_id,
          Http2ResponseHead{}, as_bytes(workload->response_body));
      if (!sent.ok())
        co_return -1;
    }
  }
  if (!run->client_consumed.wait_for(std::chrono::seconds(5)))
    co_return -1;
  co_return workload->streams;
}

Task<int> hyle_client(AllocatorRef allocator, TcpStream stream,
                      HyleRunState *run) noexcept {
  const Workload *workload = run->workload;
  hyle::io::Http2Limits limits;
  limits.connection_receive_window = workload->connection_window;
  auto made = Http2ClientSession::create(allocator, std::move(stream), limits);
  if (!made.has_value())
    co_return -1;
  Http2ClientSession session = std::move(*made);
  const Http2RequestHead head{.method = "POST",
                              .scheme = "http",
                              .authority = "localhost",
                              .path = "/v1/completions"};
  for (int base = 0; base < workload->streams; base += 2) {
    std::array<hyle::io::Http2StreamId, 2> submitted;
    for (auto &id : submitted) {
      auto result = co_await session.send_request(
          head, as_bytes(workload->request_body));
      if (!result.has_value())
        co_return -1;
      id = *result;
    }
    std::array<hyle::io::Http2StreamId, 2> completed;
    for (auto &id : completed) {
      auto next = co_await session.next_response();
      if (!next.has_value() || !next->has_value())
        co_return -1;
      Http2IncomingResponse response = std::move(**next);
      if (response.head.status() != 200)
        co_return -1;
      id = response.stream_id;
      std::size_t received = 0;
      auto verify =
          [&](const hyle::io::HttpBodyChunkView &next) noexcept -> Status {
        if (received + next.bytes.size() > workload->response_body.size() ||
            std::memcmp(next.bytes.data(),
                        workload->response_body.data() + received,
                        next.bytes.size()) != 0)
          return Status::DataLoss("HTTP/2 benchmark body mismatch");
        received += next.bytes.size();
        return Status::Ok();
      };
      Status read = co_await response.body.consume(verify, kBodyReadBytes);
      if (!read.ok() || received != workload->response_body.size())
        co_return -1;
    }
    if (completed[0] != submitted[1] || completed[1] != submitted[0])
      co_return -1;
  }
  run->client_consumed.signal();
  co_return workload->streams;
}

class HyleHarness {
public:
  HyleHarness() : scheduler_(1, system_allocator(), 1024), io_(scheduler_) {
    if (!io_.valid())
      std::abort();
  }
  ~HyleHarness() { scheduler_.shutdown(); }

  Scheduler &scheduler() noexcept { return scheduler_; }
  IoContext &io() noexcept { return io_; }

  int run(const Workload &workload) {
    SocketPair pair = make_socket_pair();
    if (!pair.valid())
      return -1;
    auto adopted_client =
        TcpStream::adopt(io_, kOwner, pair.release_client());
    if (!adopted_client)
      return -1;
    auto adopted_server =
        TcpStream::adopt(io_, kOwner, pair.release_server());
    if (!adopted_server)
      return -1;
    TcpStream client = std::move(*adopted_client);
    TcpStream server = std::move(*adopted_server);
    HyleRunState state(workload);

    auto server_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_server(system_allocator(), std::move(server), &state));
    auto client_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_client(system_allocator(), std::move(client), &state));
    auto client_result = client_join.join();
    auto server_result = server_join.join();
    if (!client_result || !server_result ||
        *client_result != workload.streams ||
        *server_result != workload.streams)
      return -1;
    return workload.streams;
  }

private:
  Scheduler scheduler_;
  IoContext io_;
};

// Direct nghttp2 peer -------------------------------------------------------

struct RawHeader {
  std::string name;
  std::string value;
};

struct RawStream {
  std::int32_t id = 0;
  std::size_t sent = 0;
  std::size_t received = 0;
  std::size_t inbound_start = 0;
  std::vector<std::uint8_t, hyle::mem::SystemStdAllocator<std::uint8_t>> inbound;
  std::vector<RawHeader, hyle::mem::SystemStdAllocator<RawHeader>> headers;
  bool input_end_received = false;
  bool input_done = false;
  bool queued_for_read = false;
  bool response_submitted = false;
  bool output_done = false;
};

struct RawWire;

struct RawEndpoint {
  const Workload *workload = nullptr;
  AllocatorRef allocator = system_allocator();
  nghttp2_mem ngmem{};
  nghttp2_session *session = nullptr;
  std::vector<RawStream, hyle::mem::SystemStdAllocator<RawStream>> streams;
  std::vector<RawStream *, hyle::mem::SystemStdAllocator<RawStream *>> readable;
  std::size_t readable_head = 0;
  bool server = false;
  bool failed = false;
  int completed = 0;
  std::size_t next_response_pair = 0;
  RawWire *encoding_wire = nullptr;
};

// Match the public session's SystemAllocator bridge, including size prefixes
// and calloc/realloc semantics. The fixed-trace containers still differ from
// the general HTTP/2 API's state; RawHeader strings keep their std::string SSO.
void *raw_h2_malloc(std::size_t size, void *user) noexcept {
  auto *endpoint = static_cast<RawEndpoint *>(user);
  if (size > (std::numeric_limits<std::size_t>::max)() -
                 kNativeAllocationPrefix)
    return nullptr;
  const std::size_t total = size + kNativeAllocationPrefix;
  void *base = endpoint->allocator.allocate(total, kNativeAllocationAlignment);
  if (base == nullptr)
    return nullptr;
  *static_cast<std::size_t *>(base) = total;
  return static_cast<std::uint8_t *>(base) + kNativeAllocationPrefix;
}

void raw_h2_free(void *pointer, void *user) noexcept {
  if (pointer == nullptr)
    return;
  auto *endpoint = static_cast<RawEndpoint *>(user);
  void *base = static_cast<std::uint8_t *>(pointer) - kNativeAllocationPrefix;
  endpoint->allocator.deallocate(base, *static_cast<std::size_t *>(base),
                                  kNativeAllocationAlignment);
}

void *raw_h2_calloc(std::size_t count, std::size_t size, void *user) noexcept {
  if (size != 0 && count > (std::numeric_limits<std::size_t>::max)() / size)
    return nullptr;
  const std::size_t bytes = count * size;
  void *result = raw_h2_malloc(bytes, user);
  if (result != nullptr)
    std::memset(result, 0, bytes);
  return result;
}

void *raw_h2_realloc(void *pointer, std::size_t size, void *user) noexcept {
  if (pointer == nullptr)
    return raw_h2_malloc(size, user);
  if (size == 0) {
    raw_h2_free(pointer, user);
    return nullptr;
  }
  if (size > (std::numeric_limits<std::size_t>::max)() -
                 kNativeAllocationPrefix)
    return nullptr;
  auto *endpoint = static_cast<RawEndpoint *>(user);
  void *base = static_cast<std::uint8_t *>(pointer) - kNativeAllocationPrefix;
  const std::size_t old_total = *static_cast<std::size_t *>(base);
  const std::size_t new_total = size + kNativeAllocationPrefix;
  void *replacement = endpoint->allocator.reallocate(
      base, old_total, new_total, kNativeAllocationAlignment);
  if (replacement == nullptr)
    return nullptr;
  *static_cast<std::size_t *>(replacement) = new_total;
  return static_cast<std::uint8_t *>(replacement) + kNativeAllocationPrefix;
}

bool raw_queue_readable(RawEndpoint &endpoint, RawStream &stream) noexcept {
  if (stream.queued_for_read)
    return true;
  try {
    endpoint.readable.push_back(&stream);
  } catch (...) {
    endpoint.failed = true;
    return false;
  }
  stream.queued_for_read = true;
  return true;
}

nghttp2_nv raw_nv(const char *name, const char *value) noexcept {
  return {reinterpret_cast<std::uint8_t *>(const_cast<char *>(name)),
          reinterpret_cast<std::uint8_t *>(const_cast<char *>(value)),
          std::strlen(name), std::strlen(value), NGHTTP2_NV_FLAG_NONE};
}

nghttp2_ssize raw_provider(nghttp2_session *, std::int32_t stream_id,
                           std::uint8_t *, std::size_t length,
                           std::uint32_t *flags,
                           nghttp2_data_source *source, void *user_data) {
  auto *endpoint = static_cast<RawEndpoint *>(user_data);
  auto *stream = source == nullptr ? nullptr
                                   : static_cast<RawStream *>(source->ptr);
  if (endpoint == nullptr || stream == nullptr || stream->id != stream_id)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  const std::string &body = endpoint->server ? endpoint->workload->response_body
                                             : endpoint->workload->request_body;
  if (stream->sent > body.size()) {
    endpoint->failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  const std::size_t remaining = body.size() - stream->sent;
  const std::size_t count = (std::min)(remaining, length);
  stream->sent += count;
  *flags |= NGHTTP2_DATA_FLAG_NO_COPY;
  if (stream->sent == body.size()) {
    *flags |= NGHTTP2_DATA_FLAG_EOF;
    stream->output_done = true;
  }
  return static_cast<nghttp2_ssize>(count);
}

bool raw_submit_response(RawEndpoint &endpoint, RawStream &stream) noexcept {
  char length[32]{};
  std::snprintf(length, sizeof(length), "%zu",
                endpoint.workload->response_bytes);
  std::array<nghttp2_nv, 2> headers = {
      raw_nv(":status", "200"), raw_nv("content-length", length)};
  nghttp2_data_provider2 provider{};
  provider.source.ptr = &stream;
  provider.read_callback = raw_provider;
  if (nghttp2_submit_response2(endpoint.session, stream.id, headers.data(),
                               headers.size(), &provider) != 0)
    return false;
  stream.response_submitted = true;
  return true;
}

bool raw_submit_request(RawEndpoint &endpoint) noexcept {
  if (endpoint.server ||
      endpoint.streams.size() >=
          static_cast<std::size_t>(endpoint.workload->streams))
    return true;
  endpoint.streams.emplace_back();
  RawStream &stream = endpoint.streams.back();
  char length[32]{};
  std::snprintf(length, sizeof(length), "%zu",
                endpoint.workload->request_bytes);
  std::array<nghttp2_nv, 5> headers = {
      raw_nv(":method", "POST"), raw_nv(":scheme", "http"),
      raw_nv(":authority", "localhost"),
      raw_nv(":path", "/v1/completions"),
      raw_nv("content-length", length)};
  nghttp2_data_provider2 provider{};
  provider.source.ptr = &stream;
  provider.read_callback = raw_provider;
  const std::int32_t id = nghttp2_submit_request2(
      endpoint.session, nullptr, headers.data(), headers.size(), &provider,
      &stream);
  if (id < 0)
    return false;
  stream.id = id;
  return true;
}

int raw_begin_headers(nghttp2_session *session, const nghttp2_frame *frame,
                      void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  if (!endpoint.server || frame->hd.type != NGHTTP2_HEADERS ||
      frame->headers.cat != NGHTTP2_HCAT_REQUEST)
    return 0;
  if (endpoint.streams.size() >=
      static_cast<std::size_t>(endpoint.workload->streams)) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  endpoint.streams.emplace_back();
  RawStream &stream = endpoint.streams.back();
  stream.id = frame->hd.stream_id;
  nghttp2_session_set_stream_user_data(session, stream.id, &stream);
  return 0;
}

int raw_header(nghttp2_session *session, const nghttp2_frame *frame,
               const std::uint8_t *name, std::size_t name_length,
               const std::uint8_t *value, std::size_t value_length,
               std::uint8_t, void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  auto *stream = static_cast<RawStream *>(
      nghttp2_session_get_stream_user_data(session, frame->hd.stream_id));
  if (stream == nullptr) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  try {
    stream->headers.push_back(
        {std::string(reinterpret_cast<const char *>(name), name_length),
         std::string(reinterpret_cast<const char *>(value), value_length)});
  } catch (...) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  return 0;
}

int raw_data_chunk(nghttp2_session *session, std::uint8_t,
                   std::int32_t stream_id, const std::uint8_t *data,
                   std::size_t length, void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  auto *stream = static_cast<RawStream *>(
      nghttp2_session_get_stream_user_data(session, stream_id));
  if (stream == nullptr) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  const std::string &expected_body =
      endpoint.server ? endpoint.workload->request_body
                      : endpoint.workload->response_body;
  const std::size_t buffered = stream->inbound.size() - stream->inbound_start;
  if (stream->received > expected_body.size() ||
      length > expected_body.size() - stream->received ||
      buffered > expected_body.size() - stream->received - length) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (stream->inbound.capacity() == 0) {
    try {
      stream->inbound.reserve(kStreamWindowBytes);
    } catch (...) {
      endpoint.failed = true;
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  if (length > stream->inbound.capacity() - stream->inbound.size() &&
      stream->inbound_start != 0) {
    std::memmove(stream->inbound.data(),
                 stream->inbound.data() + stream->inbound_start, buffered);
    stream->inbound.resize(buffered);
    stream->inbound_start = 0;
  }
  if (length > stream->inbound.capacity() - stream->inbound.size()) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  try {
    if (length != 0)
      stream->inbound.insert(stream->inbound.end(), data, data + length);
  } catch (...) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (!raw_queue_readable(endpoint, *stream))
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  return 0;
}

int raw_frame_recv(nghttp2_session *session, const nghttp2_frame *frame,
                   void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  if ((frame->hd.type != NGHTTP2_HEADERS &&
       frame->hd.type != NGHTTP2_DATA) ||
      (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) == 0)
    return 0;
  auto *stream = static_cast<RawStream *>(
      nghttp2_session_get_stream_user_data(session, frame->hd.stream_id));
  if (stream == nullptr)
    return 0;
  stream->input_end_received = true;
  return raw_queue_readable(endpoint, *stream)
             ? 0
             : NGHTTP2_ERR_CALLBACK_FAILURE;
}

bool raw_has_header(const RawStream &stream, std::string_view name,
                    std::string_view value) noexcept {
  for (const RawHeader &header : stream.headers) {
    if (header.name == name && header.value == value)
      return true;
  }
  return false;
}

bool raw_finish_input(RawEndpoint &endpoint, RawStream &stream) noexcept {
  char expected_length[32]{};
  std::snprintf(expected_length, sizeof(expected_length), "%zu",
                endpoint.server ? endpoint.workload->request_bytes
                                : endpoint.workload->response_bytes);
  if (stream.received != (endpoint.server ? endpoint.workload->request_bytes
                                          : endpoint.workload->response_bytes) ||
      !raw_has_header(stream, "content-length", expected_length) ||
      (endpoint.server &&
       (!raw_has_header(stream, ":method", "POST") ||
        !raw_has_header(stream, ":path", "/v1/completions"))) ||
      (!endpoint.server && !raw_has_header(stream, ":status", "200"))) {
    endpoint.failed = true;
    return false;
  }
  if (!endpoint.server) {
    const std::size_t pair_base =
        static_cast<std::size_t>(endpoint.completed / 2) * 2;
    const std::size_t expected_index =
        pair_base + (endpoint.completed % 2 == 0 ? 1u : 0u);
    if (expected_index >= endpoint.streams.size() ||
        endpoint.streams[expected_index].id != stream.id) {
      endpoint.failed = true;
      return false;
    }
  }
  stream.input_done = true;
  ++endpoint.completed;
  return true;
}

bool raw_consume_readable(RawEndpoint &endpoint, bool &progress) noexcept {
  while (endpoint.readable_head != endpoint.readable.size()) {
    RawStream &stream = *endpoint.readable[endpoint.readable_head++];
    stream.queued_for_read = false;
    const std::string &expected_body =
        endpoint.server ? endpoint.workload->request_body
                        : endpoint.workload->response_body;
    while (stream.inbound_start != stream.inbound.size()) {
      const std::size_t count =
          (std::min)(kBodyReadBytes,
                     stream.inbound.size() - stream.inbound_start);
      if (stream.received + count > expected_body.size() ||
          std::memcmp(stream.inbound.data() + stream.inbound_start,
                      expected_body.data() + stream.received, count) != 0 ||
          nghttp2_session_consume_stream(endpoint.session, stream.id, count) !=
              0 ||
          nghttp2_session_consume_connection(endpoint.session, count) != 0) {
        endpoint.failed = true;
        return false;
      }
      stream.inbound_start += count;
      stream.received += count;
      progress = true;
    }
    stream.inbound.clear();
    stream.inbound_start = 0;
    if (stream.input_end_received && !stream.input_done &&
        !raw_finish_input(endpoint, stream))
      return false;
  }
  endpoint.readable.clear();
  endpoint.readable_head = 0;
  return true;
}

int raw_send_data(nghttp2_session *, nghttp2_frame *, const std::uint8_t *,
                  std::size_t, nghttp2_data_source *, void *) noexcept;

bool raw_initialize(RawEndpoint &endpoint) noexcept {
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
                                                          raw_begin_headers);
  nghttp2_session_callbacks_set_on_header_callback(callbacks, raw_header);
  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(callbacks,
                                                            raw_data_chunk);
  nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks,
                                                       raw_frame_recv);
  nghttp2_session_callbacks_set_send_data_callback(callbacks, raw_send_data);
  nghttp2_option_set_no_auto_window_update(option, 1);
  nghttp2_option_set_peer_max_concurrent_streams(option, 100);
  nghttp2_option_set_max_send_header_block_length(option, 64 * 1024);
  nghttp2_option_set_max_settings(option, 32);
  nghttp2_option_set_max_continuations(option, 16);
  endpoint.ngmem = nghttp2_mem{&endpoint, raw_h2_malloc, raw_h2_free,
                              raw_h2_calloc, raw_h2_realloc};
  const int created = endpoint.server
                          ? nghttp2_session_server_new3(&endpoint.session,
                                callbacks, &endpoint, option, &endpoint.ngmem)
                          : nghttp2_session_client_new3(&endpoint.session,
                                callbacks, &endpoint, option, &endpoint.ngmem);
  nghttp2_option_del(option);
  nghttp2_session_callbacks_del(callbacks);
  if (created != 0)
    return false;
  nghttp2_settings_entry settings[4] = {
      {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100},
      {NGHTTP2_SETTINGS_MAX_HEADER_LIST_SIZE, 64 * 1024},
      {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, 64 * 1024},
      {NGHTTP2_SETTINGS_ENABLE_PUSH, 0},
  };
  const std::size_t setting_count = endpoint.server ? 3 : 4;
  return nghttp2_submit_settings(endpoint.session, NGHTTP2_FLAG_NONE,
                                 settings, setting_count) == 0 &&
         nghttp2_session_set_local_window_size(endpoint.session,
                                               NGHTTP2_FLAG_NONE, 0,
                                               static_cast<std::int32_t>(
                                                   endpoint.workload
                                                       ->connection_window)) ==
             0;
}

struct RawWire {
  RawWire() noexcept = default;
  RawWire(const RawWire &) = delete;
  RawWire &operator=(const RawWire &) = delete;
  ~RawWire() { reset_buffers(); }

  bool initialize_buffers(AllocatorRef selected) noexcept {
    if (!selected.valid() || pending.data() != nullptr || input.data() != nullptr)
      return false;
    allocator = selected;
    auto *pending_data = static_cast<std::uint8_t *>(
        allocator.allocate(kWireBytes, alignof(std::max_align_t)));
    if (pending_data == nullptr)
      return false;
    pending = {pending_data, kWireBytes};
    auto *input_data = static_cast<std::uint8_t *>(
        allocator.allocate(kWireBytes, alignof(std::max_align_t)));
    if (input_data == nullptr) {
      reset_buffers();
      return false;
    }
    input = {input_data, kWireBytes};
    return true;
  }

  void reset_buffers() noexcept {
    if (input.data() != nullptr)
      allocator.deallocate(input.data(), kWireBytes, alignof(std::max_align_t));
    if (pending.data() != nullptr)
      allocator.deallocate(pending.data(), kWireBytes, alignof(std::max_align_t));
    input = {};
    pending = {};
    begin = end = 0;
  }

  int descriptor = -1;
  AllocatorRef allocator;
  std::span<std::uint8_t> pending;
  std::size_t begin = 0;
  std::size_t end = 0;
  std::span<std::uint8_t> input;
};

int raw_send_data(nghttp2_session *, nghttp2_frame *frame,
                  const std::uint8_t *frame_header, std::size_t length,
                  nghttp2_data_source *source, void *user_data) noexcept {
  auto *endpoint = static_cast<RawEndpoint *>(user_data);
  auto *stream = source == nullptr ? nullptr
                                   : static_cast<RawStream *>(source->ptr);
  if (endpoint == nullptr)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  RawWire *wire = endpoint->encoding_wire;
  const std::string &body = endpoint->server ? endpoint->workload->response_body
                                             : endpoint->workload->request_body;
  const auto fail = [&]() noexcept {
    endpoint->failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  };
  if (wire == nullptr || stream == nullptr || frame == nullptr ||
      frame_header == nullptr || frame->hd.type != NGHTTP2_DATA ||
      frame->hd.stream_id != stream->id || stream->sent > body.size() ||
      length > stream->sent || wire->begin != wire->end ||
      wire->pending.size() < 9 ||
      frame->hd.length > wire->pending.size() - 9)
    return fail();
  const std::size_t padding = frame->data.padlen;
  if (padding > frame->hd.length || length != frame->hd.length - padding)
    return fail();

  // NO_COPY skips nghttp2's DATA bounce buffer. The owned wire still retains
  // one complete frame until every partial socket write has finished.
  std::memcpy(wire->pending.data(), frame_header, 9);
  std::size_t offset = 9;
  if (padding != 0)
    wire->pending[offset++] = static_cast<std::uint8_t>(padding - 1);
  if (length != 0) {
    std::memcpy(wire->pending.data() + offset,
                body.data() + stream->sent - length, length);
    offset += length;
  }
  if (padding > 1)
    std::memset(wire->pending.data() + offset, 0, padding - 1);
  wire->end = 9 + frame->hd.length;
  return NGHTTP2_ERR_PAUSE;
}

bool raw_fill_output(RawEndpoint &endpoint, RawWire &wire) noexcept {
  if (wire.begin != wire.end)
    return true;
  if (endpoint.encoding_wire != nullptr)
    return false;
  wire.begin = wire.end = 0;
  const std::uint8_t *data = nullptr;
  endpoint.encoding_wire = &wire;
  const nghttp2_ssize length =
      nghttp2_session_mem_send2(endpoint.session, &data);
  endpoint.encoding_wire = nullptr;
  if (endpoint.failed || length < 0 ||
      static_cast<std::size_t>(length) > wire.pending.size())
    return false;
  if (length != 0) {
    if (wire.end != 0 || data == nullptr)
      return false;
    std::memcpy(wire.pending.data(), data, static_cast<std::size_t>(length));
    wire.end = static_cast<std::size_t>(length);
  }
  // A paused NO_COPY callback returns zero while leaving its complete frame
  // in wire; an ordinary zero return leaves the wire empty.
  return true;
}

bool raw_write(RawWire &wire, bool &progress,
                bool single_attempt = false) noexcept {
  // Match TcpStream's immediate socket seam, including per-call nonblocking
  // and SIGPIPE suppression where supported; plain write uses another libc
  // entry and relies on the process-wide signal setting.
  int flags = MSG_DONTWAIT;
#if defined(MSG_NOSIGNAL)
  flags |= MSG_NOSIGNAL;
#endif
  while (wire.begin != wire.end) {
    const ssize_t written = ::send(wire.descriptor,
                                    wire.pending.data() + wire.begin,
                                    wire.end - wire.begin, flags);
    if (written > 0) {
      wire.begin += static_cast<std::size_t>(written);
      progress = true;
      if (single_attempt) {
        if (wire.begin == wire.end)
          wire.begin = wire.end = 0;
        return true;
      }
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      return true;
    return false;
  }
  wire.begin = wire.end = 0;
  return true;
}

bool raw_read(RawEndpoint &endpoint, RawWire &wire, bool &progress) noexcept {
  for (;;) {
    const ssize_t count =
        ::recv(wire.descriptor, wire.input.data(), wire.input.size(),
               MSG_DONTWAIT);
    if (count > 0) {
      progress = true;
      const nghttp2_ssize consumed = nghttp2_session_mem_recv2(
          endpoint.session, wire.input.data(), static_cast<std::size_t>(count));
      return consumed == count;
    }
    if (count == 0)
      return false;
    if (errno == EINTR)
      continue;
    return errno == EAGAIN || errno == EWOULDBLOCK;
  }
}

// Match the public HYLE fixed-body convenience exactly: one request body is
// across the transport before the second request starts; both streams remain
// open until the server has accepted the pair. The server then writes the
// second complete response before starting the first, making completion order
// observable without granting either route a larger hidden outbound queue.
bool raw_advance_client(RawEndpoint &client,
                         const RawWire &client_wire) noexcept {
  if (client.streams.empty()) {
    if (!raw_submit_request(client))
      return false;
  } else if (client.streams.size() <
             static_cast<std::size_t>(client.workload->streams)) {
    const std::size_t count = client.streams.size();
    if ((count & 1u) != 0) {
      const RawStream &first = client.streams.back();
      if (first.output_done && client_wire.begin == client_wire.end &&
          !raw_submit_request(client))
        return false;
    } else if (client.completed == static_cast<int>(count) &&
               !raw_submit_request(client)) {
      return false;
    }
  }
  return true;
}

bool raw_advance_server(RawEndpoint &server,
                         const RawWire &server_wire) noexcept {
  // Submitted pairs need no more response scheduling work. Do not rescan
  // completed streams on every transport turn as the connection grows.
  for (std::size_t base = server.next_response_pair;
       base + 1 < server.streams.size(); base += 2) {
    RawStream &first = server.streams[base];
    RawStream &second = server.streams[base + 1];
    if (!first.input_done || !second.input_done)
      continue;
    if (!second.response_submitted) {
      if (!raw_submit_response(server, second))
        return false;
    } else if (second.output_done &&
               server_wire.begin == server_wire.end &&
               !first.response_submitted) {
      if (!raw_submit_response(server, first))
        return false;
      server.next_response_pair = base + 2;
    }
  }
  return true;
}

bool raw_advance(RawEndpoint &client, RawEndpoint &server,
                 const RawWire &client_wire,
                 const RawWire &server_wire) noexcept {
  return raw_advance_client(client, client_wire) &&
         raw_advance_server(server, server_wire);
}

int run_raw(const Workload &workload) noexcept {
  SocketPair pair = make_socket_pair();
  if (!pair.valid())
    return -1;
  RawEndpoint client;
  client.workload = &workload;
  RawEndpoint server;
  server.workload = &workload;
  server.server = true;
  try {
    client.streams.reserve(static_cast<std::size_t>(workload.streams));
    server.streams.reserve(static_cast<std::size_t>(workload.streams));
  } catch (...) {
    return -1;
  }
  RawWire client_wire;
  RawWire server_wire;
  client_wire.descriptor = pair.release_client();
  server_wire.descriptor = pair.release_server();
  const auto cleanup = [&] {
    if (client.session != nullptr)
      nghttp2_session_del(client.session);
    if (server.session != nullptr)
      nghttp2_session_del(server.session);
    close_descriptor(client_wire.descriptor);
    close_descriptor(server_wire.descriptor);
  };
  if (!client_wire.initialize_buffers(system_allocator()) ||
      !server_wire.initialize_buffers(system_allocator()) ||
      !raw_initialize(client) || !raw_initialize(server)) {
    cleanup();
    return -1;
  }

  while (client.completed != workload.streams && !client.failed &&
         !server.failed) {
    bool progress = false;
    if (!raw_advance(client, server, client_wire, server_wire) ||
        !raw_fill_output(client, client_wire) ||
        !raw_fill_output(server, server_wire) ||
        !raw_write(client_wire, progress) ||
         !raw_write(server_wire, progress) ||
         !raw_read(client, client_wire, progress) ||
         !raw_read(server, server_wire, progress) ||
         !raw_consume_readable(client, progress) ||
         !raw_consume_readable(server, progress)) {
      client.failed = true;
      break;
    }
    if (progress)
      continue;
    pollfd descriptors[2] = {
        {client_wire.descriptor,
         static_cast<short>(POLLIN |
                            (client_wire.begin != client_wire.end
                                 ? POLLOUT
                                 : 0)),
         0},
        {server_wire.descriptor,
         static_cast<short>(POLLIN |
                            (server_wire.begin != server_wire.end
                                 ? POLLOUT
                                 : 0)),
         0},
    };
    int ready;
    do {
      ready = ::poll(descriptors, 2, 5000);
    } while (ready < 0 && errno == EINTR);
    if (ready <= 0) {
      client.failed = true;
      break;
    }
  }
  const bool ok = !client.failed && !server.failed &&
                  client.completed == workload.streams &&
                  server.completed == workload.streams;
  cleanup();
  return ok ? workload.streams : -1;
}

struct RawTaskEpisode {
  explicit RawTaskEpisode(const Workload &selected) noexcept
      : workload(selected) {}
  const Workload &workload;
  std::array<RawEndpoint, 2> endpoint;
  std::array<RawWire, 2> wire;
  hyle::sched::Event client_consumed;
  bool failed = false;
};

bool raw_task_server_flushed(const RawEndpoint &endpoint,
                              const RawWire &wire) noexcept {
  const auto count = static_cast<std::size_t>(endpoint.workload->streams);
  return count >= 2 && endpoint.streams.size() == count &&
         endpoint.next_response_pair == count &&
         endpoint.streams[count - 2].output_done && wire.begin == wire.end;
}

struct RawTaskCleanup {
  RawTaskEpisode &episode;
  unsigned role;
  TcpStream &socket;
  bool success = false;

  ~RawTaskCleanup() {
    auto &endpoint = episode.endpoint[role];
    if (!success) {
      endpoint.failed = true;
      episode.failed = true;
      episode.client_consumed.signal();
    }
    if (endpoint.session != nullptr) {
      nghttp2_session_del(endpoint.session);
      endpoint.session = nullptr;
    }
    // Session callbacks refer to streams. Retire the session first, then free
    // every endpoint-owned container and socket before its join completes.
    decltype(endpoint.readable){}.swap(endpoint.readable);
    decltype(endpoint.streams){}.swap(endpoint.streams);
    endpoint.readable_head = 0;
    episode.wire[role].reset_buffers();
    socket = TcpStream{};
  }
};

bool raw_task_feed_async(RawEndpoint &endpoint, RawWire &wire,
                          hyle::io::TransferResult result) noexcept {
  if (!result.ok() || result.eof || result.bytes == 0 ||
      result.bytes > wire.input.size())
    return false;
  if (nghttp2_session_mem_recv2(endpoint.session, wire.input.data(),
                                result.bytes) !=
      static_cast<nghttp2_ssize>(result.bytes))
    return false;
  bool consumed = false;
  return raw_consume_readable(endpoint, consumed);
}

Task<int> raw_task_endpoint(AllocatorRef allocator, TcpStream socket,
                             RawTaskEpisode *episode, unsigned role) noexcept {
  auto &endpoint = episode->endpoint[role];
  auto &wire = episode->wire[role];
  RawTaskCleanup cleanup{*episode, role, socket};
  if (!wire.initialize_buffers(allocator))
    co_return -1;
  try {
    endpoint.streams.reserve(static_cast<std::size_t>(episode->workload.streams));
  } catch (...) {
    co_return -1;
  }
  if (!raw_initialize(endpoint))
    co_return -1;
  for (;;) {
    if (episode->failed || endpoint.failed)
      co_return -1;
    if (role == 0 && endpoint.completed == episode->workload.streams) {
      episode->client_consumed.signal();
      cleanup.success = true;
      co_return episode->workload.streams;
    }
    if (role == 1 && raw_task_server_flushed(endpoint, wire)) {
      const bool done =
          episode->client_consumed.wait_for(std::chrono::seconds(5));
      if (!done || episode->failed ||
          endpoint.completed != episode->workload.streams)
        co_return -1;
      cleanup.success = true;
      co_return episode->workload.streams;
    }
    if (!(role == 0 ? raw_advance_client(endpoint, wire)
                   : raw_advance_server(endpoint, wire)) ||
        !raw_fill_output(endpoint, wire))
      co_return -1;
    bool written = false;
    if (!raw_write(wire, written))
      co_return -1;
    if (role == 1 && raw_task_server_flushed(endpoint, wire))
      continue;
    // As in the validated two-Task driver, reenter encoding after a complete
    // flush rather than making an extra empty read before the next frame.
    if (written && wire.begin == wire.end)
      continue;
    bool progress = false;
    if (!raw_read(endpoint, wire, progress) ||
        !raw_consume_readable(endpoint, progress))
      co_return -1;
    if (progress)
      continue;

    hyle::sched::Fiber::yield();
    if (episode->failed)
      co_return -1;
    progress = false;
    if (!raw_read(endpoint, wire, progress) ||
        !raw_consume_readable(endpoint, progress))
      co_return -1;
    if (progress)
      continue;
    if (wire.begin != wire.end) {
      written = false;
      if (!raw_write(wire, written, true))
        co_return -1;
      if (written)
        continue;
      const auto result = co_await socket.write_some(
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(wire.pending.data() +
                                                  wire.begin),
              wire.end - wire.begin),
          hyle::io::OperationOptions{});
      if (!result.ok() || result.bytes == 0 ||
          result.bytes > wire.end - wire.begin)
        co_return -1;
      wire.begin += result.bytes;
      if (wire.begin == wire.end)
        wire.begin = wire.end = 0;
    } else {
      const auto result = co_await socket.read_some(
          std::span<std::byte>(
              reinterpret_cast<std::byte *>(wire.input.data()),
              wire.input.size()),
          hyle::io::OperationOptions{});
      if (!raw_task_feed_async(endpoint, wire, result))
        co_return -1;
    }
  }
}

// Sharing a Scheduler and endpoint Tasks aligns execution, not all API work:
// HYLE also maintains generic owner/cancellation/deadline, borrow/ACK and
// flush-epoch contracts. This raw peer implements only this fixed trace.
int run_task_peer(HyleHarness &harness, const Workload &workload,
                    bool check_descriptors = false) {
  SocketPair pair = make_socket_pair();
  if (!pair.valid())
    return -1;
  auto client = TcpStream::adopt(harness.io(), kOwner, pair.release_client());
  auto server = TcpStream::adopt(harness.io(), kOwner, pair.release_server());
  if (!client || !server)
    return -1;
  // Metadata lives on the caller stack. Each owner Task allocates and frees
  // two uninitialized 64KiB buffers inside the complete timed episode.
  RawTaskEpisode episode{workload};
  for (unsigned role = 0; role != 2; ++role) {
    episode.endpoint[role].workload = &workload;
    episode.endpoint[role].server = role == 1;
  }
  episode.wire[0].descriptor = static_cast<int>(client->native_handle());
  episode.wire[1].descriptor = static_cast<int>(server->native_handle());
  auto server_join = hyle::sched::spawn_task(
      harness.scheduler(), system_allocator(),
      raw_task_endpoint(system_allocator(), std::move(*server), &episode, 1));
  auto client_join = hyle::sched::spawn_task(
      harness.scheduler(), system_allocator(),
      raw_task_endpoint(system_allocator(), std::move(*client), &episode, 0));
  auto client_result = client_join.join();
  auto server_result = server_join.join();
  // Extra descriptor observations are requested only by untimed warmup.
  if (check_descriptors) {
    for (const auto &wire : episode.wire) {
      errno = 0;
      if (::fcntl(wire.descriptor, F_GETFD) != -1 || errno != EBADF)
        return -1;
    }
  }
  return !episode.failed && client_result && server_result &&
                 *client_result == workload.streams &&
                 *server_result == workload.streams
             ? workload.streams
             : -1;
}

class RawHarness {
public:
  RawHarness() : worker_([this] { worker_loop(); }) {}
  ~RawHarness() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    worker_.join();
  }
  RawHarness(const RawHarness &) = delete;
  RawHarness &operator=(const RawHarness &) = delete;

  int run(const Workload &workload) {
    std::unique_lock lock(mutex_);
    workload_ = &workload;
    pending_ = true;
    complete_ = false;
    ready_.notify_one();
    done_.wait(lock, [this] { return complete_; });
    return result_;
  }

private:
  void worker_loop() noexcept {
    for (;;) {
      const Workload *workload = nullptr;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [this] { return pending_ || stopping_; });
        if (stopping_)
          return;
        workload = workload_;
        pending_ = false;
      }
      const int result = run_raw(*workload);
      {
        std::lock_guard lock(mutex_);
        result_ = result;
        complete_ = true;
      }
      done_.notify_one();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable done_;
  const Workload *workload_ = nullptr;
  int result_ = -1;
  bool pending_ = false;
  bool complete_ = false;
  bool stopping_ = false;
  // The worker can run immediately, so initialize every field it reads first.
  std::thread worker_;
};


// The four cells in a block have the same mean position in time for both arms.
// Keep complete blocks as the unit of analysis; episode iterations are not
// independent samples. --paired-self measures this comparison chain with Hyle
// on both sides rather than assuming the timer's resolution is its accuracy.
int run_paired_comparison(bool self) {
  constexpr unsigned kBlocks = 12;
  constexpr std::uint64_t kEpisodes = 64;
  constexpr unsigned kWarmups = 16;
  const Workload workload = make_workload(
      "HTTP/2 paired complete episode 4x256KiB", 4, 256 * 1024,
      256 * 1024, NGHTTP2_INITIAL_CONNECTION_WINDOW_SIZE);
  HyleHarness hyle;

  // Read back socket capacities once outside the timed episodes. The common
  // socket factory checks every setsockopt; Linux may round/double the request.
  SocketPair configuration = make_socket_pair();
  if (!configuration.valid())
    return 2;
  const int descriptors[] = {configuration.release_client(),
                             configuration.release_server()};
  int send_capacity[2]{}, receive_capacity[2]{};
  bool socket_ok = true;
  for (unsigned index = 0; index != 2; ++index) {
    socklen_t length = sizeof(int);
    socket_ok = socket_ok &&
        ::getsockopt(descriptors[index], SOL_SOCKET, SO_SNDBUF,
                     &send_capacity[index], &length) == 0 &&
        length == sizeof(int);
    length = sizeof(int);
    socket_ok = socket_ok &&
        ::getsockopt(descriptors[index], SOL_SOCKET, SO_RCVBUF,
                     &receive_capacity[index], &length) == 0 &&
        length == sizeof(int);
    close_descriptor(descriptors[index]);
  }
  if (!socket_ok)
    return 2;

  for (unsigned round = 0; round != kWarmups; ++round) {
    const auto warm_hyle = [&] { return hyle.run(workload); };
    const auto warm_reference = [&] {
      return self ? hyle.run(workload)
                  : run_task_peer(hyle, workload, round == 0);
    };
    const int first = round % 2 == 0 ? warm_hyle() : warm_reference();
    const int second = round % 2 == 0 ? warm_reference() : warm_hyle();
    if (first != workload.streams || second != workload.streams) {
      std::fprintf(stderr, "paired warmup failed round=%u\n", round);
      return 2;
    }
  }
  // Observe the smallest nonzero delta of the same steady clock used for
  // timing. This is a clock quantum observation, not a calibrated error bound.
  using Clock = ankerl::nanobench::Clock;
  auto clock_delta = Clock::duration::max();
  for (unsigned sample = 0; sample != 20; ++sample) {
    const auto begin = Clock::now();
    auto end = Clock::now();
    while (end == begin)
      end = Clock::now();
    clock_delta = (std::min)(clock_delta, end - begin);
  }
  const double clock_ns =
      std::chrono::duration<double, std::nano>(clock_delta).count();
  std::printf("{\"type\":\"metadata\",\"mode\":\"%s\","
              "\"blocks\":%u,\"episodes_per_cell\":%llu,"
              "\"warmups_per_arm\":%u,\"streams_per_episode\":4,"
              "\"application_bytes_per_episode\":2097152,"
              "\"socket_requested_bytes\":4096,"
              "\"socket_send_bytes\":[%d,%d],"
              "\"socket_receive_bytes\":[%d,%d],"
              "\"clock_min_delta_ns\":%.17g,\"caller_cpu\":%d}\n",
              self ? "self" : "paired", kBlocks,
              static_cast<unsigned long long>(kEpisodes), kWarmups,
              send_capacity[0], send_capacity[1],
              receive_capacity[0], receive_capacity[1], clock_ns,
              ::sched_getcpu());
  std::fflush(stdout);

  struct Cell {
    bool hyle_arm = false;
    bool valid = false;
    double seconds_per_episode = 0;
    double iterations = 0;
    long voluntary_switches = 0;
    long involuntary_switches = 0;
    long minor_faults = 0;
    long major_faults = 0;
    int cpu_before = -1;
    int cpu_after = -1;
  };
  for (unsigned block = 0; block != kBlocks; ++block) {
    std::array<Cell, 4> cells{};
    bool block_valid = true;
    for (unsigned position = 0; position != cells.size(); ++position) {
      Cell &cell = cells[position];
      // Alternate ABBA and BAAB, with A = Hyle and B = reference.
      cell.hyle_arm = ((position == 0 || position == 3) == (block % 2 == 0));
      cell.valid = true;
      auto run_hyle = [&] {
        if (cell.valid && hyle.run(workload) != workload.streams)
          cell.valid = false;
      };
      auto run_reference = [&] {
        if (cell.valid &&
            run_task_peer(hyle, workload) != workload.streams)
          cell.valid = false;
      };
      ankerl::nanobench::Bench bench;
      bench.output(nullptr)
          .performanceCounters(false)
          .batch(4)
          .epochs(1)
          .epochIterations(kEpisodes);
      struct rusage before{}, after{};
      cell.cpu_before = ::sched_getcpu();
      const bool before_ok = ::getrusage(RUSAGE_SELF, &before) == 0;
      if (cell.hyle_arm || self)
        bench.run("hyle", run_hyle);
      else
        bench.run("reference", run_reference);
      const bool after_ok = ::getrusage(RUSAGE_SELF, &after) == 0;
      cell.cpu_after = ::sched_getcpu();
      const auto &result = bench.results().back();
      using Measure = ankerl::nanobench::Result::Measure;
      cell.seconds_per_episode = result.get(0, Measure::elapsed);
      cell.iterations = result.get(0, Measure::iterations);
      cell.valid = cell.valid && before_ok && after_ok &&
                   result.size() == 1 &&
                   cell.iterations == static_cast<double>(kEpisodes) &&
                   cell.seconds_per_episode > 0;
      cell.voluntary_switches = after.ru_nvcsw - before.ru_nvcsw;
      cell.involuntary_switches = after.ru_nivcsw - before.ru_nivcsw;
      cell.minor_faults = after.ru_minflt - before.ru_minflt;
      cell.major_faults = after.ru_majflt - before.ru_majflt;
      block_valid = block_valid && cell.valid;
      if (!cell.valid)
        break;
    }
    // Write only outside the timed cells. Keep every completed block, even if
    // a later block fails; never silently discard a slow or preempted sample.
    for (unsigned position = 0; position != cells.size(); ++position) {
      const Cell &cell = cells[position];
      if (cell.iterations == 0)
        break;
      std::printf("{\"type\":\"cell\",\"block\":%u,\"position\":%u,"
                  "\"arm\":\"%s\",\"valid\":%s,"
                  "\"iterations\":%.0f,\"seconds_per_episode\":%.17g,"
                  "\"voluntary_switches\":%ld,\"involuntary_switches\":%ld,"
                  "\"minor_faults\":%ld,\"major_faults\":%ld,"
                  "\"cpu_before\":%d,\"cpu_after\":%d}\n",
                  block, position, cell.hyle_arm ? "hyle" : "reference",
                  cell.valid ? "true" : "false", cell.iterations,
                  cell.seconds_per_episode, cell.voluntary_switches,
                  cell.involuntary_switches, cell.minor_faults,
                  cell.major_faults, cell.cpu_before, cell.cpu_after);
    }
    std::fflush(stdout);
    if (!block_valid) {
      std::fprintf(stderr, "paired block failed block=%u\n", block);
      return 2;
    }
  }
  return std::ferror(stdout) ? 2 : 0;
}

void run_workload(const Workload &workload, HyleHarness &hyle,
                  RawHarness &peer, bool peer_first) {
  auto run_hyle = [&] {
    if (hyle.run(workload) != workload.streams)
      std::abort();
  };
  auto run_peer = [&] {
    if (peer.run(workload) != workload.streams)
      std::abort();
  };
  if (peer_first) {
    run_peer();
    run_hyle();
  } else {
    run_hyle();
    run_peer();
  }
  ankerl::nanobench::Bench bench;
  bench.title(workload.title)
      .relative(true)
      .batch(static_cast<std::uint64_t>(workload.streams))
      .unit("stream")
      .minEpochIterations(5)
      .epochs(9);
  if (peer_first) {
    bench.run("nghttp2 direct HTTP/2 session", run_peer);
    bench.run("HYLE HTTP/2 public session", run_hyle);
  } else {
    bench.run("HYLE HTTP/2 public session", run_hyle);
    bench.run("nghttp2 direct HTTP/2 session", run_peer);
  }
  if (const char *prefix = std::getenv("HYLE_BENCH_JSON")) {
    std::ofstream output(std::string(prefix) + "-" +
                         std::to_string(workload.streams) + ".json");
    if (!output)
      std::abort();
    bench.render(ankerl::nanobench::templates::json(), output);
    if (!output)
      std::abort();
  }
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc == 2 && std::string_view(argv[1]) == "--paired")
    return run_paired_comparison(false);
  if (argc == 2 && std::string_view(argv[1]) == "--paired-self")
    return run_paired_comparison(true);
  if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--paired | --paired-self]\n", argv[0]);
    return 2;
  }
  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  const char *run_nonce = std::getenv("HYLE_PERF_RUN_NONCE");
  std::fprintf(stderr,
               "# HYLE_IO_HTTP2_RUN execution_order=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               run_nonce != nullptr ? run_nonce : "unmanaged");
  const Workload normal = make_workload(
      "HTTP/2 multiplexed inference streams 64x1KiB", 64, 1024, 1024);
  const Workload holdout = make_workload(
      "HTTP/2 flow-control holdout 4x256KiB", 4, 256 * 1024,
      256 * 1024, NGHTTP2_INITIAL_CONNECTION_WINDOW_SIZE);
  HyleHarness hyle;
  RawHarness peer;
  if (std::getenv("HYLE_BENCH_SMOKE") != nullptr) {
    const int hyle_normal = hyle.run(normal);
    const int peer_normal = peer.run(normal);
    const int hyle_holdout = hyle.run(holdout);
    const int peer_holdout = peer.run(holdout);
    std::fprintf(stderr,
                 "# HYLE_IO_HTTP2_SMOKE hyle_normal=%d peer_normal=%d "
                 "hyle_holdout=%d peer_holdout=%d\n",
                 hyle_normal, peer_normal, hyle_holdout, peer_holdout);
    return hyle_normal == normal.streams && peer_normal == normal.streams &&
                   hyle_holdout == holdout.streams &&
                   peer_holdout == holdout.streams
               ? 0
               : 2;
  }
  run_workload(normal, hyle, peer, peer_first);
  run_workload(holdout, hyle, peer, peer_first);
  return 0;
}
