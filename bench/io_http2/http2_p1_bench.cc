// Same-contract HTTP/2 P1 evidence for multiplexed inference streams.
//
// Both routes own one fresh nonblocking Unix socket pair and execute client and
// server HTTP/2 state on one persistent worker. They submit two concurrent
// POST streams at a time, retain and validate every request byte, complete the
// responses in reverse stream order, incrementally consume every response
// byte, update both flow-control windows, and retire all connection state. The
// HYLE route uses only public HYLE APIs. The peer uses the pinned public
// nghttp2 session API directly; it does not replace HPACK or frame machinery.

#include <nanobench.h>
#include <nghttp2/nghttp2.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/http2.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

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
    (void)::setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF,
                       &kSocketBufferBytes, sizeof(kSocketBufferBytes));
    (void)::setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF,
                       &kSocketBufferBytes, sizeof(kSocketBufferBytes));
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

Task<Status> read_hyle_body(AllocatorRef,
                            hyle::io::Http2BodyReader &body,
                            std::string_view expected) noexcept {
  std::size_t received = 0;
  for (;;) {
    auto next = co_await body.read(16 * 1024);
    if (!next.has_value())
      co_return std::move(next).error();
    if (received + next->bytes.size() > expected.size() ||
        std::memcmp(next->bytes.data(), expected.data() + received,
                    next->bytes.size()) != 0)
      co_return Status::DataLoss("HTTP/2 benchmark body mismatch");
    received += next->bytes.size();
    if (next->end_stream) {
      if (received != expected.size())
        co_return Status::DataLoss("HTTP/2 benchmark body length mismatch");
      co_return Status::Ok();
    }
  }
}

Task<int> hyle_server(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload) noexcept {
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
      Status read = co_await read_hyle_body(allocator, request.body,
                                            workload->request_body);
      if (!read.ok())
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
  co_return workload->streams;
}

Task<int> hyle_client(AllocatorRef allocator, TcpStream stream,
                      const Workload *workload) noexcept {
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
      Status read = co_await read_hyle_body(allocator, response.body,
                                            workload->response_body);
      if (!read.ok())
        co_return -1;
    }
    if (completed[0] != submitted[1] || completed[1] != submitted[0])
      co_return -1;
  }
  co_return workload->streams;
}

class HyleHarness {
public:
  HyleHarness() : scheduler_(1, system_allocator(), 1024), io_(scheduler_) {
    if (!io_.valid())
      std::abort();
  }
  ~HyleHarness() { scheduler_.shutdown(); }

  int run(const Workload &workload) {
    SocketPair pair = make_socket_pair();
    if (!pair.valid())
      return -1;
    const int client_descriptor = pair.release_client();
    const int server_descriptor = pair.release_server();
    hyle::io::detail::IoContextAccess::bind_handle(
        io_, kOwner, client_descriptor);
    hyle::io::detail::IoContextAccess::bind_handle(
        io_, kOwner, server_descriptor);
    TcpStream client(io_, client_descriptor, kOwner);
    TcpStream server(io_, server_descriptor, kOwner);

    auto server_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_server(system_allocator(), std::move(server), &workload));
    auto client_join = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_client(system_allocator(), std::move(client), &workload));
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

struct RawStream {
  std::int32_t id = 0;
  std::size_t sent = 0;
  std::size_t received = 0;
  bool method_ok = false;
  bool path_ok = false;
  bool status_ok = false;
  bool content_length_ok = false;
  bool input_done = false;
  bool response_submitted = false;
  bool output_done = false;
};

struct RawEndpoint {
  const Workload *workload = nullptr;
  nghttp2_session *session = nullptr;
  std::vector<RawStream> streams;
  bool server = false;
  bool failed = false;
  int completed = 0;

  RawStream *find(std::int32_t id) noexcept {
    for (auto &stream : streams) {
      if (stream.id == id)
        return &stream;
    }
    return nullptr;
  }
};

nghttp2_nv raw_nv(const char *name, const char *value) noexcept {
  return {reinterpret_cast<std::uint8_t *>(const_cast<char *>(name)),
          reinterpret_cast<std::uint8_t *>(const_cast<char *>(value)),
          std::strlen(name), std::strlen(value), NGHTTP2_NV_FLAG_NONE};
}

nghttp2_ssize raw_provider(nghttp2_session *, std::int32_t,
                           std::uint8_t *buffer, std::size_t length,
                           std::uint32_t *flags,
                           nghttp2_data_source *source, void *user_data) {
  auto *endpoint = static_cast<RawEndpoint *>(user_data);
  auto *stream = static_cast<RawStream *>(source->ptr);
  if (endpoint == nullptr || stream == nullptr)
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  const std::string &body = endpoint->server ? endpoint->workload->response_body
                                             : endpoint->workload->request_body;
  const std::size_t remaining = body.size() - stream->sent;
  const std::size_t count = (std::min)(remaining, length);
  if (count != 0)
    std::memcpy(buffer, body.data() + stream->sent, count);
  stream->sent += count;
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

bool bytes_equal(const std::uint8_t *value, std::size_t length,
                 std::string_view expected) noexcept {
  return length == expected.size() &&
         std::memcmp(value, expected.data(), length) == 0;
}

int raw_header(nghttp2_session *, const nghttp2_frame *frame,
               const std::uint8_t *name, std::size_t name_length,
               const std::uint8_t *value, std::size_t value_length,
               std::uint8_t, void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  RawStream *stream = endpoint.find(frame->hd.stream_id);
  if (stream == nullptr) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (bytes_equal(name, name_length, ":method"))
    stream->method_ok = bytes_equal(value, value_length, "POST");
  else if (bytes_equal(name, name_length, ":path"))
    stream->path_ok = bytes_equal(value, value_length, "/v1/completions");
  else if (bytes_equal(name, name_length, ":status"))
    stream->status_ok = bytes_equal(value, value_length, "200");
  else if (bytes_equal(name, name_length, "content-length")) {
    char expected[32]{};
    std::snprintf(expected, sizeof(expected), "%zu",
                  endpoint.server ? endpoint.workload->request_bytes
                                  : endpoint.workload->response_bytes);
    stream->content_length_ok =
        bytes_equal(value, value_length, expected);
  }
  return 0;
}

int raw_data_chunk(nghttp2_session *session, std::uint8_t,
                   std::int32_t stream_id, const std::uint8_t *data,
                   std::size_t length, void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  RawStream *stream = endpoint.find(stream_id);
  const std::string &expected_body =
      endpoint.server ? endpoint.workload->request_body
                      : endpoint.workload->response_body;
  if (stream == nullptr || stream->received + length > expected_body.size() ||
      std::memcmp(data, expected_body.data() + stream->received, length) != 0) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  stream->received += length;
  if (nghttp2_session_consume_stream(session, stream_id, length) != 0 ||
      nghttp2_session_consume_connection(session, length) != 0) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  return 0;
}

int raw_frame_recv(nghttp2_session *, const nghttp2_frame *frame,
                   void *user_data) {
  auto &endpoint = *static_cast<RawEndpoint *>(user_data);
  if ((frame->hd.type != NGHTTP2_HEADERS &&
       frame->hd.type != NGHTTP2_DATA) ||
      (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) == 0)
    return 0;
  RawStream *stream = endpoint.find(frame->hd.stream_id);
  if (stream == nullptr)
    return 0;
  const std::size_t expected = endpoint.server
                                   ? endpoint.workload->request_bytes
                                   : endpoint.workload->response_bytes;
  if (stream->received != expected ||
      !stream->content_length_ok ||
      (endpoint.server && (!stream->method_ok || !stream->path_ok)) ||
      (!endpoint.server && !stream->status_ok)) {
    endpoint.failed = true;
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (!endpoint.server) {
    const std::size_t pair_base =
        static_cast<std::size_t>(endpoint.completed / 2) * 2;
    const std::size_t expected_index =
        pair_base + (endpoint.completed % 2 == 0 ? 1u : 0u);
    if (expected_index >= endpoint.streams.size() ||
        endpoint.streams[expected_index].id != frame->hd.stream_id) {
      endpoint.failed = true;
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  stream->input_done = true;
  ++endpoint.completed;
  return 0;
}

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
  nghttp2_option_set_no_auto_window_update(option, 1);
  nghttp2_option_set_peer_max_concurrent_streams(option, 100);
  nghttp2_option_set_max_send_header_block_length(option, 64 * 1024);
  nghttp2_option_set_max_settings(option, 32);
  nghttp2_option_set_max_continuations(option, 16);
  const int created = endpoint.server
                          ? nghttp2_session_server_new2(
                                &endpoint.session, callbacks, &endpoint, option)
                          : nghttp2_session_client_new2(
                                &endpoint.session, callbacks, &endpoint, option);
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
  int descriptor = -1;
  std::array<std::uint8_t, kWireBytes> pending{};
  std::size_t begin = 0;
  std::size_t end = 0;
  std::array<std::uint8_t, kWireBytes> input{};
};

bool raw_fill_output(RawEndpoint &endpoint, RawWire &wire) noexcept {
  if (wire.begin != wire.end)
    return true;
  wire.begin = wire.end = 0;
  const std::uint8_t *data = nullptr;
  const nghttp2_ssize length =
      nghttp2_session_mem_send2(endpoint.session, &data);
  if (length < 0 || static_cast<std::size_t>(length) > wire.pending.size())
    return false;
  if (length != 0) {
    std::memcpy(wire.pending.data(), data, static_cast<std::size_t>(length));
    wire.end = static_cast<std::size_t>(length);
  }
  return true;
}

bool raw_write(RawWire &wire, bool &progress) noexcept {
  while (wire.begin != wire.end) {
    const ssize_t written = ::write(wire.descriptor,
                                    wire.pending.data() + wire.begin,
                                    wire.end - wire.begin);
    if (written > 0) {
      wire.begin += static_cast<std::size_t>(written);
      progress = true;
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
        ::read(wire.descriptor, wire.input.data(), wire.input.size());
    if (count > 0) {
      progress = true;
      const nghttp2_ssize consumed = nghttp2_session_mem_recv2(
          endpoint.session, wire.input.data(), static_cast<std::size_t>(count));
      if (consumed != count)
        return false;
      continue;
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
bool raw_advance(RawEndpoint &client, RawEndpoint &server,
                 const RawWire &client_wire,
                 const RawWire &server_wire) noexcept {
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

  for (std::size_t base = 0; base + 1 < server.streams.size(); base += 2) {
    RawStream &first = server.streams[base];
    RawStream &second = server.streams[base + 1];
    if (!first.input_done || !second.input_done)
      continue;
    if (!second.response_submitted) {
      if (!raw_submit_response(server, second))
        return false;
    } else if (second.output_done &&
               server_wire.begin == server_wire.end &&
               !first.response_submitted &&
               !raw_submit_response(server, first)) {
      return false;
    }
  }
  return true;
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
  client.streams.reserve(static_cast<std::size_t>(workload.streams));
  server.streams.reserve(static_cast<std::size_t>(workload.streams));
  RawWire client_wire{.descriptor = pair.release_client()};
  RawWire server_wire{.descriptor = pair.release_server()};
  const auto cleanup = [&] {
    if (client.session != nullptr)
      nghttp2_session_del(client.session);
    if (server.session != nullptr)
      nghttp2_session_del(server.session);
    close_descriptor(client_wire.descriptor);
    close_descriptor(server_wire.descriptor);
  };
  if (!raw_initialize(client) || !raw_initialize(server)) {
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
        !raw_read(server, server_wire, progress)) {
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
  std::thread worker_;
  const Workload *workload_ = nullptr;
  int result_ = -1;
  bool pending_ = false;
  bool complete_ = false;
  bool stopping_ = false;
};

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
}

} // namespace

int main() {
  std::signal(SIGPIPE, SIG_IGN);
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
    return hyle.run(normal) == normal.streams &&
                   peer.run(normal) == normal.streams &&
                   hyle.run(holdout) == holdout.streams &&
                   peer.run(holdout) == holdout.streams
               ? 0
               : 2;
  }
  run_workload(normal, hyle, peer, peer_first);
  run_workload(holdout, hyle, peer, peer_first);
  return 0;
}
