#include "hyle/io/http.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "picohttpparser.h"

namespace hyle::io {
namespace {

constexpr std::uint32_t kInitialBufferBytes = 4096;

template <class T> Result<T> http_failure(Status status) noexcept {
  return make_unexpected(std::move(status));
}

bool ascii_iequal(StringView left, StringView right) noexcept {
  if (left.size() != right.size())
    return false;
  for (std::uint32_t i = 0; i < left.size(); ++i) {
    unsigned char a = static_cast<unsigned char>(left[i]);
    unsigned char b = static_cast<unsigned char>(right[i]);
    if (a >= 'A' && a <= 'Z')
      a = static_cast<unsigned char>(a + ('a' - 'A'));
    if (b >= 'A' && b <= 'Z')
      b = static_cast<unsigned char>(b + ('a' - 'A'));
    if (a != b)
      return false;
  }
  return true;
}

StringView trim_ows(StringView value) noexcept {
  std::uint32_t first = 0;
  std::uint32_t last = value.size();
  while (first != last && (value[first] == ' ' || value[first] == '\t')) {
    ++first;
  }
  while (last != first && (value[last - 1] == ' ' || value[last - 1] == '\t')) {
    --last;
  }
  return value.substr(first, last - first);
}

bool header_has_token(HttpHeadersView headers, StringView name,
                      StringView wanted) noexcept {
  for (const HttpHeader &header : headers) {
    if (!ascii_iequal(header.name, name))
      continue;
    StringView value = header.value;
    std::uint32_t begin = 0;
    while (begin <= value.size()) {
      std::uint32_t comma = value.find(',', begin);
      if (comma == StringView::npos)
        comma = value.size();
      if (ascii_iequal(trim_ows(value.substr(begin, comma - begin)), wanted)) {
        return true;
      }
      if (comma == value.size())
        break;
      begin = comma + 1;
    }
  }
  return false;
}

bool contains_crlf(StringView value) noexcept {
  for (char byte : value) {
    if (byte == '\r' || byte == '\n')
      return true;
  }
  return false;
}

bool valid_header_name(StringView name) noexcept {
  if (name.empty())
    return false;
  static constexpr char kSeparators[] = "()<>@,;:\\\"/[]?={} \t";
  for (char byte : name) {
    const unsigned char value = static_cast<unsigned char>(byte);
    if (value <= 0x20 || value >= 0x7f ||
        std::strchr(kSeparators, value) != nullptr) {
      return false;
    }
  }
  return true;
}

bool valid_header_value(StringView value) noexcept {
  for (char byte : value) {
    const unsigned char ch = static_cast<unsigned char>(byte);
    if (ch == '\r' || ch == '\n' || ch == 0x7f || (ch < 0x20 && ch != '\t')) {
      return false;
    }
  }
  return true;
}

bool valid_request_target(StringView target) noexcept {
  if (target.empty() || contains_crlf(target))
    return false;
  for (char byte : target) {
    const unsigned char ch = static_cast<unsigned char>(byte);
    if (ch <= 0x20 || ch == 0x7f)
      return false;
  }
  return true;
}

bool parse_u64(StringView text, std::uint64_t &output) noexcept {
  if (text.empty())
    return false;
  std::uint64_t value = 0;
  for (char byte : text) {
    if (byte < '0' || byte > '9')
      return false;
    const std::uint64_t digit = static_cast<std::uint64_t>(byte - '0');
    if (value > ((std::numeric_limits<std::uint64_t>::max)() - digit) / 10) {
      return false;
    }
    value = value * 10 + digit;
  }
  output = value;
  return true;
}

struct GrowBuffer {
  explicit GrowBuffer(mem::AllocatorRef source) noexcept : allocator(source) {}
  ~GrowBuffer() {
    if (data != nullptr) {
      allocator.deallocate(data, capacity, alignof(std::max_align_t));
    }
  }
  GrowBuffer(const GrowBuffer &) = delete;
  GrowBuffer &operator=(const GrowBuffer &) = delete;

  bool reserve(std::size_t requested) noexcept {
    if (requested <= capacity)
      return true;
    if (requested > (std::numeric_limits<std::uint32_t>::max)())
      return false;
    std::size_t next = capacity == 0 ? kInitialBufferBytes : capacity;
    while (next < requested) {
      if (next > (std::numeric_limits<std::uint32_t>::max)() / 2) {
        next = requested;
        break;
      }
      next *= 2;
    }
    void *replacement =
        allocator.reallocate(data, capacity, next, alignof(std::max_align_t));
    if (replacement == nullptr)
      return false;
    data = static_cast<char *>(replacement);
    capacity = static_cast<std::uint32_t>(next);
    return true;
  }

  bool append(const void *source, std::size_t count) noexcept {
    if (count == 0)
      return true;
    if (count > (std::numeric_limits<std::uint32_t>::max)() - size ||
        !reserve(static_cast<std::size_t>(size) + count)) {
      return false;
    }
    std::memcpy(data + size, source, count);
    size += static_cast<std::uint32_t>(count);
    return true;
  }

  bool append(StringView source) noexcept {
    return append(source.data(), source.size());
  }

  void consume(std::size_t count) noexcept {
    if (count >= size) {
      size = 0;
      return;
    }
    std::memmove(data, data + count, size - count);
    size -= static_cast<std::uint32_t>(count);
  }

  void clear() noexcept { size = 0; }

  void release_if_empty_above(std::size_t retained_bytes) noexcept {
    if (size == 0 && capacity > retained_bytes)
      release();
  }

  void release() noexcept {
    if (data != nullptr) {
      allocator.deallocate(data, capacity, alignof(std::max_align_t));
    }
    data = nullptr;
    size = 0;
    capacity = 0;
  }

  mem::AllocatorRef allocator;
  char *data = nullptr;
  std::uint32_t size = 0;
  std::uint32_t capacity = 0;
};

bool put(GrowBuffer &output, const char *bytes, std::size_t size) noexcept {
  return output.append(bytes, size);
}

bool put(GrowBuffer &output, StringView value) noexcept {
  return output.append(value);
}

bool put_cstr(GrowBuffer &output, const char *value) noexcept {
  return put(output, value, std::strlen(value));
}

bool put_u64(GrowBuffer &output, std::uint64_t value) noexcept {
  char digits[20];
  std::size_t begin = sizeof(digits);
  do {
    digits[--begin] = static_cast<char>('0' + (value % 10));
    value /= 10;
  } while (value != 0);
  return put(output, digits + begin, sizeof(digits) - begin);
}

std::size_t format_hex(char *destination, std::size_t capacity,
                       std::uint64_t value) noexcept {
  static constexpr char kHex[] = "0123456789abcdef";
  char reverse[16];
  std::size_t count = 0;
  do {
    reverse[count++] = kHex[value & 0xfu];
    value >>= 4;
  } while (value != 0);
  if (count > capacity)
    return 0;
  for (std::size_t i = 0; i < count; ++i) {
    destination[i] = reverse[count - i - 1];
  }
  return count;
}

const char *reason_phrase(int status) noexcept {
  switch (status) {
  case 200:
    return "OK";
  case 201:
    return "Created";
  case 202:
    return "Accepted";
  case 204:
    return "No Content";
  case 206:
    return "Partial Content";
  case 300:
    return "Multiple Choices";
  case 301:
    return "Moved Permanently";
  case 302:
    return "Found";
  case 304:
    return "Not Modified";
  case 307:
    return "Temporary Redirect";
  case 308:
    return "Permanent Redirect";
  case 400:
    return "Bad Request";
  case 401:
    return "Unauthorized";
  case 403:
    return "Forbidden";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 408:
    return "Request Timeout";
  case 409:
    return "Conflict";
  case 411:
    return "Length Required";
  case 413:
    return "Payload Too Large";
  case 417:
    return "Expectation Failed";
  case 429:
    return "Too Many Requests";
  case 500:
    return "Internal Server Error";
  case 501:
    return "Not Implemented";
  case 502:
    return "Bad Gateway";
  case 503:
    return "Service Unavailable";
  case 504:
    return "Gateway Timeout";
  default:
    return "";
  }
}

const char *version_text(HttpVersion version) noexcept {
  return version == HttpVersion::kHttp10 ? "HTTP/1.0" : "HTTP/1.1";
}

Status validate_limits(mem::AllocatorRef allocator,
                       const HttpTransport &transport,
                       const HttpLimits &limits) noexcept {
  if (!allocator.valid()) {
    return Status::InvalidArgument("HTTP requires a valid allocator");
  }
  if (!transport.valid()) {
    return Status::InvalidArgument("HTTP requires a connected transport");
  }
  if (limits.max_head_bytes < 4 ||
      limits.max_head_bytes > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::InvalidArgument(
        "HTTP max_head_bytes must fit the parser buffer");
  }
  if (limits.max_header_count == 0 ||
      limits.max_header_count > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::InvalidArgument(
        "HTTP max_header_count must be nonzero and bounded");
  }
  if (limits.read_chunk_bytes == 0 ||
      limits.read_chunk_bytes > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::InvalidArgument(
        "HTTP read_chunk_bytes must be nonzero and bounded");
  }
  if (limits.max_retained_body_bytes >
      (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::InvalidArgument(
        "HTTP max_retained_body_bytes must fit the body buffer");
  }
  if (limits.max_retained_head_bytes >
      (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::InvalidArgument(
        "HTTP max_retained_head_bytes must fit the parser buffer");
  }
  if (limits.expect_continue_wait < time::Duration::zero()) {
    return Status::InvalidArgument(
        "HTTP expect_continue_wait must be nonnegative");
  }
  return Status{};
}

struct HeaderStorage {
  HeaderStorage(mem::AllocatorRef source, std::size_t maximum,
                std::size_t retained_head_bytes) noexcept
      : allocator(source), head(source), capacity(maximum),
        max_retained_head_bytes(retained_head_bytes) {
    raw = allocator.allocate_n<phr_header>(capacity);
    headers = allocator.allocate_n<HttpHeader>(capacity);
    if (raw == nullptr || headers == nullptr)
      reset_allocations();
  }
  ~HeaderStorage() { reset_allocations(); }
  HeaderStorage(const HeaderStorage &) = delete;
  HeaderStorage &operator=(const HeaderStorage &) = delete;

  bool valid() const noexcept {
    return capacity != 0 && raw != nullptr && headers != nullptr;
  }

  void reset_message() noexcept {
    head.clear();
    head.release_if_empty_above(max_retained_head_bytes);
    count = 0;
  }

  Status capture(const char *source_base, std::size_t head_bytes,
                 std::size_t parsed_count) noexcept {
    if (head_bytes > (std::numeric_limits<std::uint32_t>::max)() ||
        parsed_count > capacity) {
      return Status::ResourceExhausted("HTTP header limits exceeded");
    }
    head.clear();
    if (!head.append(source_base, head_bytes)) {
      return Status::ResourceExhausted(
          "HTTP could not allocate stable header storage");
    }
    for (std::size_t i = 0; i < parsed_count; ++i) {
      if (raw[i].name == nullptr) {
        return Status::InvalidArgument(
            "obsolete folded HTTP header is not accepted");
      }
      const std::ptrdiff_t name_offset = raw[i].name - source_base;
      const std::ptrdiff_t value_offset = raw[i].value - source_base;
      if (name_offset < 0 || value_offset < 0 ||
          static_cast<std::size_t>(name_offset) > head_bytes ||
          static_cast<std::size_t>(value_offset) > head_bytes ||
          raw[i].name_len >
              head_bytes - static_cast<std::size_t>(name_offset) ||
          raw[i].value_len >
              head_bytes - static_cast<std::size_t>(value_offset)) {
        return Status::Internal("HTTP parser returned an invalid header view");
      }
      raw[i].name = head.data + name_offset;
      raw[i].value = head.data + value_offset;
      headers[i] = HttpHeader{
          StringView(raw[i].name, static_cast<std::uint32_t>(raw[i].name_len)),
          StringView(raw[i].value,
                     static_cast<std::uint32_t>(raw[i].value_len))};
    }
    count = parsed_count;
    return Status{};
  }

  HttpHeadersView view() const noexcept {
    return HttpHeadersView(std::span<const HttpHeader>(headers, count));
  }

private:
  void reset_allocations() noexcept {
    if (raw != nullptr)
      allocator.deallocate_n(raw, capacity);
    if (headers != nullptr)
      allocator.deallocate_n(headers, capacity);
    raw = nullptr;
    headers = nullptr;
  }

public:
  mem::AllocatorRef allocator;
  GrowBuffer head;
  phr_header *raw = nullptr;
  HttpHeader *headers = nullptr;
  std::size_t capacity = 0;
  std::size_t count = 0;
  std::size_t max_retained_head_bytes = 0;
};

enum class BodyFraming : std::uint8_t {
  kNone,
  kContentLength,
  kChunked,
  kCloseDelimited,
};

enum class WriteFraming : std::uint8_t {
  kFixed,
  kChunked,
  kCloseDelimited,
  kSuppressed,
};

enum class ConnectionKind : std::uint8_t {
  kServer,
  kClient,
};

struct Framing {
  BodyFraming kind = BodyFraming::kNone;
  std::uint64_t length = 0;
};

Status framing_protocol_error(bool request, const char *message) noexcept {
  return request ? Status::InvalidArgument(message) : Status::DataLoss(message);
}

Result<Framing> inspect_framing(HttpHeadersView headers,
                                std::uint64_t body_limit,
                                bool request) noexcept {
  std::optional<StringView> transfer_encoding;
  std::optional<StringView> content_length;
  for (const HttpHeader &header : headers) {
    if (ascii_iequal(header.name, "transfer-encoding")) {
      if (transfer_encoding.has_value()) {
        return http_failure<Framing>(framing_protocol_error(
            request, "multiple Transfer-Encoding fields are ambiguous"));
      }
      transfer_encoding = trim_ows(header.value);
      continue;
    }
    if (ascii_iequal(header.name, "content-length")) {
      const StringView value = trim_ows(header.value);
      if (content_length.has_value() && !ascii_iequal(*content_length, value)) {
        return http_failure<Framing>(framing_protocol_error(
            request, "conflicting Content-Length fields are ambiguous"));
      }
      content_length = value;
    }
  }

  if (transfer_encoding.has_value()) {
    if (content_length.has_value()) {
      return http_failure<Framing>(framing_protocol_error(
          request, "Transfer-Encoding and Content-Length cannot coexist"));
    }
    if (!ascii_iequal(*transfer_encoding, "chunked")) {
      return http_failure<Framing>(framing_protocol_error(
          request, "HTTP supports only one terminal chunked transfer coding"));
    }
    return Framing{BodyFraming::kChunked, 0};
  }

  if (content_length.has_value()) {
    std::uint64_t length = 0;
    if (!parse_u64(*content_length, length)) {
      return http_failure<Framing>(framing_protocol_error(
          request, "Content-Length is not a valid decimal"));
    }
    if (length > body_limit) {
      return http_failure<Framing>(
          Status::ResourceExhausted("HTTP body exceeds max_body_bytes"));
    }
    return Framing{BodyFraming::kContentLength, length};
  }

  return Framing{request ? BodyFraming::kNone : BodyFraming::kCloseDelimited,
                 0};
}

Status validate_headers(std::span<const HttpHeader> headers,
                        bool request) noexcept {
  for (const HttpHeader &header : headers) {
    if (!valid_header_name(header.name) || !valid_header_value(header.value)) {
      return Status::InvalidArgument("HTTP header contains invalid bytes");
    }
    if (ascii_iequal(header.name, "content-length") ||
        ascii_iequal(header.name, "transfer-encoding") ||
        ascii_iequal(header.name, "connection") ||
        (request && (ascii_iequal(header.name, "host") ||
                     ascii_iequal(header.name, "expect")))) {
      return Status::InvalidArgument(
          "HTTP framing header is owned by the message head");
    }
  }
  return Status{};
}

Status validate_trailers(std::span<const HttpHeader> trailers) noexcept {
  for (const HttpHeader &trailer : trailers) {
    if (!valid_header_name(trailer.name) ||
        !valid_header_value(trailer.value)) {
      return Status::InvalidArgument("HTTP trailer contains invalid bytes");
    }
    if (ascii_iequal(trailer.name, "content-length") ||
        ascii_iequal(trailer.name, "transfer-encoding") ||
        ascii_iequal(trailer.name, "host") ||
        ascii_iequal(trailer.name, "connection") ||
        ascii_iequal(trailer.name, "trailer") ||
        ascii_iequal(trailer.name, "te")) {
      return Status::InvalidArgument(
          "HTTP trailer uses a forbidden framing field");
    }
  }
  return Status{};
}

} // namespace

std::optional<StringView>
HttpHeadersView::find(StringView name) const noexcept {
  for (const HttpHeader &header : headers_) {
    if (ascii_iequal(header.name, name))
      return header.value;
  }
  return std::nullopt;
}

namespace detail {

struct HttpBodyState {
  HttpBodyState(mem::AllocatorRef allocator, const HttpLimits &limits) noexcept
      : decoded(allocator), aggregate(allocator),
        trailers(allocator, limits.max_header_count,
                 limits.max_retained_head_bytes),
        max_retained_head_bytes(limits.max_retained_head_bytes),
        max_retained_body_bytes(limits.max_retained_body_bytes) {}

  void begin(HttpTransport *byte_transport, GrowBuffer *byte_buffer,
             BodyFraming body_framing, std::uint64_t content_length,
             const HttpLimits &limits_value,
             bool reading_request_body) noexcept {
    transport = byte_transport;
    wire = byte_buffer;
    framing = body_framing;
    remaining_bytes = content_length;
    max_body_bytes = limits_value.max_body_bytes;
    read_chunk_bytes = limits_value.read_chunk_bytes;
    max_head_bytes = limits_value.max_head_bytes;
    request_body = reading_request_body;
    total_bytes = 0;
    pending_consume = 0;
    decoded_offset = 0;
    decoded.clear();
    aggregate.clear();
    trailers.reset_message();
    std::memset(&decoder, 0, sizeof(decoder));
    decoder.consume_trailer = 0;
    ++generation;
    active = true;
    done = body_framing == BodyFraming::kNone ||
           (body_framing == BodyFraming::kContentLength && content_length == 0);
    abandoned = false;
    busy = false;
    chunk_terminal = false;
    collecting = false;
  }

  // Retiring a completed borrow is the connection boundary that invalidates
  // its last returned view. Consume those bytes before the shared wire buffer
  // is interpreted as the next pipelined message.
  void retire() noexcept {
    if (pending_consume != 0 && wire != nullptr) {
      wire->consume(pending_consume);
      pending_consume = 0;
    }
    if (wire != nullptr) {
      wire->release_if_empty_above(max_retained_head_bytes);
    }
    if (aggregate.capacity > max_retained_body_bytes)
      aggregate.release();
    active = false;
    ++generation;
  }

  HttpTransport *transport = nullptr;
  GrowBuffer *wire = nullptr;
  GrowBuffer decoded;
  GrowBuffer aggregate;
  HeaderStorage trailers;
  phr_chunked_decoder decoder{};
  BodyFraming framing = BodyFraming::kNone;
  std::uint64_t generation = 0;
  std::uint64_t remaining_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t max_body_bytes = 0;
  std::size_t read_chunk_bytes = 0;
  std::size_t max_head_bytes = 0;
  std::size_t max_retained_head_bytes = 0;
  std::size_t max_retained_body_bytes = 0;
  std::size_t pending_consume = 0;
  std::size_t decoded_offset = 0;
  bool active = false;
  bool done = true;
  bool abandoned = false;
  bool busy = false;
  bool chunk_terminal = false;
  bool collecting = false;
  bool request_body = true;
};

struct HttpWriteState {
  HttpTransport *transport = nullptr;
  GrowBuffer *scratch = nullptr;
  void *owner = nullptr;
  ConnectionKind owner_kind = ConnectionKind::kServer;
  WriteFraming framing = WriteFraming::kFixed;
  std::uint64_t generation = 0;
  std::optional<std::uint64_t> remaining_bytes{};
  std::size_t max_retained_scratch_bytes = 0;
  bool active = false;
  bool abandoned = false;
  bool busy = false;
  bool close_after = false;
  bool wrote_body = false;
};

class HttpConnectionAccess {
public:
  static HttpBodyReader reader(HttpBodyState &state,
                               mem::AllocatorRef allocator) noexcept {
    return HttpBodyReader(&state, state.generation, allocator);
  }

  static HttpBodyWriter writer(HttpWriteState &state,
                               mem::AllocatorRef allocator) noexcept {
    return HttpBodyWriter(&state, state.generation, allocator);
  }

  static sched::Task<Result<HttpBodyChunk>>
  source_next(mem::AllocatorRef allocator, HttpBodySource source,
              OperationOptions options) noexcept {
    return source.next_(allocator, source.context_, options);
  }

  static bool source_valid(HttpBodySource source) noexcept {
    return source.next_ != nullptr;
  }

  static std::optional<std::uint64_t>
  source_length(HttpBodySource source) noexcept {
    return source.content_length_;
  }

  static bool writer_suppressed(const HttpBodyWriter &writer) noexcept {
    return writer.state_ != nullptr &&
           writer.state_->generation == writer.generation_ &&
           writer.state_->framing == WriteFraming::kSuppressed;
  }

  static bool writer_matches(const HttpBodyWriter &writer,
                             const HttpWriteState &state) noexcept {
    return writer.state_ == &state && state.active && !state.abandoned &&
           writer.generation_ == state.generation;
  }

  static bool writer_has_written(const HttpBodyWriter &writer) noexcept {
    return writer.state_ != nullptr && writer.state_->wrote_body;
  }

  static void disarm_writer(HttpBodyWriter &writer,
                            HttpWriteState &state) noexcept {
    state.active = false;
    if (writer.state_ == &state && writer.generation_ == state.generation) {
      writer.state_ = nullptr;
      writer.generation_ = 0;
    }
  }

  static void writer_finished(HttpWriteState &state) noexcept;
  static void writer_abandoned(HttpWriteState &state) noexcept;
};

} // namespace detail

bool HttpTransport::valid() const noexcept {
  if (const auto *tcp = std::get_if<TcpStream>(&stream_))
    return tcp->valid();
#if defined(HYLE_IO_TLS)
  if (const auto *tls = std::get_if<TlsStream>(&stream_))
    return tls->valid();
#endif
  return false;
}

bool HttpTransport::encrypted() const noexcept {
#if defined(HYLE_IO_TLS)
  return std::holds_alternative<TlsStream>(stream_);
#else
  return false;
#endif
}

StringView HttpTransport::negotiated_protocol() const noexcept {
#if defined(HYLE_IO_TLS)
  if (const auto *tls = std::get_if<TlsStream>(&stream_))
    return tls->negotiated_protocol();
#endif
  return {};
}

unsigned HttpTransport::owner_worker() const noexcept {
  if (const auto *tcp = std::get_if<TcpStream>(&stream_)) {
    return tcp->owner_worker();
  }
#if defined(HYLE_IO_TLS)
  if (const auto *tls = std::get_if<TlsStream>(&stream_)) {
    return tls->owner_worker();
  }
#endif
  return 0;
}

namespace detail {

ImmediateTransferResult
http_transport_try_read(HttpTransport &transport,
                        std::span<std::byte> buffer) noexcept {
  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_))
    return tcp_try_read_some(*tcp, buffer);
#if defined(HYLE_IO_TLS)
  if (std::holds_alternative<TlsStream>(transport.stream_)) {
    return {{}, ImmediateTransferState::kUnsupported};
  }
#endif
  return {TransferResult{
              0, Status::FailedPrecondition("HTTP transport is invalid"), false},
          ImmediateTransferState::kReady};
}

ImmediateTransferResult
http_transport_try_write(HttpTransport &transport,
                         std::span<const std::byte> buffer) noexcept {
  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_))
    return tcp_try_write_some(*tcp, buffer);
#if defined(HYLE_IO_TLS)
  if (std::holds_alternative<TlsStream>(transport.stream_))
    return {{}, ImmediateTransferState::kUnsupported};
#endif
  return {TransferResult{
              0, Status::FailedPrecondition("HTTP transport is invalid"), false},
          ImmediateTransferState::kReady};
}

sched::Task<TransferResult>
http_transport_read(mem::AllocatorRef allocator, HttpTransport &transport,
                    std::span<std::byte> buffer,
                    OperationOptions options) noexcept {
#if defined(HYLE_IO_TLS)
  if (auto *tls = std::get_if<TlsStream>(&transport.stream_)) {
    co_return co_await tls_read(allocator, *tls, buffer, options);
  }
#else
  (void)allocator;
#endif
  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_)) {
    co_return co_await tcp->read_some(buffer, options);
  }
  co_return TransferResult{
      0, Status::FailedPrecondition("HTTP transport is invalid"), false};
}

sched::Task<TransferResult>
http_transport_write(mem::AllocatorRef allocator, HttpTransport &transport,
                     std::span<const std::byte> buffer,
                     OperationOptions options) noexcept {
#if defined(HYLE_IO_TLS)
  if (auto *tls = std::get_if<TlsStream>(&transport.stream_)) {
    co_return co_await tls_write(allocator, *tls, buffer, options);
  }
#else
  (void)allocator;
#endif
  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_)) {
    co_return co_await tcp->write_some(buffer, options);
  }
  co_return TransferResult{
      0, Status::FailedPrecondition("HTTP transport is invalid"), false};
}

namespace {

Status validate_http_buffer_sequence(std::span<const ConstBuffer> buffers,
                                     std::size_t &total) noexcept {
  if (buffers.size() > kMaxBufferSequenceSegments) {
    return Status::OutOfRange(
        "HTTP buffer sequence exceeds the portable segment limit");
  }
  total = 0;
  for (const ConstBuffer buffer : buffers) {
    if (!buffer.empty() && buffer.data() == nullptr) {
      return Status::InvalidArgument(
          "HTTP buffer sequence contains a null segment");
    }
    if (buffer.size() > kMaxBufferSequenceBytes - total) {
      return Status::OutOfRange(
          "HTTP buffer sequence exceeds the portable byte limit");
    }
    total += buffer.size();
  }
  return Status{};
}

sched::Task<BufferSequenceResult>
ready_buffer_sequence(mem::AllocatorRef allocator,
                      BufferSequenceResult result) noexcept {
  (void)allocator;
  co_return result;
}

#if defined(HYLE_IO_TLS)
sched::Task<BufferSequenceResult>
tls_write_all_buffers(mem::AllocatorRef allocator, TlsStream &tls,
                      std::span<const ConstBuffer> buffers,
                      OperationOptions options) noexcept {
  std::size_t transferred = 0;
  for (std::size_t segment = 0; segment < buffers.size(); ++segment) {
    const ConstBuffer buffer = buffers[segment];
    std::size_t offset = 0;
    while (offset != buffer.size()) {
      const auto bytes = std::span<const std::byte>(
          static_cast<const std::byte *>(buffer.data()) + offset,
          buffer.size() - offset);
      auto task = tls_write(allocator, tls, bytes, options);
      if (!task.valid()) {
        co_return BufferSequenceResult{
            {transferred, BufferSequenceCursor{segment, offset}, false},
            Status::ResourceExhausted(
                "HTTP TLS vector-write coroutine allocation failed"),
            false};
      }
      TransferResult result = co_await task;
      if (!result.status.ok()) {
        co_return BufferSequenceResult{
            {transferred, BufferSequenceCursor{segment, offset}, false},
            std::move(result.status), result.eof};
      }
      if (result.bytes == 0 || result.bytes > bytes.size()) {
        co_return BufferSequenceResult{
            {transferred, BufferSequenceCursor{segment, offset}, false},
            Status::Unavailable(
                "HTTP TLS transport made no valid vector-write progress"),
            result.eof};
      }
      offset += result.bytes;
      transferred += result.bytes;
    }
  }
  co_return BufferSequenceResult{
      {transferred, BufferSequenceCursor{buffers.size(), 0}, true}, Status{},
      false};
}
#endif

} // namespace

sched::Task<BufferSequenceResult>
http_transport_write_all(mem::AllocatorRef allocator,
                         HttpTransport &transport,
                         std::span<const ConstBuffer> buffers,
                         OperationOptions options) noexcept {
  std::size_t total = 0;
  Status valid = validate_http_buffer_sequence(buffers, total);
  if (!valid.ok())
    return ready_buffer_sequence(
        allocator, BufferSequenceResult{{}, std::move(valid), false});
  if (total == 0) {
    return ready_buffer_sequence(allocator, BufferSequenceResult{
        {0, BufferSequenceCursor{buffers.size(), 0}, true}, Status{}, false});
  }

#if defined(HYLE_IO_TLS)
  if (auto *tls = std::get_if<TlsStream>(&transport.stream_))
    return tls_write_all_buffers(allocator, *tls, buffers, options);
#endif

  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_))
    return hyle::io::write_all(allocator, *tcp, buffers, options);
  return ready_buffer_sequence(allocator, BufferSequenceResult{
      {}, Status::FailedPrecondition("HTTP transport is invalid"), false});
}

sched::Task<Status> http_transport_shutdown(mem::AllocatorRef allocator,
                                            HttpTransport &transport,
                                            OperationOptions options) noexcept {
#if defined(HYLE_IO_TLS)
  if (auto *tls = std::get_if<TlsStream>(&transport.stream_)) {
    Status status = co_await tls_shutdown(
        allocator, *tls,
        TlsShutdownOptions{TlsShutdownMode::kSendOnly, options});
    transport.stream_.template emplace<std::monostate>();
    co_return status;
  }
#else
  (void)allocator;
  (void)options;
#endif
  if (auto *tcp = std::get_if<TcpStream>(&transport.stream_)) {
    Status status = tcp->shutdown_write();
    transport.stream_.template emplace<std::monostate>();
    co_return status;
  }
  co_return Status::FailedPrecondition("HTTP transport is invalid");
}

} // namespace detail

namespace {

sched::Task<TransferResult> read_more(mem::AllocatorRef allocator,
                                      GrowBuffer &buffer,
                                      HttpTransport &transport,
                                      std::size_t maximum_read,
                                      OperationOptions options) noexcept {
  if (maximum_read == 0 ||
      maximum_read >
          (std::numeric_limits<std::uint32_t>::max)() - buffer.size ||
      !buffer.reserve(static_cast<std::size_t>(buffer.size) + maximum_read)) {
    co_return TransferResult{
        0, Status::ResourceExhausted("HTTP input buffer allocation failed"),
        false};
  }
  const std::size_t writable =
      std::min<std::size_t>(maximum_read, buffer.capacity - buffer.size);
  auto task = transport.read_some(
      allocator,
      std::span<std::byte>(
          reinterpret_cast<std::byte *>(buffer.data + buffer.size), writable),
      options);
  if (!task.valid()) {
    co_return TransferResult{
        0,
        Status::ResourceExhausted(
            "HTTP transport read coroutine allocation failed"),
        false};
  }
  TransferResult result = co_await task;
  if (result.bytes > writable) {
    co_return TransferResult{
        0, Status::Internal("HTTP transport over-reported read progress"),
        false};
  }
  buffer.size += static_cast<std::uint32_t>(result.bytes);
  co_return result;
}

sched::Task<Status> write_all(mem::AllocatorRef allocator,
                              HttpTransport &transport,
                              std::span<const std::byte> bytes,
                              OperationOptions options) noexcept {
  std::size_t offset = 0;
  while (offset != bytes.size()) {
    auto task = transport.write_some(allocator, bytes.subspan(offset), options);
    if (!task.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP transport write coroutine allocation failed");
    }
    TransferResult result = co_await task;
    if (!result.status.ok())
      co_return std::move(result.status);
    if (result.bytes == 0 || result.bytes > bytes.size() - offset) {
      co_return Status::Unavailable(
          "HTTP transport made no valid write progress");
    }
    offset += result.bytes;
  }
  co_return Status{};
}

Status buffer_sequence_status(BufferSequenceResult result) noexcept {
  if (!result.status.ok())
    return std::move(result.status);
  if (!result.progress.complete) {
    return Status::Unavailable(
        "HTTP transport did not complete the buffer sequence");
  }
  return Status{};
}

sched::Task<Status>
write_all_sequential(mem::AllocatorRef allocator, HttpTransport &transport,
                     std::span<const ConstBuffer> buffers,
                     OperationOptions options) noexcept {
  for (const ConstBuffer buffer : buffers) {
    if (buffer.empty())
      continue;
    auto task = write_all(
        allocator, transport,
        std::span<const std::byte>(
            static_cast<const std::byte *>(buffer.data()), buffer.size()),
        options);
    if (!task.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP sequential-write coroutine allocation failed");
    }
    Status status = co_await task;
    if (!status.ok())
      co_return status;
  }
  co_return Status{};
}

sched::Task<Result<HttpHeadersView>>
read_trailers(mem::AllocatorRef allocator, detail::HttpBodyState &state,
              OperationOptions options) noexcept {
  std::size_t last_length = 0;
  for (;;) {
    std::size_t header_count = state.trailers.capacity;
    const int parsed =
        phr_parse_headers(state.wire->data, state.wire->size,
                          state.trailers.raw, &header_count, last_length);
    if (parsed > 0) {
      Status captured = state.trailers.capture(
          state.wire->data, static_cast<std::size_t>(parsed), header_count);
      if (!captured.ok()) {
        co_return http_failure<HttpHeadersView>(std::move(captured));
      }
      Status valid = validate_trailers(state.trailers.view().entries());
      if (!valid.ok()) {
        co_return http_failure<HttpHeadersView>(
            state.request_body
                ? std::move(valid)
                : Status::DataLoss("invalid HTTP response trailer field"));
      }
      state.wire->consume(static_cast<std::size_t>(parsed));
      co_return state.trailers.view();
    }
    if (parsed == -1) {
      co_return http_failure<HttpHeadersView>(
          state.request_body
              ? Status::InvalidArgument("malformed HTTP trailer block")
              : Status::DataLoss("malformed HTTP response trailer block"));
    }
    last_length = state.wire->size;
    if (state.wire->size >= state.max_head_bytes) {
      co_return http_failure<HttpHeadersView>(
          Status::ResourceExhausted("HTTP trailers exceed max_head_bytes"));
    }
    const std::size_t amount = std::min(
        state.read_chunk_bytes, state.max_head_bytes - state.wire->size);
    auto task =
        read_more(allocator, *state.wire, *state.transport, amount, options);
    if (!task.valid()) {
      co_return http_failure<HttpHeadersView>(Status::ResourceExhausted(
          "HTTP trailer read coroutine allocation failed"));
    }
    TransferResult result = co_await task;
    if (!result.status.ok()) {
      co_return http_failure<HttpHeadersView>(std::move(result.status));
    }
    if (result.eof || result.bytes == 0) {
      co_return http_failure<HttpHeadersView>(
          Status::DataLoss("HTTP peer closed inside trailer block"));
    }
  }
}

struct BusyGuard {
  explicit BusyGuard(bool &flag) noexcept : flag_(&flag) { flag = true; }
  ~BusyGuard() { *flag_ = false; }
  BusyGuard(const BusyGuard &) = delete;
  BusyGuard &operator=(const BusyGuard &) = delete;
  bool *flag_;
};

Result<HttpBodyChunkView> body_failure(detail::HttpBodyState &state,
                                       Status status) noexcept {
  state.abandoned = true;
  return http_failure<HttpBodyChunkView>(std::move(status));
}

} // namespace

HttpBodyReader::~HttpBodyReader() { abandon_(); }

HttpBodyReader::HttpBodyReader(HttpBodyReader &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      allocator_(other.allocator_) {}

HttpBodyReader &HttpBodyReader::operator=(HttpBodyReader &&other) noexcept {
  if (this != &other) {
    abandon_();
    state_ = std::exchange(other.state_, nullptr);
    generation_ = std::exchange(other.generation_, 0);
    allocator_ = other.allocator_;
  }
  return *this;
}

bool HttpBodyReader::valid() const noexcept {
  return state_ != nullptr && state_->active &&
         state_->generation == generation_ && !state_->abandoned;
}

bool HttpBodyReader::done() const noexcept { return valid() && state_->done; }

std::optional<std::uint64_t> HttpBodyReader::remaining() const noexcept {
  if (!valid() || state_->framing != BodyFraming::kContentLength) {
    return std::nullopt;
  }
  return state_->remaining_bytes;
}

void HttpBodyReader::abandon_() noexcept {
  if (state_ != nullptr && state_->active &&
      state_->generation == generation_ && !state_->done) {
    state_->abandoned = true;
  }
  state_ = nullptr;
  generation_ = 0;
}

sched::Task<Result<HttpBodyChunkView>>
http_body_read(mem::AllocatorRef allocator, HttpBodyReader &reader,
               std::size_t requested_bytes, OperationOptions options) noexcept {
  detail::HttpBodyState *state = reader.state_;
  if (state == nullptr || !state->active ||
      state->generation != reader.generation_) {
    co_return http_failure<HttpBodyChunkView>(
        Status::FailedPrecondition("HTTP body reader is stale"));
  }
  if (state->abandoned) {
    co_return http_failure<HttpBodyChunkView>(
        Status::FailedPrecondition("HTTP body reader was abandoned"));
  }
  if (state->busy) {
    co_return http_failure<HttpBodyChunkView>(
        Status::FailedPrecondition("only one HTTP body read may be in flight"));
  }
  BusyGuard guard(state->busy);

  if (state->pending_consume != 0) {
    state->wire->consume(state->pending_consume);
    state->pending_consume = 0;
  }
  if (!state->collecting)
    state->aggregate.clear();
  if (state->done) {
    co_return HttpBodyChunkView{{}, state->trailers.view(), true};
  }

  std::size_t maximum =
      requested_bytes == 0 ? state->read_chunk_bytes : requested_bytes;
  maximum = std::min(maximum, state->read_chunk_bytes);
  if (maximum == 0) {
    co_return body_failure(
        *state, Status::InvalidArgument("HTTP body read size must be nonzero"));
  }

  if (state->framing == BodyFraming::kContentLength) {
    while (state->wire->size == 0) {
      const std::size_t amount =
          static_cast<std::size_t>(std::min<std::uint64_t>(
              state->read_chunk_bytes, state->remaining_bytes));
      auto task = read_more(allocator, *state->wire, *state->transport, amount,
                            options);
      if (!task.valid()) {
        co_return body_failure(
            *state, Status::ResourceExhausted(
                        "HTTP body read coroutine allocation failed"));
      }
      TransferResult result = co_await task;
      if (!result.status.ok()) {
        co_return body_failure(*state, std::move(result.status));
      }
      if (result.eof || result.bytes == 0) {
        co_return body_failure(
            *state, Status::DataLoss(
                        "HTTP peer closed before Content-Length completed"));
      }
    }
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(
        std::min<std::size_t>(state->wire->size, maximum),
        state->remaining_bytes));
    state->remaining_bytes -= count;
    state->total_bytes += count;
    state->pending_consume = count;
    state->done = state->remaining_bytes == 0;
    co_return HttpBodyChunkView{
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(state->wire->data), count),
        {},
        state->done};
  }

  if (state->framing == BodyFraming::kCloseDelimited) {
    while (state->wire->size == 0) {
      auto task = read_more(allocator, *state->wire, *state->transport,
                            state->read_chunk_bytes, options);
      if (!task.valid()) {
        co_return body_failure(
            *state, Status::ResourceExhausted(
                        "HTTP body read coroutine allocation failed"));
      }
      TransferResult result = co_await task;
      if (!result.status.ok()) {
        co_return body_failure(*state, std::move(result.status));
      }
      if (result.eof || result.bytes == 0) {
        state->done = true;
        co_return HttpBodyChunkView{{}, {}, true};
      }
    }
    const std::size_t count = std::min<std::size_t>(state->wire->size, maximum);
    if (count > state->max_body_bytes - state->total_bytes) {
      co_return body_failure(
          *state, Status::ResourceExhausted(
                      "HTTP close-delimited body exceeds max_body_bytes"));
    }
    state->total_bytes += count;
    state->pending_consume = count;
    co_return HttpBodyChunkView{
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(state->wire->data), count),
        {},
        false};
  }

  // Chunked decoding keeps parser state across transport fragments. Decoded
  // payload is copied once into reusable connection storage because pico
  // compacts trailing pipelined bytes into the same input buffer.
  for (;;) {
    if (state->decoded_offset < state->decoded.size) {
      const std::size_t available = state->decoded.size - state->decoded_offset;
      const std::size_t count = std::min(available, maximum);
      const auto *data = reinterpret_cast<const std::byte *>(
          state->decoded.data + state->decoded_offset);
      state->decoded_offset += count;
      co_return HttpBodyChunkView{
          std::span<const std::byte>(data, count), {}, false};
    }
    state->decoded.clear();
    state->decoded_offset = 0;

    if (state->chunk_terminal) {
      auto task = read_trailers(allocator, *state, options);
      if (!task.valid()) {
        co_return body_failure(*state,
                               Status::ResourceExhausted(
                                   "HTTP trailer coroutine allocation failed"));
      }
      Result<HttpHeadersView> trailers = co_await task;
      if (!trailers) {
        co_return body_failure(*state, std::move(trailers).error());
      }
      state->done = true;
      co_return HttpBodyChunkView{{}, *trailers, true};
    }

    if (state->wire->size != 0) {
      std::size_t decoded_bytes = state->wire->size;
      const ssize_t trailing = phr_decode_chunked(
          &state->decoder, state->wire->data, &decoded_bytes);
      if (trailing == -1) {
        co_return body_failure(
            *state,
            state->request_body
                ? Status::InvalidArgument("malformed chunked HTTP body")
                : Status::DataLoss("malformed chunked HTTP response body"));
      }
      if (decoded_bytes > state->max_body_bytes - state->total_bytes) {
        co_return body_failure(*state,
                               Status::ResourceExhausted(
                                   "chunked HTTP body exceeds max_body_bytes"));
      }
      if (decoded_bytes != 0 &&
          !state->decoded.append(state->wire->data, decoded_bytes)) {
        co_return body_failure(
            *state,
            Status::ResourceExhausted("HTTP decoded chunk allocation failed"));
      }
      state->total_bytes += decoded_bytes;
      if (trailing >= 0) {
        const std::size_t tail = static_cast<std::size_t>(trailing);
        if (tail != 0) {
          std::memmove(state->wire->data, state->wire->data + decoded_bytes,
                       tail);
        }
        state->wire->size = static_cast<std::uint32_t>(tail);
        state->chunk_terminal = true;
      } else {
        state->wire->clear();
      }
      if (state->decoded.size != 0)
        continue;
      if (state->chunk_terminal)
        continue;
    }

    auto task = read_more(allocator, *state->wire, *state->transport,
                          state->read_chunk_bytes, options);
    if (!task.valid()) {
      co_return body_failure(*state,
                             Status::ResourceExhausted(
                                 "HTTP body read coroutine allocation failed"));
    }
    TransferResult result = co_await task;
    if (!result.status.ok()) {
      co_return body_failure(*state, std::move(result.status));
    }
    if (result.eof || result.bytes == 0) {
      co_return body_failure(
          *state, Status::DataLoss("HTTP peer closed inside chunked body"));
    }
  }
}

sched::Task<Result<StringView>>
http_body_read_all(mem::AllocatorRef allocator, HttpBodyReader &reader,
                   std::size_t max_bytes, OperationOptions options) noexcept {
  detail::HttpBodyState *state = reader.state_;
  if (state == nullptr || !state->active ||
      state->generation != reader.generation_ || state->abandoned) {
    co_return http_failure<StringView>(
        Status::FailedPrecondition("HTTP body reader is stale"));
  }
  if (max_bytes > (std::numeric_limits<std::uint32_t>::max)()) {
    co_return http_failure<StringView>(
        Status::InvalidArgument("HTTP read_all limit must fit a StringView"));
  }
  if (state->framing == BodyFraming::kContentLength &&
      state->remaining_bytes > max_bytes) {
    state->abandoned = true;
    co_return http_failure<StringView>(
        Status::ResourceExhausted("HTTP body exceeds read_all limit"));
  }

  state->aggregate.clear();
  state->collecting = true;
  for (;;) {
    auto task = http_body_read(allocator, reader, 0, options);
    if (!task.valid()) {
      state->collecting = false;
      state->abandoned = true;
      co_return http_failure<StringView>(Status::ResourceExhausted(
          "HTTP body convenience coroutine allocation failed"));
    }
    Result<HttpBodyChunkView> next = co_await task;
    if (!next) {
      state->collecting = false;
      co_return http_failure<StringView>(std::move(next).error());
    }
    if (next->bytes.size() > max_bytes - state->aggregate.size ||
        !state->aggregate.append(next->bytes.data(), next->bytes.size())) {
      state->collecting = false;
      state->abandoned = true;
      co_return http_failure<StringView>(
          Status::ResourceExhausted("HTTP body exceeds read_all limit"));
    }
    if (next->end_stream)
      break;
  }
  state->collecting = false;
  co_return StringView(state->aggregate.data, state->aggregate.size);
}

sched::Task<Status> http_body_discard(mem::AllocatorRef allocator,
                                      HttpBodyReader &reader,
                                      OperationOptions options) noexcept {
  for (;;) {
    auto task = http_body_read(allocator, reader, 0, options);
    if (!task.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP body discard coroutine allocation failed");
    }
    Result<HttpBodyChunkView> chunk = co_await task;
    if (!chunk)
      co_return std::move(chunk).error();
    if (chunk->end_stream)
      co_return Status{};
  }
}

struct HttpServerConnection::Impl {
  Impl(mem::AllocatorRef allocator, HttpTransport byte_transport,
       const HttpLimits &limits) noexcept
      : transport(std::move(byte_transport)), wire(allocator),
        output(allocator),
        message(allocator, limits.max_header_count,
                limits.max_retained_head_bytes),
        body(allocator, limits) {}

  bool ready() const noexcept {
    return transport.valid() && message.valid() && body.trailers.valid();
  }

  HttpTransport transport;
  GrowBuffer wire;
  GrowBuffer output;
  HeaderStorage message;
  detail::HttpBodyState body;
  detail::HttpWriteState writer;
  HttpRequestView current_request{};
  bool request_active = false;
  bool response_started = false;
  bool continue_sent = false;
  bool reusable = true;
  bool closed = false;
};

struct HttpClientConnection::Impl {
  Impl(mem::AllocatorRef allocator, HttpTransport byte_transport,
       const HttpLimits &limits) noexcept
      : transport(std::move(byte_transport)), wire(allocator),
        output(allocator),
        message(allocator, limits.max_header_count,
                limits.max_retained_head_bytes),
        body(allocator, limits) {}

  bool ready() const noexcept {
    return transport.valid() && message.valid() && body.trailers.valid();
  }

  HttpTransport transport;
  GrowBuffer wire;
  GrowBuffer output;
  HeaderStorage message;
  detail::HttpBodyState body;
  detail::HttpWriteState writer;
  HttpResponseView current_response{};
  bool request_active = false;
  bool request_finished = false;
  bool response_active = false;
  bool last_request_was_head = false;
  bool last_request_expects_continue = false;
  bool continue_decided = false;
  bool waiting_for_continue = false;
  bool reusable = true;
  bool closed = false;
};

namespace detail {

void HttpConnectionAccess::writer_finished(HttpWriteState &state) noexcept {
  if (state.owner_kind == ConnectionKind::kServer) {
    auto *owner = static_cast<HttpServerConnection::Impl *>(state.owner);
    owner->request_active = false;
    owner->response_started = false;
    if (state.close_after)
      owner->reusable = false;
  } else {
    auto *owner = static_cast<HttpClientConnection::Impl *>(state.owner);
    owner->request_finished = true;
    if (state.close_after)
      owner->reusable = false;
  }
}

void HttpConnectionAccess::writer_abandoned(HttpWriteState &state) noexcept {
  state.abandoned = true;
  state.active = false;
  if (state.owner_kind == ConnectionKind::kServer) {
    auto *owner = static_cast<HttpServerConnection::Impl *>(state.owner);
    owner->reusable = false;
    owner->response_started = false;
    owner->request_active = false;
    owner->body.active = false;
    ++owner->body.generation;
  } else {
    auto *owner = static_cast<HttpClientConnection::Impl *>(state.owner);
    owner->reusable = false;
    owner->request_active = false;
    owner->request_finished = false;
  }
}

} // namespace detail

namespace {

Status append_connection_header(GrowBuffer &output, HttpVersion version,
                                bool keep_alive) noexcept {
  if (version == HttpVersion::kHttp11 && !keep_alive) {
    if (!put_cstr(output, "Connection: close\r\n")) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  } else if (version == HttpVersion::kHttp10 && keep_alive) {
    if (!put_cstr(output, "Connection: keep-alive\r\n")) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  }
  return Status{};
}

Status append_outgoing_headers(GrowBuffer &output,
                               std::span<const HttpHeader> headers) noexcept {
  for (const HttpHeader &header : headers) {
    if (!put(output, header.name) || !put_cstr(output, ": ") ||
        !put(output, header.value) || !put_cstr(output, "\r\n")) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  }
  return Status{};
}

Status
append_framing_header(GrowBuffer &output, WriteFraming framing,
                      std::optional<std::uint64_t> content_length) noexcept {
  if (framing == WriteFraming::kFixed ||
      (framing == WriteFraming::kSuppressed && content_length.has_value())) {
    if (!put_cstr(output, "Content-Length: ") ||
        !put_u64(output, content_length.value_or(0)) ||
        !put_cstr(output, "\r\n")) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  } else if (framing == WriteFraming::kChunked) {
    if (!put_cstr(output, "Transfer-Encoding: chunked\r\n")) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  }
  return Status{};
}

Status
build_request_head(GrowBuffer &output, const HttpRequestHead &head,
                   WriteFraming framing,
                   std::optional<std::uint64_t> content_length) noexcept {
  if (!valid_header_name(head.method) || !valid_request_target(head.target) ||
      contains_crlf(head.authority)) {
    return Status::InvalidArgument(
        "HTTP request method, target, or authority is invalid");
  }
  if (head.version == HttpVersion::kHttp11 && head.authority.empty()) {
    return Status::InvalidArgument(
        "HTTP/1.1 request requires an authority for Host");
  }
  for (char byte : head.authority) {
    const unsigned char ch = static_cast<unsigned char>(byte);
    if (ch <= 0x20 || ch == 0x7f) {
      return Status::InvalidArgument("HTTP authority contains invalid bytes");
    }
  }
  Status headers = validate_headers(head.headers, true);
  if (!headers.ok())
    return headers;
  if (head.version == HttpVersion::kHttp10 &&
      framing == WriteFraming::kChunked) {
    return Status::InvalidArgument(
        "HTTP/1.0 request cannot use chunked transfer coding");
  }

  if (!put(output, head.method) || !put(output, " ", 1) ||
      !put(output, head.target) || !put(output, " ", 1) ||
      !put_cstr(output, version_text(head.version)) ||
      !put_cstr(output, "\r\n")) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  if (!head.authority.empty() &&
      (!put_cstr(output, "Host: ") || !put(output, head.authority) ||
       !put_cstr(output, "\r\n"))) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  headers = append_outgoing_headers(output, head.headers);
  if (!headers.ok())
    return headers;
  headers = append_framing_header(output, framing, content_length);
  if (!headers.ok())
    return headers;
  headers = append_connection_header(output, head.version, head.keep_alive);
  if (!headers.ok())
    return headers;
  if (head.expect_continue && !put_cstr(output, "Expect: 100-continue\r\n")) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  if (!put_cstr(output, "\r\n")) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  return Status{};
}

Status build_response_head(GrowBuffer &output, const HttpResponseHead &head,
                           WriteFraming framing,
                           std::optional<std::uint64_t> content_length,
                           HttpVersion version,
                           bool effective_keep_alive) noexcept {
  if (head.status < 200 || head.status > 999 || contains_crlf(head.reason)) {
    return Status::InvalidArgument(
        "HTTP final response status or reason is invalid");
  }
  for (char byte : head.reason) {
    const unsigned char ch = static_cast<unsigned char>(byte);
    if (ch == 0x7f || (ch < 0x20 && ch != '\t')) {
      return Status::InvalidArgument("HTTP reason contains invalid bytes");
    }
  }
  Status headers = validate_headers(head.headers, false);
  if (!headers.ok())
    return headers;
  if (version == HttpVersion::kHttp10 && framing == WriteFraming::kChunked) {
    return Status::InvalidArgument(
        "HTTP/1.0 response cannot use chunked transfer coding");
  }

  if (!put_cstr(output, version_text(version)) || !put(output, " ", 1) ||
      !put_u64(output, static_cast<std::uint64_t>(head.status)) ||
      !put(output, " ", 1)) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  if (!head.reason.empty()) {
    if (!put(output, head.reason)) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  } else {
    const char *phrase = reason_phrase(head.status);
    if (!put_cstr(output, phrase)) {
      return Status::ResourceExhausted("HTTP head allocation failed");
    }
  }
  if (!put_cstr(output, "\r\n")) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  headers = append_outgoing_headers(output, head.headers);
  if (!headers.ok())
    return headers;
  headers = append_framing_header(output, framing, content_length);
  if (!headers.ok())
    return headers;
  headers = append_connection_header(output, version, effective_keep_alive);
  if (!headers.ok())
    return headers;
  if (!put_cstr(output, "\r\n")) {
    return Status::ResourceExhausted("HTTP head allocation failed");
  }
  return Status{};
}

void begin_writer(detail::HttpWriteState &state, HttpTransport &transport,
                  GrowBuffer &scratch, std::size_t max_retained_scratch_bytes,
                  void *owner, ConnectionKind kind, WriteFraming framing,
                  std::optional<std::uint64_t> content_length,
                  bool close_after) noexcept {
  state.transport = &transport;
  state.scratch = &scratch;
  state.owner = owner;
  state.owner_kind = kind;
  state.framing = framing;
  state.remaining_bytes = content_length;
  state.max_retained_scratch_bytes = max_retained_scratch_bytes;
  ++state.generation;
  state.active = true;
  state.abandoned = false;
  state.busy = false;
  state.close_after = close_after;
  state.wrote_body = false;
}

sched::Task<Status> immediate_status(mem::AllocatorRef,
                                     Status status) noexcept {
  co_return status;
}

} // namespace

HttpBodyWriter::~HttpBodyWriter() { abandon_(); }

HttpBodyWriter::HttpBodyWriter(HttpBodyWriter &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      allocator_(other.allocator_) {}

HttpBodyWriter &HttpBodyWriter::operator=(HttpBodyWriter &&other) noexcept {
  if (this != &other) {
    abandon_();
    state_ = std::exchange(other.state_, nullptr);
    generation_ = std::exchange(other.generation_, 0);
    allocator_ = other.allocator_;
  }
  return *this;
}

bool HttpBodyWriter::valid() const noexcept {
  return state_ != nullptr && state_->active &&
         state_->generation == generation_ && !state_->abandoned;
}

std::optional<std::uint64_t> HttpBodyWriter::remaining() const noexcept {
  if (!valid())
    return std::nullopt;
  return state_->remaining_bytes;
}

void HttpBodyWriter::abandon_() noexcept {
  if (state_ != nullptr && state_->active &&
      state_->generation == generation_) {
    detail::HttpConnectionAccess::writer_abandoned(*state_);
  }
  state_ = nullptr;
  generation_ = 0;
}

sched::Task<Status> http_body_write(mem::AllocatorRef allocator,
                                    HttpBodyWriter &writer,
                                    std::span<const std::byte> bytes,
                                    OperationOptions options) noexcept {
  detail::HttpWriteState *state = writer.state_;
  if (state == nullptr || !state->active ||
      state->generation != writer.generation_) {
    co_return Status::FailedPrecondition("HTTP body writer is stale");
  }
  if (state->abandoned) {
    co_return Status::FailedPrecondition("HTTP body writer was abandoned");
  }
  if (state->busy) {
    co_return Status::FailedPrecondition(
        "only one HTTP body write may be in flight");
  }
  BusyGuard guard(state->busy);
  if (bytes.empty())
    co_return Status{};

  if (state->remaining_bytes.has_value() &&
      bytes.size() > *state->remaining_bytes) {
    detail::HttpConnectionAccess::writer_abandoned(*state);
    writer.state_ = nullptr;
    co_return Status::OutOfRange("HTTP body exceeds declared Content-Length");
  }

  if (state->framing == WriteFraming::kSuppressed) {
    if (state->remaining_bytes.has_value()) {
      *state->remaining_bytes -= bytes.size();
    }
    co_return Status{};
  }

  if (state->framing == WriteFraming::kChunked) {
    char prefix[20];
    const std::size_t digits =
        format_hex(prefix, sizeof(prefix) - 2, bytes.size());
    prefix[digits] = '\r';
    prefix[digits + 1] = '\n';
    static constexpr std::byte kCrLf[] = {std::byte{'\r'}, std::byte{'\n'}};
    Status status;
    const std::size_t framing_bytes = digits + 2 + sizeof(kCrLf);
    const bool stage_encrypted_chunk =
        state->transport->encrypted() &&
        state->max_retained_scratch_bytes >= framing_bytes &&
        bytes.size() <= state->max_retained_scratch_bytes - framing_bytes;
    if (stage_encrypted_chunk) {
      if (state->scratch == nullptr) {
        detail::HttpConnectionAccess::writer_abandoned(*state);
        writer.state_ = nullptr;
        co_return Status::Internal("HTTP writer has no output scratch buffer");
      }
      GrowBuffer &wire = *state->scratch;
      wire.clear();
      if (!wire.append(prefix, digits + 2) ||
          !wire.append(bytes.data(), bytes.size()) ||
          !wire.append(kCrLf, sizeof(kCrLf))) {
        wire.clear();
        wire.release_if_empty_above(state->max_retained_scratch_bytes);
        detail::HttpConnectionAccess::writer_abandoned(*state);
        writer.state_ = nullptr;
        co_return Status::ResourceExhausted(
            "HTTP encrypted chunk staging allocation failed");
      }
      auto task = write_all(
          allocator, *state->transport,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(wire.data), wire.size),
          options);
      if (!task.valid()) {
        wire.clear();
        wire.release_if_empty_above(state->max_retained_scratch_bytes);
        detail::HttpConnectionAccess::writer_abandoned(*state);
        writer.state_ = nullptr;
        co_return Status::ResourceExhausted(
            "HTTP encrypted chunk coroutine allocation failed");
      }
      status = co_await task;
      wire.clear();
      wire.release_if_empty_above(state->max_retained_scratch_bytes);
    } else if (state->transport->encrypted()) {
      const ConstBuffer sequence[] = {
          ConstBuffer(prefix, digits + 2), ConstBuffer(bytes),
          ConstBuffer(kCrLf, sizeof(kCrLf))};
      auto task = write_all_sequential(allocator, *state->transport, sequence,
                                       options);
      if (!task.valid()) {
        detail::HttpConnectionAccess::writer_abandoned(*state);
        writer.state_ = nullptr;
        co_return Status::ResourceExhausted(
            "HTTP encrypted chunk coroutine allocation failed");
      }
      status = co_await task;
    } else {
      const ConstBuffer sequence[] = {
          ConstBuffer(prefix, digits + 2), ConstBuffer(bytes),
          ConstBuffer(kCrLf, sizeof(kCrLf))};
      auto task = detail::http_transport_write_all(
          allocator, *state->transport, sequence, options);
      if (!task.valid()) {
        detail::HttpConnectionAccess::writer_abandoned(*state);
        writer.state_ = nullptr;
        co_return Status::ResourceExhausted(
            "HTTP chunk vector-write coroutine allocation failed");
      }
      status = buffer_sequence_status(co_await task);
    }
    if (!status.ok()) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return status;
    }
  } else {
    auto task = write_all(allocator, *state->transport, bytes, options);
    if (!task.valid()) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return Status::ResourceExhausted(
          "HTTP body coroutine allocation failed");
    }
    Status status = co_await task;
    if (!status.ok()) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return status;
    }
  }

  if (state->remaining_bytes.has_value()) {
    *state->remaining_bytes -= bytes.size();
  }
  state->wrote_body = true;
  co_return Status{};
}

sched::Task<Status> http_body_finish(mem::AllocatorRef allocator,
                                     HttpBodyWriter &writer,
                                     std::span<const HttpHeader> trailers,
                                     OperationOptions options) noexcept {
  detail::HttpWriteState *state = writer.state_;
  if (state == nullptr || !state->active ||
      state->generation != writer.generation_) {
    co_return Status::FailedPrecondition("HTTP body writer is stale");
  }
  if (state->abandoned) {
    co_return Status::FailedPrecondition("HTTP body writer was abandoned");
  }
  if (state->busy) {
    co_return Status::FailedPrecondition(
        "only one HTTP body write may be in flight");
  }
  BusyGuard guard(state->busy);

  if (state->framing != WriteFraming::kSuppressed &&
      state->remaining_bytes.has_value() && *state->remaining_bytes != 0) {
    detail::HttpConnectionAccess::writer_abandoned(*state);
    writer.state_ = nullptr;
    co_return Status::DataLoss(
        "HTTP body ended before declared Content-Length");
  }
  if (state->framing != WriteFraming::kChunked && !trailers.empty()) {
    detail::HttpConnectionAccess::writer_abandoned(*state);
    writer.state_ = nullptr;
    co_return Status::InvalidArgument(
        "HTTP trailers require chunked transfer coding");
  }

  if (state->framing == WriteFraming::kChunked) {
    Status valid = validate_trailers(trailers);
    if (!valid.ok()) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return valid;
    }
    if (state->scratch == nullptr) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return Status::Internal("HTTP writer has no output scratch buffer");
    }
    GrowBuffer &ending = *state->scratch;
    ending.clear();
    if (!put_cstr(ending, "0\r\n")) {
      ending.clear();
      ending.release_if_empty_above(state->max_retained_scratch_bytes);
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return Status::ResourceExhausted("HTTP trailer allocation failed");
    }
    valid = append_outgoing_headers(ending, trailers);
    if (!valid.ok() || !put_cstr(ending, "\r\n")) {
      ending.clear();
      ending.release_if_empty_above(state->max_retained_scratch_bytes);
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return valid.ok()
          ? Status::ResourceExhausted("HTTP trailer allocation failed")
          : valid;
    }
    auto task = write_all(
        allocator, *state->transport,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(ending.data), ending.size),
        options);
    if (!task.valid()) {
      ending.clear();
      ending.release_if_empty_above(state->max_retained_scratch_bytes);
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return Status::ResourceExhausted(
          "HTTP trailer coroutine allocation failed");
    }
    Status status = co_await task;
    ending.clear();
    ending.release_if_empty_above(state->max_retained_scratch_bytes);
    if (!status.ok()) {
      detail::HttpConnectionAccess::writer_abandoned(*state);
      writer.state_ = nullptr;
      co_return status;
    }
  }

  state->active = false;
  detail::HttpConnectionAccess::writer_finished(*state);
  writer.state_ = nullptr;
  writer.generation_ = 0;
  co_return Status{};
}

Result<HttpServerConnection>
HttpServerConnection::create(mem::AllocatorRef allocator,
                             HttpTransport transport,
                             HttpLimits limits) noexcept {
  Status valid = validate_limits(allocator, transport, limits);
  if (!valid.ok()) {
    return http_failure<HttpServerConnection>(std::move(valid));
  }
  Impl *impl = allocator.allocate_n<Impl>(1);
  if (impl == nullptr) {
    return http_failure<HttpServerConnection>(
        Status::ResourceExhausted("HTTP server connection allocation failed"));
  }
  new (impl) Impl(allocator, std::move(transport), limits);
  if (!impl->ready()) {
    impl->~Impl();
    allocator.deallocate_n(impl, 1);
    return http_failure<HttpServerConnection>(Status::ResourceExhausted(
        "HTTP server connection state allocation failed"));
  }
  return HttpServerConnection(allocator, limits, impl);
}

HttpServerConnection::~HttpServerConnection() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    allocator_.deallocate_n(impl_, 1);
  }
}

HttpServerConnection::HttpServerConnection(
    HttpServerConnection &&other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)), allocator_(other.allocator_),
      limits_(other.limits_) {}

HttpServerConnection &
HttpServerConnection::operator=(HttpServerConnection &&other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      allocator_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(other.impl_, nullptr);
    allocator_ = other.allocator_;
    limits_ = other.limits_;
  }
  return *this;
}

bool HttpServerConnection::valid() const noexcept {
  return impl_ != nullptr && !impl_->closed && impl_->transport.valid();
}

unsigned HttpServerConnection::owner_worker() const noexcept {
  return impl_ == nullptr ? 0 : impl_->transport.owner_worker();
}

sched::Task<Status>
HttpServerConnection::close(OperationOptions options) noexcept {
  if (impl_ == nullptr) {
    return immediate_status(
        allocator_,
        Status::FailedPrecondition("HTTP server connection is invalid"));
  }
  if (impl_->closed)
    return immediate_status(allocator_, Status{});
  impl_->closed = true;
  impl_->reusable = false;
  impl_->request_active = false;
  impl_->body.active = false;
  ++impl_->body.generation;
  if (impl_->writer.active) {
    detail::HttpConnectionAccess::writer_abandoned(impl_->writer);
  }
  return impl_->transport.shutdown(allocator_, options);
}

sched::Task<Result<std::optional<HttpIncomingRequest>>>
http_server_read_request(mem::AllocatorRef allocator,
                         HttpServerConnection &connection,
                         OperationOptions options) noexcept {
  HttpServerConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->transport.valid()) {
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::FailedPrecondition("HTTP server connection is invalid"));
  }
  if (impl->request_active || impl->response_started || impl->writer.active) {
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::FailedPrecondition(
            "finish the current HTTP request before reading another"));
  }
  if (!impl->reusable) {
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::FailedPrecondition(
            "HTTP server connection is no longer reusable"));
  }

  impl->body.active = false;
  ++impl->body.generation;
  impl->message.reset_message();
  std::size_t last_length = 0;
  const char *method = nullptr;
  const char *target = nullptr;
  std::size_t method_length = 0;
  std::size_t target_length = 0;
  int minor_version = -1;
  std::size_t header_count = 0;
  int parsed = -2;

  for (;;) {
    if (impl->wire.size != 0) {
      header_count = impl->message.capacity;
      parsed = phr_parse_request(impl->wire.data, impl->wire.size, &method,
                                 &method_length, &target, &target_length,
                                 &minor_version, impl->message.raw,
                                 &header_count, last_length);
      if (parsed > 0)
        break;
      if (parsed == -1) {
        impl->reusable = false;
        const Status status =
            header_count == impl->message.capacity
                ? Status::ResourceExhausted(
                      "HTTP request exceeds max_header_count")
                : Status::InvalidArgument("malformed HTTP request head");
        co_return http_failure<std::optional<HttpIncomingRequest>>(status);
      }
      last_length = impl->wire.size;
      if (impl->wire.size >= connection.limits_.max_head_bytes) {
        impl->reusable = false;
        co_return http_failure<std::optional<HttpIncomingRequest>>(
            Status::ResourceExhausted(
                "HTTP request head exceeds max_head_bytes"));
      }
    }

    const std::size_t amount =
        std::min(connection.limits_.read_chunk_bytes,
                 connection.limits_.max_head_bytes - impl->wire.size);
    auto task =
        read_more(allocator, impl->wire, impl->transport, amount, options);
    if (!task.valid()) {
      impl->reusable = false;
      co_return http_failure<std::optional<HttpIncomingRequest>>(
          Status::ResourceExhausted(
              "HTTP request read coroutine allocation failed"));
    }
    TransferResult result = co_await task;
    if (!result.status.ok()) {
      impl->reusable = false;
      co_return http_failure<std::optional<HttpIncomingRequest>>(
          std::move(result.status));
    }
    if (result.eof || result.bytes == 0) {
      impl->closed = true;
      impl->reusable = false;
      if (impl->wire.size == 0) {
        co_return std::optional<HttpIncomingRequest>{};
      }
      co_return http_failure<std::optional<HttpIncomingRequest>>(
          Status::DataLoss("HTTP peer closed inside request head"));
    }
  }

  if (minor_version != 0 && minor_version != 1) {
    impl->reusable = false;
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::InvalidArgument("unsupported HTTP/1 minor version"));
  }
  const std::size_t head_length = static_cast<std::size_t>(parsed);
  const std::ptrdiff_t method_offset = method - impl->wire.data;
  const std::ptrdiff_t target_offset = target - impl->wire.data;
  if (method_offset < 0 || target_offset < 0 ||
      static_cast<std::size_t>(method_offset) > head_length ||
      static_cast<std::size_t>(target_offset) > head_length ||
      method_length > head_length - static_cast<std::size_t>(method_offset) ||
      target_length > head_length - static_cast<std::size_t>(target_offset)) {
    impl->reusable = false;
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::Internal("HTTP parser returned an invalid request view"));
  }

  Status captured =
      impl->message.capture(impl->wire.data, head_length, header_count);
  if (!captured.ok()) {
    impl->reusable = false;
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        std::move(captured));
  }
  const StringView method_view(impl->message.head.data + method_offset,
                               static_cast<std::uint32_t>(method_length));
  const StringView target_view(impl->message.head.data + target_offset,
                               static_cast<std::uint32_t>(target_length));
  const HttpHeadersView headers = impl->message.view();

  std::size_t host_count = 0;
  for (const HttpHeader &header : headers) {
    if (ascii_iequal(header.name, "host"))
      ++host_count;
  }
  if (minor_version == 1 &&
      (host_count != 1 || !headers.find("host").has_value() ||
       headers.find("host")->empty())) {
    impl->reusable = false;
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        Status::InvalidArgument(
            "HTTP/1.1 request requires exactly one nonempty Host"));
  }

  Result<Framing> framing =
      inspect_framing(headers, connection.limits_.max_body_bytes, true);
  if (!framing) {
    impl->reusable = false;
    co_return http_failure<std::optional<HttpIncomingRequest>>(
        std::move(framing).error());
  }

  const HttpVersion version =
      minor_version == 0 ? HttpVersion::kHttp10 : HttpVersion::kHttp11;
  const bool keep_alive =
      version == HttpVersion::kHttp11
          ? !header_has_token(headers, "connection", "close")
          : header_has_token(headers, "connection", "keep-alive");
  const bool expect_continue =
      header_has_token(headers, "expect", "100-continue");
  const std::uint32_t query_at = target_view.find('?');
  const StringView path = query_at == StringView::npos
                              ? target_view
                              : target_view.substr(0, query_at);
  const StringView query = query_at == StringView::npos
                               ? StringView{}
                               : target_view.substr(query_at + 1);

  impl->wire.consume(head_length);
  impl->body.begin(&impl->transport, &impl->wire, framing->kind,
                   framing->length, connection.limits_, true);
  impl->current_request =
      HttpRequestView{method_view, target_view, path,       query,
                      version,     headers,     keep_alive, expect_continue};
  impl->request_active = true;
  impl->response_started = false;
  impl->continue_sent = false;
  impl->reusable = keep_alive;

  HttpIncomingRequest incoming{
      impl->current_request,
      detail::HttpConnectionAccess::reader(impl->body, allocator)};
  co_return std::optional<HttpIncomingRequest>(std::move(incoming));
}

sched::Task<Status>
http_server_send_continue(mem::AllocatorRef allocator,
                          HttpServerConnection &connection,
                          OperationOptions options) noexcept {
  HttpServerConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->request_active) {
    co_return Status::FailedPrecondition("HTTP server has no active request");
  }
  if (!impl->current_request.expects_continue()) {
    co_return Status::FailedPrecondition(
        "HTTP request did not ask for 100-continue");
  }
  if (impl->continue_sent || impl->response_started) {
    co_return Status::FailedPrecondition(
        "HTTP continue/final response was already started");
  }
  static constexpr char kContinue[] = "HTTP/1.1 100 Continue\r\n\r\n";
  auto task = write_all(
      allocator, impl->transport,
      std::span<const std::byte>(reinterpret_cast<const std::byte *>(kContinue),
                                 sizeof(kContinue) - 1),
      options);
  if (!task.valid()) {
    impl->reusable = false;
    co_return Status::ResourceExhausted(
        "HTTP continue coroutine allocation failed");
  }
  Status status = co_await task;
  if (status.ok()) {
    impl->continue_sent = true;
  } else {
    impl->reusable = false;
  }
  co_return status;
}

sched::Task<Result<HttpBodyWriter>> http_server_start_response(
    mem::AllocatorRef allocator, HttpServerConnection &connection,
    const HttpResponseHead &head, std::optional<std::uint64_t> content_length,
    OperationOptions options) noexcept {
  HttpServerConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->request_active) {
    co_return http_failure<HttpBodyWriter>(
        Status::FailedPrecondition("HTTP server has no active request"));
  }
  if (impl->response_started || impl->writer.active) {
    co_return http_failure<HttpBodyWriter>(
        Status::FailedPrecondition("HTTP response was already started"));
  }
  if (!impl->body.done && !impl->body.abandoned) {
    co_return http_failure<HttpBodyWriter>(Status::FailedPrecondition(
        "drain or abandon the request body before responding"));
  }
  if (head.status < 200) {
    co_return http_failure<HttpBodyWriter>(Status::InvalidArgument(
        "use send_continue for an interim HTTP response"));
  }

  const bool head_request =
      ascii_iequal(impl->current_request.method(), "HEAD");
  const bool body_abandoned = impl->body.abandoned;
  const bool status_suppresses_body =
      head.status == 204 || head.status == 205 || head.status == 304;
  if ((head.status == 204 || head.status == 205) &&
      content_length.value_or(0) != 0) {
    co_return http_failure<HttpBodyWriter>(Status::InvalidArgument(
        "this HTTP status cannot carry a response body"));
  }

  WriteFraming framing = WriteFraming::kFixed;
  std::optional<std::uint64_t> serialized_length = content_length;
  bool effective_keep_alive =
      head.keep_alive && impl->current_request.keep_alive() && !body_abandoned;
  if (head_request || status_suppresses_body) {
    framing = WriteFraming::kSuppressed;
    if (head.status == 204 || head.status == 205)
      serialized_length.reset();
  } else if (content_length.has_value()) {
    framing = WriteFraming::kFixed;
  } else if (impl->current_request.version() == HttpVersion::kHttp11) {
    framing = WriteFraming::kChunked;
  } else {
    framing = WriteFraming::kCloseDelimited;
    effective_keep_alive = false;
  }

  impl->body.retire();

  GrowBuffer &output = impl->output;
  output.clear();
  Status built = build_response_head(output, head, framing, serialized_length,
                                     impl->current_request.version(),
                                     effective_keep_alive);
  if (!built.ok()) {
    output.clear();
    output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
    co_return http_failure<HttpBodyWriter>(std::move(built));
  }
  auto task = write_all(
      allocator, impl->transport,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(output.data), output.size),
      options);
  if (!task.valid()) {
    output.clear();
    output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
    impl->reusable = false;
    co_return http_failure<HttpBodyWriter>(Status::ResourceExhausted(
        "HTTP response-head coroutine allocation failed"));
  }
  Status written = co_await task;
  output.clear();
  output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
  if (!written.ok()) {
    impl->reusable = false;
    co_return http_failure<HttpBodyWriter>(std::move(written));
  }

  begin_writer(impl->writer, impl->transport, impl->output,
               connection.limits_.max_retained_body_bytes, impl,
               ConnectionKind::kServer, framing, content_length,
               !effective_keep_alive);
  impl->response_started = true;
  co_return detail::HttpConnectionAccess::writer(impl->writer, allocator);
}

sched::Task<Status> http_server_send_fixed(mem::AllocatorRef allocator,
                                           HttpServerConnection &connection,
                                           const HttpResponseHead &head,
                                           std::span<const std::byte> body,
                                           OperationOptions options) noexcept {
  auto start = http_server_start_response(
      allocator, connection, head, static_cast<std::uint64_t>(body.size()),
      options);
  if (!start.valid()) {
    co_return Status::ResourceExhausted(
        "HTTP response-start coroutine allocation failed");
  }
  Result<HttpBodyWriter> result = co_await start;
  if (!result)
    co_return std::move(result).error();
  HttpBodyWriter writer = std::move(*result);
  if (!detail::HttpConnectionAccess::writer_suppressed(writer) &&
      !body.empty()) {
    auto write = http_body_write(allocator, writer, body, options);
    if (!write.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP response-body coroutine allocation failed");
    }
    Status status = co_await write;
    if (!status.ok())
      co_return status;
  }
  auto finish = http_body_finish(allocator, writer, {}, options);
  if (!finish.valid()) {
    co_return Status::ResourceExhausted(
        "HTTP response-finish coroutine allocation failed");
  }
  co_return co_await finish;
}

sched::Task<Status> http_server_send_source(mem::AllocatorRef allocator,
                                            HttpServerConnection &connection,
                                            const HttpResponseHead &head,
                                            HttpBodySource source,
                                            OperationOptions options) noexcept {
  if (!detail::HttpConnectionAccess::source_valid(source)) {
    co_return Status::InvalidArgument("HTTP body source is invalid");
  }
  auto start = http_server_start_response(
      allocator, connection, head,
      detail::HttpConnectionAccess::source_length(source), options);
  if (!start.valid()) {
    co_return Status::ResourceExhausted(
        "HTTP response-start coroutine allocation failed");
  }
  Result<HttpBodyWriter> result = co_await start;
  if (!result)
    co_return std::move(result).error();
  HttpBodyWriter writer = std::move(*result);
  if (detail::HttpConnectionAccess::writer_suppressed(writer)) {
    auto finish = http_body_finish(allocator, writer, {}, options);
    if (!finish.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP response-finish coroutine allocation failed");
    }
    co_return co_await finish;
  }

  for (;;) {
    auto next_task =
        detail::HttpConnectionAccess::source_next(allocator, source, options);
    if (!next_task.valid()) {
      co_return Status::ResourceExhausted(
          "HTTP body-source coroutine allocation failed");
    }
    Result<HttpBodyChunk> next = co_await next_task;
    if (!next)
      co_return std::move(next).error();
    if (next->bytes.empty() && !next->end_stream) {
      co_return Status::InvalidArgument(
          "HTTP body source returned an empty non-final chunk");
    }
    if (!next->bytes.empty()) {
      auto write = http_body_write(allocator, writer, next->bytes, options);
      if (!write.valid()) {
        co_return Status::ResourceExhausted(
            "HTTP response-body coroutine allocation failed");
      }
      Status status = co_await write;
      if (!status.ok())
        co_return status;
    }
    if (next->end_stream) {
      auto finish =
          http_body_finish(allocator, writer, next->trailers, options);
      if (!finish.valid()) {
        co_return Status::ResourceExhausted(
            "HTTP response-finish coroutine allocation failed");
      }
      co_return co_await finish;
    }
    if (!next->trailers.empty()) {
      co_return Status::InvalidArgument(
          "HTTP trailers are valid only on the final body chunk");
    }
  }
}

Result<HttpClientConnection>
HttpClientConnection::create(mem::AllocatorRef allocator,
                             HttpTransport transport,
                             HttpLimits limits) noexcept {
  Status valid = validate_limits(allocator, transport, limits);
  if (!valid.ok()) {
    return http_failure<HttpClientConnection>(std::move(valid));
  }
  Impl *impl = allocator.allocate_n<Impl>(1);
  if (impl == nullptr) {
    return http_failure<HttpClientConnection>(
        Status::ResourceExhausted("HTTP client connection allocation failed"));
  }
  new (impl) Impl(allocator, std::move(transport), limits);
  if (!impl->ready()) {
    impl->~Impl();
    allocator.deallocate_n(impl, 1);
    return http_failure<HttpClientConnection>(Status::ResourceExhausted(
        "HTTP client connection state allocation failed"));
  }
  return HttpClientConnection(allocator, limits, impl);
}

HttpClientConnection::~HttpClientConnection() {
  if (impl_ != nullptr) {
    impl_->~Impl();
    allocator_.deallocate_n(impl_, 1);
  }
}

HttpClientConnection::HttpClientConnection(
    HttpClientConnection &&other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)), allocator_(other.allocator_),
      limits_(other.limits_) {}

HttpClientConnection &
HttpClientConnection::operator=(HttpClientConnection &&other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      impl_->~Impl();
      allocator_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(other.impl_, nullptr);
    allocator_ = other.allocator_;
    limits_ = other.limits_;
  }
  return *this;
}

bool HttpClientConnection::valid() const noexcept {
  return impl_ != nullptr && !impl_->closed && impl_->transport.valid();
}

unsigned HttpClientConnection::owner_worker() const noexcept {
  return impl_ == nullptr ? 0 : impl_->transport.owner_worker();
}

sched::Task<Status>
HttpClientConnection::close(OperationOptions options) noexcept {
  if (impl_ == nullptr) {
    return immediate_status(
        allocator_,
        Status::FailedPrecondition("HTTP client connection is invalid"));
  }
  if (impl_->closed)
    return immediate_status(allocator_, Status{});
  impl_->closed = true;
  impl_->reusable = false;
  impl_->request_active = false;
  impl_->response_active = false;
  impl_->body.active = false;
  ++impl_->body.generation;
  if (impl_->writer.active) {
    detail::HttpConnectionAccess::writer_abandoned(impl_->writer);
  }
  return impl_->transport.shutdown(allocator_, options);
}

sched::Task<Result<HttpBodyWriter>> http_client_start_request(
    mem::AllocatorRef allocator, HttpClientConnection &connection,
    const HttpRequestHead &head, std::optional<std::uint64_t> content_length,
    OperationOptions options) noexcept {
  HttpClientConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->transport.valid()) {
    co_return http_failure<HttpBodyWriter>(
        Status::FailedPrecondition("HTTP client connection is invalid"));
  }
  if (impl->request_active || impl->writer.active) {
    co_return http_failure<HttpBodyWriter>(Status::FailedPrecondition(
        "finish the active HTTP request before starting another"));
  }
  if (impl->response_active) {
    if (impl->body.abandoned)
      impl->reusable = false;
    if (!impl->body.done) {
      co_return http_failure<HttpBodyWriter>(Status::FailedPrecondition(
          "drain the HTTP response body before reusing the connection"));
    }
    impl->body.retire();
    impl->response_active = false;
  }
  if (!impl->reusable) {
    co_return http_failure<HttpBodyWriter>(Status::FailedPrecondition(
        "HTTP client connection is no longer reusable"));
  }

  WriteFraming framing;
  if (content_length.has_value()) {
    framing = WriteFraming::kFixed;
  } else if (head.version == HttpVersion::kHttp11) {
    framing = WriteFraming::kChunked;
  } else {
    co_return http_failure<HttpBodyWriter>(Status::InvalidArgument(
        "streaming HTTP/1.0 request requires a known Content-Length"));
  }

  GrowBuffer &output = impl->output;
  output.clear();
  Status built = build_request_head(output, head, framing, content_length);
  if (!built.ok()) {
    output.clear();
    output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
    co_return http_failure<HttpBodyWriter>(std::move(built));
  }
  auto task = write_all(
      allocator, impl->transport,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(output.data), output.size),
      options);
  if (!task.valid()) {
    output.clear();
    output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
    impl->reusable = false;
    co_return http_failure<HttpBodyWriter>(Status::ResourceExhausted(
        "HTTP request-head coroutine allocation failed"));
  }
  Status written = co_await task;
  output.clear();
  output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
  if (!written.ok()) {
    impl->reusable = false;
    co_return http_failure<HttpBodyWriter>(std::move(written));
  }

  begin_writer(impl->writer, impl->transport, impl->output,
               connection.limits_.max_retained_body_bytes, impl,
               ConnectionKind::kClient, framing, content_length,
               !head.keep_alive);
  impl->request_active = true;
  impl->request_finished = false;
  impl->last_request_was_head = ascii_iequal(head.method, "HEAD");
  impl->last_request_expects_continue = head.expect_continue;
  impl->continue_decided = false;
  co_return detail::HttpConnectionAccess::writer(impl->writer, allocator);
}

sched::Task<Result<std::optional<HttpIncomingResponse>>>
http_client_wait_for_continue(mem::AllocatorRef allocator,
                              HttpClientConnection &connection,
                              HttpBodyWriter &writer,
                              OperationOptions options) noexcept {
  HttpClientConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->transport.valid()) {
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::FailedPrecondition("HTTP client connection is invalid"));
  }
  if (!impl->request_active || impl->request_finished ||
      !impl->last_request_expects_continue) {
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::FailedPrecondition(
            "active HTTP request did not ask for 100-continue"));
  }
  if (impl->continue_decided || impl->waiting_for_continue) {
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::FailedPrecondition(
            "HTTP continue decision was already requested"));
  }
  if (!detail::HttpConnectionAccess::writer_matches(writer, impl->writer)) {
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::FailedPrecondition(
            "HTTP continue writer does not belong to this request"));
  }
  if (detail::HttpConnectionAccess::writer_has_written(writer)) {
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::FailedPrecondition(
            "wait for 100-continue before writing the request body"));
  }

  // Expect is an optimization, not permission for a peer that ignores it to
  // deadlock a request. Use a protocol sub-deadline while preserving an
  // earlier caller deadline as the terminal whole-operation budget.
  const time::Duration continue_wait = connection.limits_.expect_continue_wait;
  if (continue_wait == time::Duration::zero()) {
    impl->continue_decided = true;
    co_return std::optional<HttpIncomingResponse>{};
  }
  OperationOptions wait_options = options;
  const time::MonotonicTimePoint fallback_deadline =
      time::deadline_after(continue_wait);
  const bool fallback_owns_deadline =
      !options.deadline.has_value() || fallback_deadline < *options.deadline;
  if (fallback_owns_deadline)
    wait_options.deadline = fallback_deadline;

  impl->waiting_for_continue = true;
  auto task = http_client_read_response(allocator, connection, wait_options);
  if (!task.valid()) {
    impl->waiting_for_continue = false;
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        Status::ResourceExhausted("HTTP continue coroutine allocation failed"));
  }
  Result<HttpIncomingResponse> response = co_await task;
  impl->waiting_for_continue = false;
  if (!response) {
    if (fallback_owns_deadline && response.error().IsDeadlineExceeded()) {
      // The timed receive has fully retired before it reports timeout, so the
      // transport and any partial response bytes remain safe to reuse. A late
      // 100 response is skipped by the normal final-response reader.
      impl->reusable = true;
      impl->continue_decided = true;
      co_return std::optional<HttpIncomingResponse>{};
    }
    co_return http_failure<std::optional<HttpIncomingResponse>>(
        std::move(response).error());
  }
  impl->continue_decided = true;
  if (response->head.status() == 100) {
    impl->body.retire();
    co_return std::optional<HttpIncomingResponse>{};
  }

  detail::HttpConnectionAccess::disarm_writer(writer, impl->writer);
  co_return std::optional<HttpIncomingResponse>(std::move(*response));
}

sched::Task<Result<HttpIncomingResponse>>
http_client_read_response(mem::AllocatorRef allocator,
                          HttpClientConnection &connection,
                          OperationOptions options) noexcept {
  HttpClientConnection::Impl *impl = connection.impl_;
  if (impl == nullptr || impl->closed || !impl->transport.valid()) {
    co_return http_failure<HttpIncomingResponse>(
        Status::FailedPrecondition("HTTP client connection is invalid"));
  }
  const bool waiting_for_continue =
      impl->waiting_for_continue && impl->request_active &&
      !impl->request_finished && impl->writer.active &&
      impl->last_request_expects_continue && !impl->continue_decided;
  const bool waiting_for_final =
      impl->request_active && impl->request_finished && !impl->writer.active;
  if (!waiting_for_continue && !waiting_for_final) {
    co_return http_failure<HttpIncomingResponse>(Status::FailedPrecondition(
        "finish an HTTP request before reading its response"));
  }
  if (impl->response_active) {
    co_return http_failure<HttpIncomingResponse>(
        Status::FailedPrecondition("HTTP response was already returned"));
  }

  for (;;) {
    impl->message.reset_message();
    std::size_t last_length = 0;
    int minor_version = -1;
    int status_code = 0;
    const char *reason = nullptr;
    std::size_t reason_length = 0;
    std::size_t header_count = 0;
    int parsed = -2;

    for (;;) {
      if (impl->wire.size != 0) {
        header_count = impl->message.capacity;
        parsed =
            phr_parse_response(impl->wire.data, impl->wire.size, &minor_version,
                               &status_code, &reason, &reason_length,
                               impl->message.raw, &header_count, last_length);
        if (parsed > 0)
          break;
        if (parsed == -1) {
          impl->reusable = false;
          const Status status =
              header_count == impl->message.capacity
                  ? Status::ResourceExhausted(
                        "HTTP response exceeds max_header_count")
                  : Status::DataLoss("malformed HTTP response head");
          co_return http_failure<HttpIncomingResponse>(status);
        }
        last_length = impl->wire.size;
        if (impl->wire.size >= connection.limits_.max_head_bytes) {
          impl->reusable = false;
          co_return http_failure<HttpIncomingResponse>(
              Status::ResourceExhausted(
                  "HTTP response head exceeds max_head_bytes"));
        }
      }

      const std::size_t amount =
          std::min(connection.limits_.read_chunk_bytes,
                   connection.limits_.max_head_bytes - impl->wire.size);
      auto task =
          read_more(allocator, impl->wire, impl->transport, amount, options);
      if (!task.valid()) {
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP response read coroutine allocation failed"));
      }
      TransferResult result = co_await task;
      if (!result.status.ok()) {
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(std::move(result.status));
      }
      if (result.eof || result.bytes == 0) {
        impl->closed = true;
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(
            Status::DataLoss("HTTP peer closed before a response head"));
      }
    }

    if (minor_version != 0 && minor_version != 1) {
      impl->reusable = false;
      co_return http_failure<HttpIncomingResponse>(
          Status::DataLoss("unsupported HTTP/1 minor version"));
    }
    const std::size_t head_length = static_cast<std::size_t>(parsed);
    const std::ptrdiff_t reason_offset = reason - impl->wire.data;
    if (reason_offset < 0 ||
        static_cast<std::size_t>(reason_offset) > head_length ||
        reason_length > head_length - static_cast<std::size_t>(reason_offset)) {
      impl->reusable = false;
      co_return http_failure<HttpIncomingResponse>(
          Status::Internal("HTTP parser returned an invalid response view"));
    }
    Status captured =
        impl->message.capture(impl->wire.data, head_length, header_count);
    if (!captured.ok()) {
      impl->reusable = false;
      co_return http_failure<HttpIncomingResponse>(std::move(captured));
    }

    const HttpHeadersView headers = impl->message.view();
    const HttpVersion version =
        minor_version == 0 ? HttpVersion::kHttp10 : HttpVersion::kHttp11;
    bool keep_alive =
        version == HttpVersion::kHttp11
            ? !header_has_token(headers, "connection", "close")
            : header_has_token(headers, "connection", "keep-alive");
    const bool informational =
        status_code >= 100 && status_code < 200 && status_code != 101;
    const bool no_body = informational || status_code == 101 ||
                         status_code == 204 || status_code == 205 ||
                         status_code == 304 || impl->last_request_was_head;

    Framing framing{BodyFraming::kNone, 0};
    if (!no_body) {
      Result<Framing> inspected =
          inspect_framing(headers, connection.limits_.max_body_bytes, false);
      if (!inspected) {
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(
            std::move(inspected).error());
      }
      framing = *inspected;
      if (framing.kind == BodyFraming::kCloseDelimited)
        keep_alive = false;
    }
    if (status_code == 101)
      keep_alive = false;

    const StringView reason_view(impl->message.head.data + reason_offset,
                                 static_cast<std::uint32_t>(reason_length));
    impl->wire.consume(head_length);
    if (informational && !(waiting_for_continue && status_code == 100)) {
      // 100 Continue and other non-upgrade informational messages have no
      // body and do not complete the request/response exchange.
      continue;
    }

    if (waiting_for_continue && status_code == 100) {
      impl->body.begin(&impl->transport, &impl->wire, BodyFraming::kNone, 0,
                       connection.limits_, false);
      impl->current_response = HttpResponseView{status_code, reason_view,
                                                version, headers, keep_alive};
      co_return HttpIncomingResponse{
          impl->current_response,
          detail::HttpConnectionAccess::reader(impl->body, allocator)};
    }

    impl->body.begin(&impl->transport, &impl->wire, framing.kind,
                     framing.length, connection.limits_, false);
    impl->current_response = HttpResponseView{status_code, reason_view, version,
                                              headers, keep_alive};
    if (waiting_for_continue) {
      // The peer rejected the request before its body. This is a successful
      // protocol transition, not an abandoned writer/connection.
      impl->writer.active = false;
    }
    impl->response_active = true;
    impl->request_active = false;
    impl->request_finished = false;
    impl->reusable = impl->reusable && keep_alive;

    HttpIncomingResponse incoming{
        impl->current_response,
        detail::HttpConnectionAccess::reader(impl->body, allocator)};
    co_return incoming;
  }
}

sched::Task<Result<HttpIncomingResponse>> http_client_request_fixed(
    mem::AllocatorRef allocator, HttpClientConnection &connection,
    const HttpRequestHead &head, std::span<const std::byte> body,
    OperationOptions options) noexcept {
  if (head.expect_continue) {
    auto start = http_client_start_request(
        allocator, connection, head, static_cast<std::uint64_t>(body.size()),
        options);
    if (!start.valid()) {
      co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
          "HTTP request-start coroutine allocation failed"));
    }
    Result<HttpBodyWriter> result = co_await start;
    if (!result) {
      co_return http_failure<HttpIncomingResponse>(std::move(result).error());
    }
    HttpBodyWriter writer = std::move(*result);
    auto wait =
        http_client_wait_for_continue(allocator, connection, writer, options);
    if (!wait.valid()) {
      co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
          "HTTP continue coroutine allocation failed"));
    }
    Result<std::optional<HttpIncomingResponse>> decision = co_await wait;
    if (!decision) {
      co_return http_failure<HttpIncomingResponse>(std::move(decision).error());
    }
    if (decision->has_value()) {
      co_return std::move(decision->value());
    }
    if (!body.empty()) {
      auto write = http_body_write(allocator, writer, body, options);
      if (!write.valid()) {
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP request-body coroutine allocation failed"));
      }
      Status status = co_await write;
      if (!status.ok()) {
        co_return http_failure<HttpIncomingResponse>(std::move(status));
      }
    }
    auto finish = http_body_finish(allocator, writer, {}, options);
    if (!finish.valid()) {
      co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
          "HTTP request-finish coroutine allocation failed"));
    }
    Status finished = co_await finish;
    if (!finished.ok()) {
      co_return http_failure<HttpIncomingResponse>(std::move(finished));
    }
  } else {
    HttpClientConnection::Impl *impl = connection.impl_;
    if (impl == nullptr || impl->closed || !impl->transport.valid()) {
      co_return http_failure<HttpIncomingResponse>(
          Status::FailedPrecondition("HTTP client connection is invalid"));
    }
    if (impl->request_active || impl->writer.active) {
      co_return http_failure<HttpIncomingResponse>(Status::FailedPrecondition(
          "finish the active HTTP request before starting another"));
    }
    if (impl->response_active) {
      if (impl->body.abandoned)
        impl->reusable = false;
      if (!impl->body.done) {
        co_return http_failure<HttpIncomingResponse>(Status::FailedPrecondition(
            "drain the HTTP response body before reusing the connection"));
      }
      impl->body.retire();
      impl->response_active = false;
    }
    if (!impl->reusable) {
      co_return http_failure<HttpIncomingResponse>(Status::FailedPrecondition(
          "HTTP client connection is no longer reusable"));
    }

    GrowBuffer &output = impl->output;
    output.clear();
    Status built = build_request_head(
        output, head, WriteFraming::kFixed,
        static_cast<std::uint64_t>(body.size()));
    if (!built.ok()) {
      output.clear();
      output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
      co_return http_failure<HttpIncomingResponse>(std::move(built));
    }
    const auto head_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(output.data), output.size);
    Status written;
    if (head_bytes.size() <= kMaxBufferSequenceBytes &&
        body.size() <= kMaxBufferSequenceBytes - head_bytes.size()) {
      const ConstBuffer sequence[] = {ConstBuffer(head_bytes),
                                      ConstBuffer(body)};
      auto write = detail::http_transport_write_all(
          allocator, impl->transport, sequence, options);
      if (!write.valid()) {
        output.clear();
        output.release_if_empty_above(
            connection.limits_.max_retained_head_bytes);
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP fixed request vector-write coroutine allocation failed"));
      }
      written = buffer_sequence_status(co_await write);
    } else {
      const ConstBuffer sequence[] = {ConstBuffer(head_bytes),
                                      ConstBuffer(body)};
      auto write = write_all_sequential(allocator, impl->transport, sequence,
                                        options);
      if (!write.valid()) {
        output.clear();
        output.release_if_empty_above(
            connection.limits_.max_retained_head_bytes);
        impl->reusable = false;
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP fixed request coroutine allocation failed"));
      }
      written = co_await write;
    }
    output.clear();
    output.release_if_empty_above(connection.limits_.max_retained_head_bytes);
    if (!written.ok()) {
      impl->reusable = false;
      co_return http_failure<HttpIncomingResponse>(std::move(written));
    }

    impl->request_active = true;
    impl->request_finished = true;
    impl->last_request_was_head = ascii_iequal(head.method, "HEAD");
    impl->last_request_expects_continue = false;
    impl->continue_decided = false;
    if (!head.keep_alive)
      impl->reusable = false;
  }
  auto response = http_client_read_response(allocator, connection, options);
  if (!response.valid()) {
    co_return http_failure<HttpIncomingResponse>(
        Status::ResourceExhausted("HTTP response coroutine allocation failed"));
  }
  co_return co_await response;
}

sched::Task<Result<HttpIncomingResponse>>
http_client_request_source(mem::AllocatorRef allocator,
                           HttpClientConnection &connection,
                           const HttpRequestHead &head, HttpBodySource source,
                           OperationOptions options) noexcept {
  if (!detail::HttpConnectionAccess::source_valid(source)) {
    co_return http_failure<HttpIncomingResponse>(
        Status::InvalidArgument("HTTP body source is invalid"));
  }
  auto start = http_client_start_request(
      allocator, connection, head,
      detail::HttpConnectionAccess::source_length(source), options);
  if (!start.valid()) {
    co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
        "HTTP request-start coroutine allocation failed"));
  }
  Result<HttpBodyWriter> result = co_await start;
  if (!result) {
    co_return http_failure<HttpIncomingResponse>(std::move(result).error());
  }
  HttpBodyWriter writer = std::move(*result);

  if (head.expect_continue) {
    auto wait =
        http_client_wait_for_continue(allocator, connection, writer, options);
    if (!wait.valid()) {
      co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
          "HTTP continue coroutine allocation failed"));
    }
    Result<std::optional<HttpIncomingResponse>> decision = co_await wait;
    if (!decision) {
      co_return http_failure<HttpIncomingResponse>(std::move(decision).error());
    }
    if (decision->has_value()) {
      co_return std::move(decision->value());
    }
  }

  for (;;) {
    auto next_task =
        detail::HttpConnectionAccess::source_next(allocator, source, options);
    if (!next_task.valid()) {
      co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
          "HTTP body-source coroutine allocation failed"));
    }
    Result<HttpBodyChunk> next = co_await next_task;
    if (!next) {
      co_return http_failure<HttpIncomingResponse>(std::move(next).error());
    }
    if (next->bytes.empty() && !next->end_stream) {
      co_return http_failure<HttpIncomingResponse>(Status::InvalidArgument(
          "HTTP body source returned an empty non-final chunk"));
    }
    if (!next->bytes.empty()) {
      auto write = http_body_write(allocator, writer, next->bytes, options);
      if (!write.valid()) {
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP request-body coroutine allocation failed"));
      }
      Status status = co_await write;
      if (!status.ok()) {
        co_return http_failure<HttpIncomingResponse>(std::move(status));
      }
    }
    if (!next->trailers.empty() && !next->end_stream) {
      co_return http_failure<HttpIncomingResponse>(Status::InvalidArgument(
          "HTTP trailers are valid only on the final body chunk"));
    }
    if (next->end_stream) {
      auto finish =
          http_body_finish(allocator, writer, next->trailers, options);
      if (!finish.valid()) {
        co_return http_failure<HttpIncomingResponse>(Status::ResourceExhausted(
            "HTTP request-finish coroutine allocation failed"));
      }
      Status finished = co_await finish;
      if (!finished.ok()) {
        co_return http_failure<HttpIncomingResponse>(std::move(finished));
      }
      break;
    }
  }

  auto response = http_client_read_response(allocator, connection, options);
  if (!response.valid()) {
    co_return http_failure<HttpIncomingResponse>(
        Status::ResourceExhausted("HTTP response coroutine allocation failed"));
  }
  co_return co_await response;
}

} // namespace hyle::io
