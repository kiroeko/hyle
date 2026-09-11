#include "hyle/io/websocket.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "hyle/container/base64.h"
#include "hyle/container/string_algo.h"
#include "hyle/platform/random.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/sync.h"
#include "websocket_mask_internal.h"

namespace hyle::io {
namespace {

constexpr container::StringView kWebSocketProtocol = "websocket";
constexpr container::StringView kWebSocketGuid =
    "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr std::uint8_t kOpcodeContinuation = 0x0;
constexpr std::uint8_t kOpcodeText = 0x1;
constexpr std::uint8_t kOpcodeBinary = 0x2;
constexpr std::uint8_t kOpcodeClose = 0x8;
constexpr std::uint8_t kOpcodePing = 0x9;
constexpr std::uint8_t kOpcodePong = 0xa;
constexpr std::size_t kMaxControlBytes = 125;
constexpr std::size_t kMaxHandshakeHeaders = 128;

enum class WebSocketRole : std::uint8_t {
  kClient,
  kServer,
};

template <class T> Result<T> ws_failure(Status status) noexcept {
  return make_unexpected(std::move(status));
}

bool ascii_iequal(container::StringView left,
                  container::StringView right) noexcept {
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

container::StringView trim_ows(container::StringView value) noexcept {
  std::uint32_t first = 0;
  std::uint32_t last = value.size();
  while (first != last && (value[first] == ' ' || value[first] == '\t'))
    ++first;
  while (last != first && (value[last - 1] == ' ' || value[last - 1] == '\t'))
    --last;
  return value.substr(first, last - first);
}

bool valid_token(container::StringView value) noexcept {
  if (value.empty())
    return false;
  for (char byte : value) {
    const unsigned char ch = static_cast<unsigned char>(byte);
    const bool alpha_numeric = (ch >= 'a' && ch <= 'z') ||
                               (ch >= 'A' && ch <= 'Z') ||
                               (ch >= '0' && ch <= '9');
    const bool punctuation = ch == '!' || ch == '#' || ch == '$' || ch == '%' ||
                             ch == '&' || ch == '\'' || ch == '*' ||
                             ch == '+' || ch == '-' || ch == '.' || ch == '^' ||
                             ch == '_' || ch == '`' || ch == '|' || ch == '~';
    if (!alpha_numeric && !punctuation)
      return false;
  }
  return true;
}

bool header_present(HttpHeadersView headers,
                    container::StringView name) noexcept {
  for (const HttpHeader &header : headers) {
    if (ascii_iequal(header.name, name))
      return true;
  }
  return false;
}

std::optional<container::StringView> unique_header(HttpHeadersView headers,
                                                   container::StringView name,
                                                   bool &duplicate) noexcept {
  std::optional<container::StringView> result;
  duplicate = false;
  for (const HttpHeader &header : headers) {
    if (!ascii_iequal(header.name, name))
      continue;
    if (result.has_value()) {
      duplicate = true;
      return result;
    }
    result = trim_ows(header.value);
  }
  return result;
}

bool token_list_contains(HttpHeadersView headers,
                         container::StringView header_name,
                         container::StringView wanted) noexcept {
  for (const HttpHeader &header : headers) {
    if (!ascii_iequal(header.name, header_name))
      continue;
    container::StringView value = header.value;
    std::uint32_t begin = 0;
    while (begin <= value.size()) {
      std::uint32_t comma = value.find(',', begin);
      if (comma == container::StringView::npos)
        comma = value.size();
      const container::StringView token =
          trim_ows(value.substr(begin, comma - begin));
      if (ascii_iequal(token, wanted))
        return true;
      if (comma == value.size())
        break;
      begin = comma + 1;
    }
  }
  return false;
}

bool subprotocol_offered(HttpHeadersView headers,
                         container::StringView selected) noexcept {
  for (const HttpHeader &header : headers) {
    if (!ascii_iequal(header.name, "sec-websocket-protocol"))
      continue;
    container::StringView value = header.value;
    std::uint32_t begin = 0;
    while (begin <= value.size()) {
      std::uint32_t comma = value.find(',', begin);
      if (comma == container::StringView::npos)
        comma = value.size();
      const container::StringView token =
          trim_ows(value.substr(begin, comma - begin));
      if (token == selected)
        return true;
      if (comma == value.size())
        break;
      begin = comma + 1;
    }
  }
  return false;
}

Status validate_subprotocol_offer(HttpHeadersView headers) noexcept {
  for (const HttpHeader &header : headers) {
    if (!ascii_iequal(header.name, "sec-websocket-protocol"))
      continue;
    container::StringView value = header.value;
    std::uint32_t begin = 0;
    for (;;) {
      std::uint32_t comma = value.find(',', begin);
      if (comma == container::StringView::npos)
        comma = value.size();
      if (!valid_token(trim_ows(value.substr(begin, comma - begin)))) {
        return Status::InvalidArgument(
            "WebSocket subprotocol offer contains an invalid token");
      }
      if (comma == value.size())
        break;
      begin = comma + 1;
    }
  }
  return Status{};
}

bool subprotocol_was_offered(std::span<const container::StringView> offered,
                             container::StringView selected) noexcept {
  for (container::StringView token : offered) {
    if (token == selected)
      return true;
  }
  return false;
}

bool owned_websocket_header(container::StringView name) noexcept {
  return ascii_iequal(name, "connection") || ascii_iequal(name, "upgrade") ||
         ascii_iequal(name, "sec-websocket-key") ||
         ascii_iequal(name, "sec-websocket-accept") ||
         ascii_iequal(name, "sec-websocket-version") ||
         ascii_iequal(name, "sec-websocket-protocol") ||
         ascii_iequal(name, "sec-websocket-extensions") ||
         ascii_iequal(name, "content-length") ||
         ascii_iequal(name, "transfer-encoding") ||
         ascii_iequal(name, "expect") || ascii_iequal(name, "host") ||
         ascii_iequal(name, "origin");
}

Status validate_extra_headers(std::span<const HttpHeader> headers) noexcept {
  if (headers.size() > kMaxHandshakeHeaders - 4) {
    return Status::ResourceExhausted(
        "WebSocket handshake has too many application headers");
  }
  for (const HttpHeader &header : headers) {
    if (owned_websocket_header(header.name)) {
      return Status::InvalidArgument(
          "WebSocket handshake header is owned by the protocol");
    }
  }
  return Status{};
}

Status validate_limits(const WebSocketLimits &limits) noexcept {
  if (limits.max_message_bytes == 0 || limits.max_frame_bytes == 0 ||
      limits.write_fragment_bytes == 0 ||
      limits.write_fragment_bytes > limits.max_frame_bytes ||
      limits.max_frame_bytes > 0x7fff'ffff'ffff'ffffull) {
    return Status::InvalidArgument("WebSocket limits are invalid");
  }
  return Status{};
}

class ByteBuffer {
public:
  explicit ByteBuffer(mem::AllocatorRef allocator) noexcept
      : allocator_(allocator) {}
  ~ByteBuffer() { reset(); }
  ByteBuffer(const ByteBuffer &) = delete;
  ByteBuffer &operator=(const ByteBuffer &) = delete;

  bool reserve(std::size_t size) noexcept {
    if (size <= capacity_)
      return true;
    void *next = allocator_.reallocate(data_, capacity_, size,
                                       alignof(std::max_align_t));
    if (next == nullptr)
      return false;
    data_ = static_cast<std::byte *>(next);
    capacity_ = size;
    return true;
  }

  bool assign(container::StringView value) noexcept {
    if (!reserve(value.size()))
      return false;
    if (!value.empty())
      std::memcpy(data_, value.data(), value.size());
    size_ = value.size();
    return true;
  }

  void reset() noexcept {
    if (data_ != nullptr)
      allocator_.deallocate(data_, capacity_, alignof(std::max_align_t));
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  std::byte *data() noexcept { return data_; }
  const std::byte *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  void set_size(std::size_t size) noexcept { size_ = size; }
  container::StringView text_view() const noexcept {
    return container::StringView(reinterpret_cast<const char *>(data_),
                                 static_cast<std::uint32_t>(size_));
  }

private:
  mem::AllocatorRef allocator_;
  std::byte *data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

std::uint32_t load_be32(const std::byte *input) noexcept {
  return (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[0]))
          << 24) |
         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[1]))
          << 16) |
         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[2]))
          << 8) |
         static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[3]));
}

void store_be32(std::byte *output, std::uint32_t value) noexcept {
  output[0] = static_cast<std::byte>(value >> 24);
  output[1] = static_cast<std::byte>(value >> 16);
  output[2] = static_cast<std::byte>(value >> 8);
  output[3] = static_cast<std::byte>(value);
}

class Sha1 {
public:
  void update(std::span<const std::byte> input) noexcept {
    total_bytes_ += input.size();
    while (!input.empty()) {
      const std::size_t count =
          std::min(input.size(), block_.size() - buffered_);
      std::memcpy(block_.data() + buffered_, input.data(), count);
      buffered_ += count;
      input = input.subspan(count);
      if (buffered_ == block_.size()) {
        compress(block_.data());
        buffered_ = 0;
      }
    }
  }

  void update(container::StringView input) noexcept {
    update(std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(input.data()), input.size()));
  }

  std::array<std::byte, 20> finish() noexcept {
    const std::uint64_t bits = total_bytes_ * 8;
    block_[buffered_++] = std::byte{0x80};
    if (buffered_ > 56) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                block_.end(), std::byte{});
      compress(block_.data());
      buffered_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_),
              block_.begin() + 56, std::byte{});
    for (unsigned i = 0; i != 8; ++i)
      block_[56 + i] = static_cast<std::byte>(bits >> (56 - i * 8));
    compress(block_.data());

    std::array<std::byte, 20> digest{};
    for (std::size_t i = 0; i != state_.size(); ++i)
      store_be32(digest.data() + i * 4, state_[i]);
    return digest;
  }

private:
  void compress(const std::byte *block) noexcept {
    std::array<std::uint32_t, 80> words{};
    for (std::size_t i = 0; i != 16; ++i)
      words[i] = load_be32(block + i * 4);
    for (std::size_t i = 16; i != words.size(); ++i) {
      words[i] = std::rotl(
          words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16], 1);
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    for (std::size_t i = 0; i != 80; ++i) {
      std::uint32_t f = 0;
      std::uint32_t k = 0;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5a827999u;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ed9eba1u;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8f1bbcdcu;
      } else {
        f = b ^ c ^ d;
        k = 0xca62c1d6u;
      }
      const std::uint32_t next = std::rotl(a, 5) + f + e + k + words[i];
      e = d;
      d = c;
      c = std::rotl(b, 30);
      b = a;
      a = next;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
  }

  std::array<std::uint32_t, 5> state_{0x67452301u, 0xefcdab89u, 0x98badcfeu,
                                      0x10325476u, 0xc3d2e1f0u};
  std::array<std::byte, 64> block_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_ = 0;
};

std::array<char, 28>
websocket_accept_value(container::StringView encoded_key) noexcept {
  Sha1 sha1;
  sha1.update(encoded_key);
  sha1.update(kWebSocketGuid);
  const std::array<std::byte, 20> digest = sha1.finish();
  std::array<char, 28> output{};
  const container::Base64Result result =
      container::base64_encode(digest, output,
                               {container::Base64Alphabet::kStandard,
                                container::Base64EncodePadding::kInclude});
  HYLE_FATAL_ASSERT(result.ok() && result.output_count == output.size());
  return output;
}

class ChaCha20Keys {
public:
  void initialize(std::span<const std::byte, 44> seed) noexcept {
    state_[0] = 0x61707865u;
    state_[1] = 0x3320646eu;
    state_[2] = 0x79622d32u;
    state_[3] = 0x6b206574u;
    for (std::size_t i = 0; i != 8; ++i)
      state_[4 + i] = load_le32(seed.data() + i * 4);
    state_[12] = 0;
    state_[13] = load_le32(seed.data() + 32);
    state_[14] = load_le32(seed.data() + 36);
    state_[15] = load_le32(seed.data() + 40);
    available_ = 0;
  }

  std::array<std::byte, 4> next() noexcept {
    if (available_ < 4)
      refill();
    std::array<std::byte, 4> result{};
    const std::size_t offset = block_.size() - available_;
    std::memcpy(result.data(), block_.data() + offset, result.size());
    available_ -= result.size();
    return result;
  }

private:
  static std::uint32_t load_le32(const std::byte *input) noexcept {
    return static_cast<std::uint32_t>(
               std::to_integer<unsigned char>(input[0])) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[1]))
            << 8) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[2]))
            << 16) |
           (static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[3]))
            << 24);
  }

  static void store_le32(std::byte *output, std::uint32_t value) noexcept {
    output[0] = static_cast<std::byte>(value);
    output[1] = static_cast<std::byte>(value >> 8);
    output[2] = static_cast<std::byte>(value >> 16);
    output[3] = static_cast<std::byte>(value >> 24);
  }

  static void quarter(std::uint32_t &a, std::uint32_t &b, std::uint32_t &c,
                      std::uint32_t &d) noexcept {
    a += b;
    d ^= a;
    d = std::rotl(d, 16);
    c += d;
    b ^= c;
    b = std::rotl(b, 12);
    a += b;
    d ^= a;
    d = std::rotl(d, 8);
    c += d;
    b ^= c;
    b = std::rotl(b, 7);
  }

  void refill() noexcept {
    std::array<std::uint32_t, 16> working = state_;
    for (unsigned round = 0; round != 10; ++round) {
      quarter(working[0], working[4], working[8], working[12]);
      quarter(working[1], working[5], working[9], working[13]);
      quarter(working[2], working[6], working[10], working[14]);
      quarter(working[3], working[7], working[11], working[15]);
      quarter(working[0], working[5], working[10], working[15]);
      quarter(working[1], working[6], working[11], working[12]);
      quarter(working[2], working[7], working[8], working[13]);
      quarter(working[3], working[4], working[9], working[14]);
    }
    for (std::size_t i = 0; i != working.size(); ++i)
      store_le32(block_.data() + i * 4, working[i] + state_[i]);
    // Treat the counter and nonce words as one 128-bit unique block input.
    // This preserves the ordinary IETF layout for the first 2^32 blocks while
    // avoiding a repeated masking-key stream on an exceptionally long-lived
    // connection after the 32-bit counter wraps.
    if (++state_[12] == 0 && ++state_[13] == 0 && ++state_[14] == 0)
      ++state_[15];
    available_ = block_.size();
  }

  std::array<std::uint32_t, 16> state_{};
  std::array<std::byte, 64> block_{};
  std::size_t available_ = 0;
};

class Utf8State {
public:
  void reset() noexcept {
    code_point_ = 0;
    minimum_ = 0;
    remaining_ = 0;
  }

  bool consume(std::span<const std::byte> input) noexcept {
    std::size_t offset = 0;
    while (remaining_ == 0 && input.size() - offset >= sizeof(std::uint64_t)) {
      std::uint64_t word = 0;
      std::memcpy(&word, input.data() + offset, sizeof(word));
      if ((word & 0x8080'8080'8080'8080ull) != 0)
        break;
      offset += sizeof(word);
    }
    for (; offset != input.size(); ++offset) {
      const unsigned char byte = std::to_integer<unsigned char>(input[offset]);
      if (remaining_ == 0) {
        if (byte < 0x80)
          continue;
        if (byte >= 0xc2 && byte <= 0xdf) {
          code_point_ = byte & 0x1fu;
          minimum_ = 0x80;
          remaining_ = 1;
        } else if (byte >= 0xe0 && byte <= 0xef) {
          code_point_ = byte & 0x0fu;
          minimum_ = 0x800;
          remaining_ = 2;
        } else if (byte >= 0xf0 && byte <= 0xf4) {
          code_point_ = byte & 0x07u;
          minimum_ = 0x10000;
          remaining_ = 3;
        } else {
          return false;
        }
        continue;
      }
      if ((byte & 0xc0u) != 0x80u)
        return false;
      code_point_ = (code_point_ << 6) | (byte & 0x3fu);
      --remaining_;
      if (remaining_ == 0 &&
          (code_point_ < minimum_ || code_point_ > 0x10ffffu ||
           (code_point_ >= 0xd800u && code_point_ <= 0xdfffu))) {
        return false;
      }
    }
    return true;
  }

  bool complete() const noexcept { return remaining_ == 0; }

private:
  std::uint32_t code_point_ = 0;
  std::uint32_t minimum_ = 0;
  unsigned remaining_ = 0;
};

bool valid_close_code(std::uint16_t code) noexcept {
  if (code >= 3000 && code <= 4999)
    return true;
  switch (code) {
  case 1000:
  case 1001:
  case 1002:
  case 1003:
  case 1007:
  case 1008:
  case 1009:
  case 1010:
  case 1011:
  case 1012:
  case 1013:
  case 1014:
    return true;
  default:
    return false;
  }
}

std::size_t encode_frame_header(std::array<std::byte, 14> &output,
                                std::uint8_t opcode, bool final,
                                std::uint64_t length, bool masked,
                                const std::array<std::byte, 4> &key) noexcept {
  output[0] = static_cast<std::byte>((final ? 0x80u : 0u) | opcode);
  const unsigned mask_bit = masked ? 0x80u : 0u;
  std::size_t size = 2;
  if (length <= 125) {
    output[1] = static_cast<std::byte>(mask_bit | length);
  } else if (length <= 0xffff) {
    output[1] = static_cast<std::byte>(mask_bit | 126u);
    output[2] = static_cast<std::byte>(length >> 8);
    output[3] = static_cast<std::byte>(length);
    size = 4;
  } else {
    output[1] = static_cast<std::byte>(mask_bit | 127u);
    for (unsigned i = 0; i != 8; ++i)
      output[2 + i] = static_cast<std::byte>(length >> (56 - i * 8));
    size = 10;
  }
  if (masked) {
    std::memcpy(output.data() + size, key.data(), key.size());
    size += key.size();
  }
  return size;
}

void signal_write_gate(void *context) noexcept {
  static_cast<sched::Event *>(context)->signal();
}

class WriteGate {
public:
  Result<void> try_acquire() noexcept {
    if (held_)
      return make_unexpected(Status::FailedPrecondition(
          "another WebSocket frame write is in flight"));
    held_ = true;
    ready_.reset();
    return {};
  }

  Status acquire_control(const OperationOptions &options) noexcept {
    while (held_) {
      if (options.cancel != nullptr && options.cancel->is_cancelled())
        return Status::Cancelled("WebSocket control write cancelled");
      if (options.deadline.has_value() &&
          *options.deadline <= time::monotonic_now()) {
        return Status::DeadlineExceeded(
            "WebSocket control write deadline expired");
      }

      ready_.reset();
      if (!held_)
        break;
      bool woke = true;
      if (options.cancel != nullptr) {
        sched::CancelCallback callback(*options.cancel, signal_write_gate,
                                       &ready_);
        if (options.deadline.has_value())
          woke = ready_.wait_until(*options.deadline);
        else
          ready_.wait();
      } else if (options.deadline.has_value()) {
        woke = ready_.wait_until(*options.deadline);
      } else {
        ready_.wait();
      }
      if (!woke)
        return Status::DeadlineExceeded(
            "WebSocket control write deadline expired");
    }
    held_ = true;
    ready_.reset();
    return Status{};
  }

  void release() noexcept {
    HYLE_FATAL_ASSERT(held_);
    held_ = false;
    ready_.signal();
  }

private:
  sched::Event ready_{sched::EventResetMode::kManual, true};
  bool held_ = false;
};

class GateGuard {
public:
  explicit GateGuard(WriteGate &gate) noexcept : gate_(&gate) {}
  ~GateGuard() { gate_->release(); }
  GateGuard(const GateGuard &) = delete;
  GateGuard &operator=(const GateGuard &) = delete;

private:
  WriteGate *gate_;
};

class BoolGuard {
public:
  explicit BoolGuard(bool &value) noexcept : value_(&value) { value = true; }
  ~BoolGuard() { *value_ = false; }
  BoolGuard(const BoolGuard &) = delete;
  BoolGuard &operator=(const BoolGuard &) = delete;

private:
  bool *value_;
};

} // namespace

namespace detail {

struct WebSocketState;

struct WebSocketWriteState {
  WebSocketState *owner = nullptr;
  std::uint64_t generation = 0;
  std::uint64_t message_bytes = 0;
  WebSocketMessageType type = WebSocketMessageType::kBinary;
  Utf8State utf8{};
  bool active = false;
  bool busy = false;
  bool first_frame = true;
};

struct WebSocketState {
  WebSocketState(mem::AllocatorRef source, WebSocketRole endpoint_role,
                 WebSocketLimits configured_limits, void *callback_context,
                 WebSocketControlCallback callback) noexcept
      : allocator(source), selected_protocol(source), write_scratch(source),
        role(endpoint_role), limits(configured_limits),
        control_context(callback_context), control_callback(callback) {
    writer.owner = this;
  }

  mem::AllocatorRef allocator;
  HttpUpgradeStream stream;
  ByteBuffer selected_protocol;
  ByteBuffer write_scratch;
  WriteGate write_gate;
  ChaCha20Keys mask_keys;
  WebSocketWriteState writer;
  Utf8State read_utf8;
  std::array<std::byte, kMaxControlBytes> control_payload{};
  std::array<char, kMaxControlBytes - 2> close_reason{};
  std::array<std::byte, 4> frame_mask{};
  WebSocketRole role;
  WebSocketLimits limits;
  WebSocketMessageType read_type = WebSocketMessageType::kBinary;
  std::uint64_t read_message_bytes = 0;
  std::uint64_t frame_remaining = 0;
  std::size_t frame_mask_offset = 0;
  std::size_t close_reason_size = 0;
  void *control_context = nullptr;
  WebSocketControlCallback control_callback = nullptr;
  bool frame_active = false;
  bool frame_final = false;
  bool read_message_active = false;
  bool read_busy = false;
  bool close_busy = false;
  bool sent_close = false;
  bool received_close = false;
  bool failed = false;
};

} // namespace detail

struct WebSocketConnection::Impl : detail::WebSocketState {
  using detail::WebSocketState::WebSocketState;
};

namespace {

Status validate_server_handshake(const HttpRequestView &request,
                                 const WebSocketServerOptions &options,
                                 container::StringView &encoded_key) noexcept {
  Status valid = validate_limits(options.limits);
  if (!valid.ok())
    return valid;
  valid = validate_extra_headers(options.headers);
  if (!valid.ok())
    return valid;
  if (request.version() != HttpVersion::kHttp11 || request.method() != "GET" ||
      !token_list_contains(request.headers(), "connection", "upgrade") ||
      !token_list_contains(request.headers(), "upgrade", "websocket")) {
    return Status::InvalidArgument(
        "request is not a WebSocket HTTP/1.1 GET Upgrade");
  }

  bool duplicate = false;
  const std::optional<container::StringView> host =
      unique_header(request.headers(), "host", duplicate);
  if (duplicate || !host.has_value() || host->empty()) {
    return Status::InvalidArgument(
        "WebSocket HTTP/1.1 request requires one non-empty Host header");
  }
  const std::optional<container::StringView> version =
      unique_header(request.headers(), "sec-websocket-version", duplicate);
  if (duplicate || !version.has_value() || *version != "13") {
    return Status::InvalidArgument(
        "WebSocket request requires one Sec-WebSocket-Version: 13");
  }
  const std::optional<container::StringView> key =
      unique_header(request.headers(), "sec-websocket-key", duplicate);
  if (duplicate || !key.has_value()) {
    return Status::InvalidArgument(
        "WebSocket request requires one Sec-WebSocket-Key");
  }
  std::array<std::byte, 16> decoded_key{};
  const container::Base64Result decoded = container::base64_decode(
      std::span<const char>(key->data(), key->size()), decoded_key,
      {container::Base64Alphabet::kStandard,
       container::Base64DecodePadding::kRequired,
       container::Base64Whitespace::kReject});
  if (!decoded.ok() || decoded.output_count != decoded_key.size()) {
    return Status::InvalidArgument(
        "Sec-WebSocket-Key must be canonical Base64 for 16 bytes");
  }
  if (header_present(request.headers(), "sec-websocket-extensions")) {
    return Status::InvalidArgument(
        "WebSocket extensions are not supported by this connection");
  }
  valid = validate_subprotocol_offer(request.headers());
  if (!valid.ok())
    return valid;
  if (!options.subprotocol.empty() &&
      (!valid_token(options.subprotocol) ||
       !subprotocol_offered(request.headers(), options.subprotocol))) {
    return Status::InvalidArgument(
        "selected WebSocket subprotocol was not offered by the client");
  }
  encoded_key = *key;
  return Status{};
}

Status validate_client_options(const WebSocketClientOptions &options) noexcept {
  Status valid = validate_limits(options.limits);
  if (!valid.ok())
    return valid;
  valid = validate_extra_headers(options.headers);
  if (!valid.ok())
    return valid;
  if (options.authority.empty() || options.target.empty()) {
    return Status::InvalidArgument(
        "WebSocket client authority and target are required");
  }
  for (std::size_t i = 0; i != options.subprotocols.size(); ++i) {
    if (!valid_token(options.subprotocols[i])) {
      return Status::InvalidArgument(
          "WebSocket subprotocol is not an HTTP token");
    }
    for (std::size_t j = 0; j != i; ++j) {
      if (options.subprotocols[j] == options.subprotocols[i]) {
        return Status::InvalidArgument(
            "WebSocket subprotocol offer contains a duplicate");
      }
    }
  }
  return Status{};
}

Status join_subprotocols(mem::AllocatorRef allocator,
                         std::span<const container::StringView> protocols,
                         ByteBuffer &output) noexcept {
  (void)allocator;
  std::size_t total = 0;
  for (std::size_t i = 0; i != protocols.size(); ++i) {
    const std::size_t separator = i == 0 ? 0 : 2;
    if (total > (std::numeric_limits<std::uint32_t>::max)() - separator ||
        protocols[i].size() >
            (std::numeric_limits<std::uint32_t>::max)() - total - separator) {
      return Status::ResourceExhausted(
          "WebSocket subprotocol offer is too large");
    }
    total += separator + protocols[i].size();
  }
  if (!output.reserve(total)) {
    return Status::ResourceExhausted(
        "WebSocket subprotocol offer allocation failed");
  }
  char *destination = reinterpret_cast<char *>(output.data());
  std::size_t offset = 0;
  for (std::size_t i = 0; i != protocols.size(); ++i) {
    if (i != 0) {
      destination[offset++] = ',';
      destination[offset++] = ' ';
    }
    std::memcpy(destination + offset, protocols[i].data(), protocols[i].size());
    offset += protocols[i].size();
  }
  output.set_size(total);
  return Status{};
}

Status validate_client_response(const HttpResponseView &response,
                                const std::array<char, 28> &expected_accept,
                                std::span<const container::StringView> offered,
                                container::StringView &selected) noexcept {
  if (response.status() != 101) {
    return Status::FailedPrecondition(
        "WebSocket server rejected the HTTP Upgrade request");
  }
  bool duplicate = false;
  const std::optional<container::StringView> accept =
      unique_header(response.headers(), "sec-websocket-accept", duplicate);
  const container::StringView expected(
      expected_accept.data(),
      static_cast<std::uint32_t>(expected_accept.size()));
  if (duplicate || !accept.has_value() || *accept != expected) {
    return Status::DataLoss("WebSocket server returned an invalid accept key");
  }
  if (header_present(response.headers(), "sec-websocket-extensions")) {
    return Status::DataLoss(
        "WebSocket server selected an unsupported extension");
  }
  const std::optional<container::StringView> protocol =
      unique_header(response.headers(), "sec-websocket-protocol", duplicate);
  if (duplicate) {
    return Status::DataLoss(
        "WebSocket server returned multiple selected subprotocols");
  }
  if (protocol.has_value()) {
    if (!valid_token(*protocol) ||
        !subprotocol_was_offered(offered, *protocol)) {
      return Status::DataLoss(
          "WebSocket server selected a subprotocol that was not offered");
    }
    selected = *protocol;
  } else {
    selected = {};
  }
  return Status{};
}

sched::Task<Status> read_exact(mem::AllocatorRef allocator,
                               detail::WebSocketState &state,
                               std::span<std::byte> output,
                               OperationOptions operation) noexcept {
  // The promise consumes this argument when allocating the coroutine frame.
  (void)allocator;
  std::size_t offset = 0;
  while (offset != output.size()) {
    auto task = state.stream.read_some(output.subspan(offset), operation);
    if (!task.valid()) {
      state.failed = true;
      co_return Status::ResourceExhausted(
          "WebSocket read coroutine allocation failed");
    }
    TransferResult result = co_await task;
    if (!result.status.ok()) {
      state.failed = true;
      co_return std::move(result.status);
    }
    if (result.eof || result.bytes == 0 ||
        result.bytes > output.size() - offset) {
      state.failed = true;
      co_return Status::DataLoss(
          "WebSocket peer closed without a complete Close handshake");
    }
    offset += result.bytes;
  }
  co_return Status{};
}

sched::Task<Status>
write_frame(mem::AllocatorRef allocator, detail::WebSocketState &state,
            std::uint8_t opcode, bool final, std::span<const std::byte> payload,
            bool internal_control, OperationOptions operation) noexcept {
  // The promise consumes this argument when allocating the coroutine frame.
  (void)allocator;
  if (state.failed || state.received_close || !state.stream.valid()) {
    co_return Status::FailedPrecondition(
        "WebSocket connection is not writable");
  }
  if (payload.size() > state.limits.max_frame_bytes ||
      ((opcode & 0x08u) != 0 && payload.size() > kMaxControlBytes)) {
    co_return Status::OutOfRange(
        "WebSocket frame exceeds its configured limit");
  }

  if (internal_control) {
    Status acquired = state.write_gate.acquire_control(operation);
    if (!acquired.ok())
      co_return acquired;
  } else {
    Result<void> acquired = state.write_gate.try_acquire();
    if (!acquired)
      co_return std::move(acquired).error();
  }
  GateGuard gate(state.write_gate);

  std::array<std::byte, 4> mask{};
  std::span<const std::byte> wire_payload = payload;
  const bool masked = state.role == WebSocketRole::kClient;
  if (masked) {
    mask = state.mask_keys.next();
    if (!state.write_scratch.reserve(payload.size())) {
      co_return Status::ResourceExhausted(
          "WebSocket masking scratch allocation failed");
    }
    if (!payload.empty())
      std::memcpy(state.write_scratch.data(), payload.data(), payload.size());
    state.write_scratch.set_size(payload.size());
    std::span<std::byte> mutable_payload(state.write_scratch.data(),
                                         payload.size());
    detail::websocket_apply_mask(mutable_payload, mask);
    wire_payload = mutable_payload;
  }

  std::array<std::byte, 14> header{};
  const std::size_t header_size =
      encode_frame_header(header, opcode, final, payload.size(), masked, mask);
  const std::array<ConstBuffer, 2> buffers{
      ConstBuffer(header.data(), header_size),
      ConstBuffer(wire_payload.data(), wire_payload.size())};
  auto task = state.stream.write_all(buffers, operation);
  if (!task.valid()) {
    state.failed = true;
    co_return Status::ResourceExhausted(
        "WebSocket frame coroutine allocation failed");
  }
  BufferSequenceResult result = co_await task;
  if (!result.status.ok()) {
    state.failed = true;
    co_return std::move(result.status);
  }
  if (!result.progress.complete) {
    state.failed = true;
    co_return Status::DataLoss(
        "WebSocket transport stopped before completing a frame");
  }
  co_return Status{};
}

sched::Task<Result<WebSocketReadResult>>
protocol_failure(mem::AllocatorRef allocator, detail::WebSocketState &state,
                 std::uint16_t close_code, Status error,
                 OperationOptions operation) noexcept {
  if (!state.sent_close && state.stream.valid()) {
    std::array<std::byte, 2> payload{static_cast<std::byte>(close_code >> 8),
                                     static_cast<std::byte>(close_code)};
    auto close_task = write_frame(allocator, state, kOpcodeClose, true, payload,
                                  true, operation);
    if (close_task.valid()) {
      Status ignored = co_await close_task;
      if (ignored.ok())
        state.sent_close = true;
    }
  }
  state.failed = true;
  if (state.stream.valid()) {
    auto shutdown = state.stream.shutdown(operation);
    if (shutdown.valid())
      (void)co_await shutdown;
  }
  co_return ws_failure<WebSocketReadResult>(std::move(error));
}

sched::Task<Result<WebSocketReadResult>>
read_message_chunk(mem::AllocatorRef allocator, detail::WebSocketState &state,
                   std::span<std::byte> output,
                   OperationOptions operation) noexcept {
  for (;;) {
    if (state.frame_active) {
      if (output.empty()) {
        co_return ws_failure<WebSocketReadResult>(Status::InvalidArgument(
            "WebSocket read buffer must be non-empty for payload progress"));
      }
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(state.frame_remaining, output.size()));
      auto read = read_exact(allocator, state, output.first(count), operation);
      if (!read.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket payload coroutine allocation failed"));
      }
      Status status = co_await read;
      if (!status.ok())
        co_return ws_failure<WebSocketReadResult>(std::move(status));
      if (state.role == WebSocketRole::kServer) {
        detail::websocket_apply_mask(output.first(count), state.frame_mask,
                                     state.frame_mask_offset);
        state.frame_mask_offset += count;
      }
      if (state.read_type == WebSocketMessageType::kText &&
          !state.read_utf8.consume(output.first(count))) {
        auto failed = protocol_failure(
            allocator, state, 1007,
            Status::DataLoss("WebSocket text message contains invalid UTF-8"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
      state.frame_remaining -= count;
      bool end_message = false;
      if (state.frame_remaining == 0) {
        state.frame_active = false;
        if (state.frame_final) {
          if (state.read_type == WebSocketMessageType::kText &&
              !state.read_utf8.complete()) {
            auto failed = protocol_failure(
                allocator, state, 1007,
                Status::DataLoss(
                    "WebSocket text message ends inside a UTF-8 sequence"),
                operation);
            if (!failed.valid()) {
              state.failed = true;
              co_return ws_failure<
                  WebSocketReadResult>(Status::ResourceExhausted(
                  "WebSocket protocol-failure coroutine allocation failed"));
            }
            co_return co_await failed;
          }
          end_message = true;
          state.read_message_active = false;
          state.read_message_bytes = 0;
          state.read_utf8.reset();
        }
      }
      co_return WebSocketReadResult{count, state.read_type, end_message,
                                    std::nullopt};
    }

    std::array<std::byte, 14> header{};
    auto first = read_exact(allocator, state,
                            std::span<std::byte>(header.data(), 2), operation);
    if (!first.valid()) {
      state.failed = true;
      co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
          "WebSocket frame-header coroutine allocation failed"));
    }
    Status first_status = co_await first;
    if (!first_status.ok())
      co_return ws_failure<WebSocketReadResult>(std::move(first_status));

    const unsigned first_byte = std::to_integer<unsigned char>(header[0]);
    const unsigned second_byte = std::to_integer<unsigned char>(header[1]);
    const bool final = (first_byte & 0x80u) != 0;
    const std::uint8_t opcode = static_cast<std::uint8_t>(first_byte & 0x0fu);
    const bool masked = (second_byte & 0x80u) != 0;
    const unsigned length_code = second_byte & 0x7fu;
    const bool control = (opcode & 0x08u) != 0;
    if ((first_byte & 0x70u) != 0 ||
        !(opcode == kOpcodeContinuation || opcode == kOpcodeText ||
          opcode == kOpcodeBinary || opcode == kOpcodeClose ||
          opcode == kOpcodePing || opcode == kOpcodePong) ||
        masked != (state.role == WebSocketRole::kServer) ||
        (control && (!final || length_code > 125))) {
      auto failed = protocol_failure(
          allocator, state, 1002,
          Status::DataLoss("WebSocket frame header violates RFC 6455"),
          operation);
      if (!failed.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket protocol-failure coroutine allocation failed"));
      }
      co_return co_await failed;
    }

    std::uint64_t length = length_code;
    std::size_t header_offset = 2;
    if (length_code == 126) {
      auto extended = read_exact(
          allocator, state,
          std::span<std::byte>(header.data() + header_offset, 2), operation);
      if (!extended.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket extended-length coroutine allocation failed"));
      }
      Status status = co_await extended;
      if (!status.ok())
        co_return ws_failure<WebSocketReadResult>(std::move(status));
      length =
          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(header[2]))
           << 8) |
          std::to_integer<unsigned char>(header[3]);
      header_offset += 2;
      if (length < 126) {
        auto failed = protocol_failure(
            allocator, state, 1002,
            Status::DataLoss("WebSocket frame length is not canonical"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
    } else if (length_code == 127) {
      auto extended = read_exact(
          allocator, state,
          std::span<std::byte>(header.data() + header_offset, 8), operation);
      if (!extended.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket extended-length coroutine allocation failed"));
      }
      Status status = co_await extended;
      if (!status.ok())
        co_return ws_failure<WebSocketReadResult>(std::move(status));
      length = 0;
      for (std::size_t i = 0; i != 8; ++i) {
        length = (length << 8) |
                 std::to_integer<unsigned char>(header[header_offset + i]);
      }
      header_offset += 8;
      if (length <= 0xffff || (length & 0x8000'0000'0000'0000ull) != 0) {
        auto failed = protocol_failure(
            allocator, state, 1002,
            Status::DataLoss("WebSocket frame length is not canonical"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
    }

    std::array<std::byte, 4> mask{};
    if (masked) {
      auto key = read_exact(
          allocator, state,
          std::span<std::byte>(header.data() + header_offset, mask.size()),
          operation);
      if (!key.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket mask-key coroutine allocation failed"));
      }
      Status status = co_await key;
      if (!status.ok())
        co_return ws_failure<WebSocketReadResult>(std::move(status));
      std::memcpy(mask.data(), header.data() + header_offset, mask.size());
    }

    if (control) {
      const std::size_t control_size = static_cast<std::size_t>(length);
      if (control_size != 0) {
        auto payload = read_exact(
            allocator, state,
            std::span<std::byte>(state.control_payload.data(), control_size),
            operation);
        if (!payload.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket control coroutine allocation failed"));
        }
        Status status = co_await payload;
        if (!status.ok())
          co_return ws_failure<WebSocketReadResult>(std::move(status));
        if (masked) {
          detail::websocket_apply_mask(
              std::span<std::byte>(state.control_payload.data(), control_size),
              mask);
        }
      }
      const std::span<const std::byte> payload(state.control_payload.data(),
                                               control_size);
      if (opcode == kOpcodePing || opcode == kOpcodePong) {
        if (state.control_callback != nullptr) {
          state.control_callback(state.control_context,
                                 opcode == kOpcodePing
                                     ? WebSocketControlType::kPing
                                     : WebSocketControlType::kPong,
                                 payload);
        }
        if (opcode == kOpcodePing && !state.sent_close) {
          auto pong = write_frame(allocator, state, kOpcodePong, true, payload,
                                  true, operation);
          if (!pong.valid()) {
            state.failed = true;
            co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
                "WebSocket pong coroutine allocation failed"));
          }
          Status status = co_await pong;
          if (!status.ok())
            co_return ws_failure<WebSocketReadResult>(std::move(status));
        }
        continue;
      }

      if (control_size == 1) {
        auto failed = protocol_failure(
            allocator, state, 1002,
            Status::DataLoss("WebSocket Close payload has a truncated code"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }

      std::optional<std::uint16_t> close_code;
      container::StringView reason{};
      if (control_size >= 2) {
        const std::uint16_t code = static_cast<std::uint16_t>(
            (std::to_integer<unsigned char>(state.control_payload[0]) << 8) |
            std::to_integer<unsigned char>(state.control_payload[1]));
        if (!valid_close_code(code)) {
          auto failed = protocol_failure(
              allocator, state, 1002,
              Status::DataLoss("WebSocket Close code is invalid"), operation);
          if (!failed.valid()) {
            state.failed = true;
            co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
                "WebSocket protocol-failure coroutine allocation failed"));
          }
          co_return co_await failed;
        }
        close_code = code;
        reason = container::StringView(
            reinterpret_cast<const char *>(state.control_payload.data() + 2),
            static_cast<std::uint32_t>(control_size - 2));
        if (!container::utf8_validate(reason)) {
          auto failed = protocol_failure(
              allocator, state, 1007,
              Status::DataLoss("WebSocket Close reason is not valid UTF-8"),
              operation);
          if (!failed.valid()) {
            state.failed = true;
            co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
                "WebSocket protocol-failure coroutine allocation failed"));
          }
          co_return co_await failed;
        }
        state.close_reason_size = reason.size();
        if (!reason.empty()) {
          std::memcpy(state.close_reason.data(), reason.data(), reason.size());
        }
      } else {
        state.close_reason_size = 0;
      }

      if (!state.sent_close) {
        auto reply = write_frame(allocator, state, kOpcodeClose, true, payload,
                                 true, operation);
        if (!reply.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket Close reply coroutine allocation failed"));
        }
        Status status = co_await reply;
        if (!status.ok())
          co_return ws_failure<WebSocketReadResult>(std::move(status));
        state.sent_close = true;
      }
      state.received_close = true;
      auto shutdown = state.stream.shutdown(operation);
      if (!shutdown.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket shutdown coroutine allocation failed"));
      }
      Status shutdown_status = co_await shutdown;
      if (!shutdown_status.ok()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(std::move(shutdown_status));
      }
      const container::StringView stored_reason(
          state.close_reason.data(),
          static_cast<std::uint32_t>(state.close_reason_size));
      co_return WebSocketReadResult{0, WebSocketMessageType::kBinary, false,
                                    WebSocketClose{close_code, stored_reason}};
    }

    if (length > state.limits.max_frame_bytes) {
      auto failed = protocol_failure(
          allocator, state, 1009,
          Status::ResourceExhausted("WebSocket frame exceeds max_frame_bytes"),
          operation);
      if (!failed.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket protocol-failure coroutine allocation failed"));
      }
      co_return co_await failed;
    }

    if (opcode == kOpcodeContinuation) {
      if (!state.read_message_active) {
        auto failed = protocol_failure(
            allocator, state, 1002,
            Status::DataLoss("unexpected WebSocket continuation frame"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
    } else {
      if (state.read_message_active) {
        auto failed = protocol_failure(
            allocator, state, 1002,
            Status::DataLoss(
                "new WebSocket data frame interrupted a fragmented message"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
      state.read_message_active = true;
      state.read_type = opcode == kOpcodeText ? WebSocketMessageType::kText
                                              : WebSocketMessageType::kBinary;
      state.read_message_bytes = 0;
      state.read_utf8.reset();
    }

    if (state.read_message_bytes > state.limits.max_message_bytes ||
        length > state.limits.max_message_bytes - state.read_message_bytes) {
      auto failed =
          protocol_failure(allocator, state, 1009,
                           Status::ResourceExhausted(
                               "WebSocket message exceeds max_message_bytes"),
                           operation);
      if (!failed.valid()) {
        state.failed = true;
        co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
            "WebSocket protocol-failure coroutine allocation failed"));
      }
      co_return co_await failed;
    }
    state.read_message_bytes += length;
    state.frame_remaining = length;
    state.frame_mask = mask;
    state.frame_mask_offset = 0;
    state.frame_final = final;
    state.frame_active = length != 0;

    if (length == 0 && final) {
      if (state.read_type == WebSocketMessageType::kText &&
          !state.read_utf8.complete()) {
        auto failed = protocol_failure(
            allocator, state, 1007,
            Status::DataLoss(
                "WebSocket text message ends inside a UTF-8 sequence"),
            operation);
        if (!failed.valid()) {
          state.failed = true;
          co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
              "WebSocket protocol-failure coroutine allocation failed"));
        }
        co_return co_await failed;
      }
      const WebSocketMessageType type = state.read_type;
      state.read_message_active = false;
      state.read_message_bytes = 0;
      state.read_utf8.reset();
      co_return WebSocketReadResult{0, type, true, std::nullopt};
    }
  }
}

void abandon_writer(detail::WebSocketWriteState &writer) noexcept {
  if (!writer.active)
    return;
  writer.active = false;
  writer.busy = false;
  ++writer.generation;
  if (writer.owner != nullptr)
    writer.owner->failed = true;
}

void retire_unstarted_writer(detail::WebSocketWriteState &writer,
                             detail::WebSocketWriteState *&state_handle,
                             std::uint64_t &handle_generation) noexcept {
  HYLE_FATAL_ASSERT(writer.active && writer.first_frame);
  writer.active = false;
  writer.busy = false;
  ++writer.generation;
  state_handle = nullptr;
  handle_generation = 0;
}

} // namespace

WebSocketConnection::~WebSocketConnection() {
  if (impl_ != nullptr) {
    if (impl_->writer.active)
      abandon_writer(impl_->writer);
    impl_->~Impl();
    allocator_.deallocate_n(impl_, 1);
  }
}

WebSocketConnection::WebSocketConnection(WebSocketConnection &&other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)), allocator_(other.allocator_) {
}

WebSocketConnection &
WebSocketConnection::operator=(WebSocketConnection &&other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      if (impl_->writer.active)
        abandon_writer(impl_->writer);
      impl_->~Impl();
      allocator_.deallocate_n(impl_, 1);
    }
    impl_ = std::exchange(other.impl_, nullptr);
    allocator_ = other.allocator_;
  }
  return *this;
}

bool WebSocketConnection::valid() const noexcept {
  return impl_ != nullptr && !impl_->failed && !impl_->received_close &&
         impl_->stream.valid();
}

bool WebSocketConnection::is_client() const noexcept {
  return impl_ != nullptr && impl_->role == WebSocketRole::kClient;
}

unsigned WebSocketConnection::owner_worker() const noexcept {
  return impl_ == nullptr ? 0 : impl_->stream.owner_worker();
}

container::StringView
WebSocketConnection::selected_subprotocol() const noexcept {
  return impl_ == nullptr ? container::StringView{}
                          : impl_->selected_protocol.text_view();
}

sched::Task<Result<WebSocketConnection>> WebSocketConnection::accept(
    mem::AllocatorRef allocator, HttpServerConnection &http,
    HttpIncomingRequest &&request, WebSocketServerOptions options,
    OperationOptions operation) noexcept {
  if (!allocator.valid()) {
    co_return ws_failure<WebSocketConnection>(
        Status::InvalidArgument("WebSocket allocator is invalid"));
  }
  container::StringView encoded_key;
  Status valid = validate_server_handshake(request.head, options, encoded_key);
  if (!valid.ok())
    co_return ws_failure<WebSocketConnection>(std::move(valid));

  Impl *impl = allocator.allocate_n<Impl>(1);
  if (impl == nullptr) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket server connection allocation failed"));
  }
  new (impl) Impl(allocator, WebSocketRole::kServer, options.limits,
                  options.control_context, options.control_callback);
  WebSocketConnection connection(allocator, impl);
  if (!impl->selected_protocol.assign(options.subprotocol)) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket selected-subprotocol allocation failed"));
  }

  const std::array<char, 28> accept_value = websocket_accept_value(encoded_key);
  std::array<HttpHeader, kMaxHandshakeHeaders> headers{};
  std::size_t count = 0;
  headers[count++] = HttpHeader{
      "Sec-WebSocket-Accept",
      container::StringView(accept_value.data(),
                            static_cast<std::uint32_t>(accept_value.size()))};
  if (!options.subprotocol.empty()) {
    headers[count++] =
        HttpHeader{"Sec-WebSocket-Protocol", options.subprotocol};
  }
  for (const HttpHeader &header : options.headers)
    headers[count++] = header;

  auto accepted = http.accept_upgrade(
      std::move(request), kWebSocketProtocol,
      std::span<const HttpHeader>(headers.data(), count), operation);
  if (!accepted.valid()) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket server handshake coroutine allocation failed"));
  }
  Result<HttpUpgradeStream> stream = co_await accepted;
  if (!stream)
    co_return ws_failure<WebSocketConnection>(std::move(stream).error());
  impl->stream = std::move(*stream);
  co_return std::move(connection);
}

sched::Task<Result<WebSocketConnection>> WebSocketConnection::connect(
    mem::AllocatorRef allocator, HttpTransport transport,
    WebSocketClientOptions options, OperationOptions operation) noexcept {
  if (!allocator.valid()) {
    co_return ws_failure<WebSocketConnection>(
        Status::InvalidArgument("WebSocket allocator is invalid"));
  }
  Status valid = validate_client_options(options);
  if (!valid.ok())
    co_return ws_failure<WebSocketConnection>(std::move(valid));

  std::array<std::byte, 60> entropy{};
  Status random = platform::secure_random(entropy);
  if (!random.ok())
    co_return ws_failure<WebSocketConnection>(std::move(random));

  std::array<char, 24> encoded_key{};
  const container::Base64Result encoded = container::base64_encode(
      std::span<const std::byte>(entropy.data(), 16), encoded_key,
      {container::Base64Alphabet::kStandard,
       container::Base64EncodePadding::kInclude});
  if (!encoded.ok() || encoded.output_count != encoded_key.size()) {
    co_return ws_failure<WebSocketConnection>(
        Status::Internal("WebSocket client key encoding failed"));
  }
  const container::StringView key_view(
      encoded_key.data(), static_cast<std::uint32_t>(encoded_key.size()));
  const std::array<char, 28> expected_accept = websocket_accept_value(key_view);

  ByteBuffer protocol_offer(allocator);
  valid = join_subprotocols(allocator, options.subprotocols, protocol_offer);
  if (!valid.ok())
    co_return ws_failure<WebSocketConnection>(std::move(valid));

  Impl *impl = allocator.allocate_n<Impl>(1);
  if (impl == nullptr) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket client connection allocation failed"));
  }
  new (impl) Impl(allocator, WebSocketRole::kClient, options.limits,
                  options.control_context, options.control_callback);
  WebSocketConnection connection(allocator, impl);
  impl->mask_keys.initialize(
      std::span<const std::byte, 44>(entropy.data() + 16, 44));

  Result<HttpClientConnection> created = HttpClientConnection::create(
      allocator, std::move(transport), options.http_limits);
  if (!created)
    co_return ws_failure<WebSocketConnection>(std::move(created).error());
  HttpClientConnection http = std::move(*created);

  std::array<HttpHeader, kMaxHandshakeHeaders> headers{};
  std::size_t count = 0;
  headers[count++] = HttpHeader{"Sec-WebSocket-Key", key_view};
  headers[count++] = HttpHeader{"Sec-WebSocket-Version", "13"};
  if (!options.origin.empty())
    headers[count++] = HttpHeader{"Origin", options.origin};
  if (!options.subprotocols.empty()) {
    headers[count++] =
        HttpHeader{"Sec-WebSocket-Protocol", protocol_offer.text_view()};
  }
  for (const HttpHeader &header : options.headers)
    headers[count++] = header;

  const HttpRequestHead request{
      .method = "GET",
      .target = options.target,
      .authority = options.authority,
      .headers = std::span<const HttpHeader>(headers.data(), count),
      .version = HttpVersion::kHttp11,
      .keep_alive = true,
      .expect_continue = false,
  };
  auto request_task =
      http.request_upgrade(request, kWebSocketProtocol, operation);
  if (!request_task.valid()) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket client handshake coroutine allocation failed"));
  }
  Result<HttpIncomingResponse> response = co_await request_task;
  if (!response)
    co_return ws_failure<WebSocketConnection>(std::move(response).error());

  container::StringView selected;
  valid = validate_client_response(response->head, expected_accept,
                                   options.subprotocols, selected);
  if (!valid.ok())
    co_return ws_failure<WebSocketConnection>(std::move(valid));
  if (!impl->selected_protocol.assign(selected)) {
    co_return ws_failure<WebSocketConnection>(Status::ResourceExhausted(
        "WebSocket selected-subprotocol allocation failed"));
  }
  Result<HttpUpgradeStream> stream =
      http.take_upgrade(std::move(*response), kWebSocketProtocol);
  if (!stream)
    co_return ws_failure<WebSocketConnection>(std::move(stream).error());
  impl->stream = std::move(*stream);
  co_return std::move(connection);
}

Result<WebSocketMessageWriter>
WebSocketConnection::start_message(WebSocketMessageType type) noexcept {
  if (impl_ == nullptr || impl_->failed || impl_->received_close ||
      impl_->sent_close || !impl_->stream.valid()) {
    return ws_failure<WebSocketMessageWriter>(
        Status::FailedPrecondition("WebSocket connection is not writable"));
  }
  if (type != WebSocketMessageType::kText &&
      type != WebSocketMessageType::kBinary) {
    return ws_failure<WebSocketMessageWriter>(
        Status::InvalidArgument("WebSocket message type is invalid"));
  }
  detail::WebSocketWriteState &writer = impl_->writer;
  if (writer.active) {
    return ws_failure<WebSocketMessageWriter>(Status::FailedPrecondition(
        "finish the active WebSocket message before starting another"));
  }
  ++writer.generation;
  writer.message_bytes = 0;
  writer.type = type;
  writer.utf8.reset();
  writer.active = true;
  writer.busy = false;
  writer.first_frame = true;
  return WebSocketMessageWriter(&writer, writer.generation, allocator_);
}

static sched::Task<Result<WebSocketReadResult>>
websocket_read(mem::AllocatorRef allocator, detail::WebSocketState *impl,
               std::span<std::byte> output,
               OperationOptions operation) noexcept {
  if (impl == nullptr || impl->failed || impl->received_close ||
      !impl->stream.valid()) {
    co_return ws_failure<WebSocketReadResult>(
        Status::FailedPrecondition("WebSocket connection is not readable"));
  }
  if (impl->read_busy || impl->close_busy) {
    co_return ws_failure<WebSocketReadResult>(
        Status::FailedPrecondition("only one WebSocket read may be in flight"));
  }
  BoolGuard busy(impl->read_busy);
  auto task = read_message_chunk(allocator, *impl, output, operation);
  if (!task.valid()) {
    impl->failed = true;
    co_return ws_failure<WebSocketReadResult>(Status::ResourceExhausted(
        "WebSocket message-read coroutine allocation failed"));
  }
  co_return co_await task;
}

sched::Task<Result<WebSocketReadResult>>
WebSocketConnection::read_some(std::span<std::byte> output,
                               OperationOptions operation) noexcept {
  return websocket_read(allocator_, impl_, output, operation);
}

sched::Task<Status> websocket_send_text(mem::AllocatorRef allocator,
                                        WebSocketConnection &connection,
                                        container::StringView text,
                                        OperationOptions operation) noexcept {
  (void)allocator;
  if (!container::utf8_validate(text))
    co_return Status::InvalidArgument("WebSocket text is not valid UTF-8");
  Result<WebSocketMessageWriter> started =
      connection.start_message(WebSocketMessageType::kText);
  if (!started)
    co_return std::move(started).error();
  WebSocketMessageWriter writer = std::move(*started);
  const auto bytes = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(text.data()), text.size());
  auto task = writer.write(bytes, true, operation);
  if (!task.valid()) {
    co_return Status::ResourceExhausted(
        "WebSocket text coroutine allocation failed");
  }
  co_return co_await task;
}

sched::Task<Status>
WebSocketConnection::send_text(container::StringView text,
                               OperationOptions operation) noexcept {
  return websocket_send_text(allocator_, *this, text, operation);
}

sched::Task<Status> websocket_send_binary(mem::AllocatorRef allocator,
                                          WebSocketConnection &connection,
                                          std::span<const std::byte> bytes,
                                          OperationOptions operation) noexcept {
  (void)allocator;
  Result<WebSocketMessageWriter> started =
      connection.start_message(WebSocketMessageType::kBinary);
  if (!started)
    co_return std::move(started).error();
  WebSocketMessageWriter writer = std::move(*started);
  auto task = writer.write(bytes, true, operation);
  if (!task.valid()) {
    co_return Status::ResourceExhausted(
        "WebSocket binary coroutine allocation failed");
  }
  co_return co_await task;
}

sched::Task<Status>
WebSocketConnection::send_binary(std::span<const std::byte> bytes,
                                 OperationOptions operation) noexcept {
  return websocket_send_binary(allocator_, *this, bytes, operation);
}

static sched::Task<Status> websocket_ping(mem::AllocatorRef allocator,
                                          detail::WebSocketState *impl,
                                          std::span<const std::byte> payload,
                                          OperationOptions operation) noexcept {
  if (impl == nullptr || impl->failed || impl->sent_close ||
      impl->received_close || !impl->stream.valid()) {
    co_return Status::FailedPrecondition(
        "WebSocket connection is not writable");
  }
  if (payload.size() > kMaxControlBytes)
    co_return Status::OutOfRange("WebSocket ping exceeds 125 bytes");
  if (impl->writer.active)
    co_return Status::FailedPrecondition(
        "a WebSocket message write is already active");
  auto task = write_frame(allocator, *impl, kOpcodePing, true, payload, false,
                          operation);
  if (!task.valid()) {
    co_return Status::ResourceExhausted(
        "WebSocket ping coroutine allocation failed");
  }
  co_return co_await task;
}

sched::Task<Status>
WebSocketConnection::ping(std::span<const std::byte> payload,
                          OperationOptions operation) noexcept {
  return websocket_ping(allocator_, impl_, payload, operation);
}

static sched::Task<Status> websocket_pong(mem::AllocatorRef allocator,
                                          detail::WebSocketState *impl,
                                          std::span<const std::byte> payload,
                                          OperationOptions operation) noexcept {
  if (impl == nullptr || impl->failed || impl->sent_close ||
      impl->received_close || !impl->stream.valid()) {
    co_return Status::FailedPrecondition(
        "WebSocket connection is not writable");
  }
  if (payload.size() > kMaxControlBytes)
    co_return Status::OutOfRange("WebSocket pong exceeds 125 bytes");
  if (impl->writer.active)
    co_return Status::FailedPrecondition(
        "a WebSocket message write is already active");
  auto task = write_frame(allocator, *impl, kOpcodePong, true, payload, false,
                          operation);
  if (!task.valid()) {
    co_return Status::ResourceExhausted(
        "WebSocket pong coroutine allocation failed");
  }
  co_return co_await task;
}

sched::Task<Status>
WebSocketConnection::pong(std::span<const std::byte> payload,
                          OperationOptions operation) noexcept {
  return websocket_pong(allocator_, impl_, payload, operation);
}

static sched::Task<Status>
websocket_close(mem::AllocatorRef allocator, detail::WebSocketState *impl,
                WebSocketClose close_value,
                OperationOptions operation) noexcept {
  if (impl == nullptr || impl->failed)
    co_return Status::FailedPrecondition("WebSocket connection is invalid");
  if (impl->received_close)
    co_return Status{};
  if (!impl->stream.valid())
    co_return Status::FailedPrecondition("WebSocket connection is invalid");
  if (impl->read_busy || impl->close_busy) {
    co_return Status::FailedPrecondition(
        "finish the active WebSocket read before close");
  }
  if (impl->writer.active) {
    co_return Status::FailedPrecondition(
        "finish the active WebSocket message before close");
  }
  if ((!close_value.code.has_value() && !close_value.reason.empty()) ||
      (close_value.code.has_value() && !valid_close_code(*close_value.code)) ||
      close_value.reason.size() > kMaxControlBytes - 2 ||
      !container::utf8_validate(close_value.reason)) {
    co_return Status::InvalidArgument("WebSocket Close value is invalid");
  }
  BoolGuard closing(impl->close_busy);
  BoolGuard reading(impl->read_busy);

  if (!impl->sent_close) {
    std::array<std::byte, kMaxControlBytes> payload{};
    std::size_t size = 0;
    if (close_value.code.has_value()) {
      payload[0] = static_cast<std::byte>(*close_value.code >> 8);
      payload[1] = static_cast<std::byte>(*close_value.code);
      size = 2;
      if (!close_value.reason.empty()) {
        std::memcpy(payload.data() + size, close_value.reason.data(),
                    close_value.reason.size());
        size += close_value.reason.size();
      }
    }
    auto send = write_frame(allocator, *impl, kOpcodeClose, true,
                            std::span<const std::byte>(payload.data(), size),
                            false, operation);
    if (!send.valid()) {
      impl->failed = true;
      co_return Status::ResourceExhausted(
          "WebSocket Close coroutine allocation failed");
    }
    Status status = co_await send;
    if (!status.ok())
      co_return status;
    impl->sent_close = true;
  }

  std::array<std::byte, 4096> discard{};
  while (!impl->received_close) {
    auto read = read_message_chunk(allocator, *impl, discard, operation);
    if (!read.valid()) {
      impl->failed = true;
      co_return Status::ResourceExhausted(
          "WebSocket Close-read coroutine allocation failed");
    }
    Result<WebSocketReadResult> result = co_await read;
    if (!result) {
      impl->failed = true;
      co_return std::move(result).error();
    }
  }
  co_return Status{};
}

sched::Task<Status>
WebSocketConnection::close(WebSocketClose close_value,
                           OperationOptions operation) noexcept {
  return websocket_close(allocator_, impl_, close_value, operation);
}

WebSocketMessageWriter::~WebSocketMessageWriter() { abandon_(); }

WebSocketMessageWriter::WebSocketMessageWriter(
    WebSocketMessageWriter &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      generation_(std::exchange(other.generation_, 0)),
      allocator_(other.allocator_) {}

WebSocketMessageWriter &
WebSocketMessageWriter::operator=(WebSocketMessageWriter &&other) noexcept {
  if (this != &other) {
    abandon_();
    state_ = std::exchange(other.state_, nullptr);
    generation_ = std::exchange(other.generation_, 0);
    allocator_ = other.allocator_;
  }
  return *this;
}

bool WebSocketMessageWriter::valid() const noexcept {
  return state_ != nullptr && state_->active && !state_->busy &&
         state_->generation == generation_ && state_->owner != nullptr &&
         !state_->owner->failed;
}

void WebSocketMessageWriter::abandon_() noexcept {
  if (state_ != nullptr && state_->generation == generation_)
    abandon_writer(*state_);
  state_ = nullptr;
  generation_ = 0;
}

static sched::Task<Status> websocket_writer_write(
    mem::AllocatorRef allocator, detail::WebSocketWriteState *&state_handle,
    std::uint64_t &handle_generation, std::span<const std::byte> bytes,
    bool end_message, OperationOptions operation) noexcept {
  detail::WebSocketWriteState *writer = state_handle;
  if (writer == nullptr || !writer->active ||
      writer->generation != handle_generation || writer->owner == nullptr ||
      writer->owner->failed) {
    co_return Status::FailedPrecondition(
        "WebSocket message writer is not active");
  }
  if (writer->busy) {
    co_return Status::FailedPrecondition(
        "only one WebSocket message write may be in flight");
  }
  if (bytes.empty() && !end_message) {
    co_return Status::InvalidArgument(
        "empty non-final WebSocket write makes no progress");
  }
  detail::WebSocketState &state = *writer->owner;
  if (state.sent_close || state.received_close || !state.stream.valid()) {
    co_return Status::FailedPrecondition(
        "WebSocket connection is closing or closed");
  }
  if (writer->message_bytes > state.limits.max_message_bytes ||
      bytes.size() > state.limits.max_message_bytes - writer->message_bytes) {
    if (writer->first_frame) {
      retire_unstarted_writer(*writer, state_handle, handle_generation);
    } else {
      state.failed = true;
    }
    co_return Status::OutOfRange(
        "outgoing WebSocket message exceeds max_message_bytes");
  }

  Utf8State next_utf8 = writer->utf8;
  if (writer->type == WebSocketMessageType::kText &&
      (!next_utf8.consume(bytes) || (end_message && !next_utf8.complete()))) {
    if (writer->first_frame) {
      retire_unstarted_writer(*writer, state_handle, handle_generation);
    } else {
      state.failed = true;
    }
    co_return Status::InvalidArgument(
        "outgoing WebSocket text is not valid complete UTF-8");
  }

  BoolGuard busy(writer->busy);
  std::size_t offset = 0;
  bool emitted = false;
  do {
    const std::size_t count =
        std::min(bytes.size() - offset, state.limits.write_fragment_bytes);
    const bool last = offset + count == bytes.size();
    const bool final = end_message && last;
    const std::uint8_t opcode =
        writer->first_frame
            ? (writer->type == WebSocketMessageType::kText ? kOpcodeText
                                                           : kOpcodeBinary)
            : kOpcodeContinuation;
    auto task = write_frame(allocator, state, opcode, final,
                            bytes.subspan(offset, count), true, operation);
    if (!task.valid()) {
      if (!emitted && writer->first_frame && !state.failed) {
        retire_unstarted_writer(*writer, state_handle, handle_generation);
      } else {
        state.failed = true;
      }
      co_return Status::ResourceExhausted(
          "WebSocket message-frame coroutine allocation failed");
    }
    Status status = co_await task;
    if (!status.ok()) {
      if (!emitted && writer->first_frame && !state.failed) {
        retire_unstarted_writer(*writer, state_handle, handle_generation);
      } else {
        state.failed = true;
      }
      co_return status;
    }
    emitted = true;
    writer->first_frame = false;
    offset += count;
  } while (offset != bytes.size());

  writer->message_bytes += bytes.size();
  writer->utf8 = next_utf8;
  if (end_message) {
    writer->active = false;
    state_handle = nullptr;
    handle_generation = 0;
  }
  co_return Status{};
}

sched::Task<Status>
WebSocketMessageWriter::write(std::span<const std::byte> bytes,
                              bool end_message,
                              OperationOptions operation) noexcept {
  return websocket_writer_write(allocator_, state_, generation_, bytes,
                                end_message, operation);
}

} // namespace hyle::io
