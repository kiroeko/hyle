#include "hyle/io/transfer.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/file.h"
#include "hyle/io/http.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/time/clock.h"

namespace hyle::io {
namespace {

using OwnedChars = container::SmallVector<char, 96, mem::AllocatorRef>;

enum class HttpValidatorKind : std::uint8_t {
  kNone,
  kStrongEtag,
  kLastModified,
};

struct UrlParts {
  bool secure = false;
  std::string_view authority{};
  std::string_view host{};
  std::string_view target{};
  std::uint16_t port = 0;
};

constexpr bool ascii_iequal(char lhs, char rhs) noexcept {
  const auto lower = [](char value) constexpr noexcept {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A'))
                                        : value;
  };
  return lower(lhs) == lower(rhs);
}

bool ascii_iequals(std::string_view lhs, std::string_view rhs) noexcept {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!ascii_iequal(lhs[i], rhs[i]))
      return false;
  }
  return true;
}

std::string_view trim_ows(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1);
  return value;
}

bool contains_nul_or_control(std::string_view value) noexcept {
  for (const unsigned char byte : value) {
    if (byte == 0 || byte == 0x7f || byte < 0x20)
      return true;
  }
  return false;
}

bool contains_url_forbidden(std::string_view value) noexcept {
  for (const unsigned char byte : value) {
    if (byte <= 0x20 || byte == 0x7f)
      return true;
  }
  return false;
}

Status parse_port(std::string_view text, std::uint16_t &output) noexcept {
  if (text.empty())
    return Status::InvalidArgument("URL port is empty");
  unsigned value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value == 0 || value > 65535) {
    return Status::InvalidArgument("URL port must be in [1, 65535]");
  }
  output = static_cast<std::uint16_t>(value);
  return {};
}

Status parse_url(std::string_view url, UrlParts &result) noexcept {
  result = {};
  if (url.empty())
    return Status::InvalidArgument("transfer URL is empty");
  if (contains_url_forbidden(url))
    return Status::InvalidArgument(
        "transfer URL contains whitespace or a control byte");
  if (url.find('#') != std::string_view::npos)
    return Status::InvalidArgument("transfer URL fragments are not supported");

  const std::size_t scheme_end = url.find("://");
  if (scheme_end == std::string_view::npos)
    return Status::InvalidArgument("transfer URL must be absolute HTTP(S)");
  const std::string_view scheme = url.substr(0, scheme_end);
  if (ascii_iequals(scheme, "http")) {
    result.secure = false;
    result.port = 80;
  } else if (ascii_iequals(scheme, "https")) {
    result.secure = true;
    result.port = 443;
  } else {
    return Status::InvalidArgument("transfer URL scheme must be http or https");
  }

  const std::size_t authority_begin = scheme_end + 3;
  const std::size_t authority_end = url.find_first_of("/?", authority_begin);
  const std::size_t authority_size =
      (authority_end == std::string_view::npos ? url.size() : authority_end) -
      authority_begin;
  result.authority = url.substr(authority_begin, authority_size);
  if (result.authority.empty())
    return Status::InvalidArgument("transfer URL host is empty");
  if (result.authority.find('@') != std::string_view::npos)
    return Status::InvalidArgument(
        "transfer URL credentials/userinfo are not supported");

  if (result.authority.front() == '[') {
    const std::size_t close = result.authority.find(']');
    if (close == std::string_view::npos || close == 1)
      return Status::InvalidArgument("transfer URL has an invalid IPv6 host");
    result.host = result.authority.substr(1, close - 1);
    const std::string_view remainder = result.authority.substr(close + 1);
    if (!remainder.empty()) {
      if (remainder.front() != ':')
        return Status::InvalidArgument("transfer URL has invalid authority");
      HYLE_RETURN_IF_ERROR(parse_port(remainder.substr(1), result.port));
    }
  } else {
    const std::size_t first_colon = result.authority.find(':');
    const std::size_t last_colon = result.authority.rfind(':');
    if (first_colon != std::string_view::npos && first_colon != last_colon) {
      return Status::InvalidArgument("IPv6 URL hosts must use square brackets");
    }
    if (last_colon == std::string_view::npos) {
      result.host = result.authority;
    } else {
      result.host = result.authority.substr(0, last_colon);
      HYLE_RETURN_IF_ERROR(
          parse_port(result.authority.substr(last_colon + 1), result.port));
    }
  }
  if (result.host.empty())
    return Status::InvalidArgument("transfer URL host is empty");

  if (authority_end == std::string_view::npos) {
    result.target = {};
  } else {
    result.target = url.substr(authority_end);
  }
  return {};
}

std::string_view chars_view(const OwnedChars &chars) noexcept {
  if (chars.empty())
    return {};
  return std::string_view(chars.data(), chars.size() - 1);
}

container::StringView public_view(const OwnedChars &chars) noexcept {
  if (chars.empty())
    return {};
  return container::StringView(
      chars.data(), static_cast<std::uint32_t>(chars.size() - 1), true);
}

bool assign_text(OwnedChars &destination, std::string_view source) noexcept {
  if (source.size() == (std::numeric_limits<std::size_t>::max)())
    return false;
  if (!destination.try_resize(source.size() + 1))
    return false;
  if (!source.empty())
    std::memcpy(destination.data(), source.data(), source.size());
  destination[source.size()] = '\0';
  return true;
}

bool append_text(OwnedChars &destination, std::string_view suffix) noexcept {
  const std::size_t prefix = destination.empty() ? 0 : destination.size() - 1;
  if (suffix.size() > (std::numeric_limits<std::size_t>::max)() - prefix - 1) {
    return false;
  }
  if (!destination.try_resize(prefix + suffix.size() + 1))
    return false;
  if (!suffix.empty())
    std::memcpy(destination.data() + prefix, suffix.data(), suffix.size());
  destination[prefix + suffix.size()] = '\0';
  return true;
}

Status derive_destination_directory(std::string_view destination,
                                    OwnedChars &directory) noexcept {
  if (destination.empty())
    return Status::InvalidArgument("transfer destination path is empty");
  if (contains_nul_or_control(destination))
    return Status::InvalidArgument(
        "transfer destination path contains a control byte");
  const std::size_t separator = destination.find_last_of("/\\");
  if (separator == destination.size() - 1)
    return Status::InvalidArgument(
        "transfer destination must name a file, not a directory");
#if HYLE_OS_WINDOWS
  if (destination.size() >= 2 && destination[1] == ':' &&
      (destination.size() == 2 ||
       (destination[2] != '/' && destination[2] != '\\'))) {
    return Status::InvalidArgument(
        "drive-relative transfer destinations are ambiguous");
  }
#endif
  std::string_view value = ".";
  if (separator != std::string_view::npos) {
    value = destination.substr(0, separator + 1);
  }
  if (!assign_text(directory, value))
    return Status::ResourceExhausted(
        "transfer destination-directory allocation failed");
  return {};
}

void remove_last_uri_segment(OwnedChars &output) noexcept {
  std::string_view value = chars_view(output);
  const std::size_t slash = value.rfind('/');
  const std::size_t new_size = slash == std::string_view::npos ? 0 : slash;
  output.try_resize(new_size + 1);
  output[new_size] = '\0';
}

bool append_normalized_uri_path(OwnedChars &output,
                                std::string_view path) noexcept {
  while (!path.empty()) {
    if (path.starts_with("../")) {
      path.remove_prefix(3);
    } else if (path.starts_with("./")) {
      path.remove_prefix(2);
    } else if (path.starts_with("/./")) {
      path.remove_prefix(2);
    } else if (path == "/.") {
      path = "/";
    } else if (path.starts_with("/../")) {
      path.remove_prefix(3);
      remove_last_uri_segment(output);
    } else if (path == "/..") {
      path = "/";
      remove_last_uri_segment(output);
    } else if (path == "." || path == "..") {
      path = {};
    } else {
      const std::size_t next =
          path.front() == '/' ? path.find('/', 1) : path.find('/');
      const std::size_t count =
          next == std::string_view::npos ? path.size() : next;
      if (!append_text(output, path.substr(0, count)))
        return false;
      path.remove_prefix(count);
    }
  }
  return true;
}

Status resolve_redirect(mem::AllocatorRef allocator, std::string_view current,
                        std::string_view location, bool allow_downgrade,
                        OwnedChars &output) noexcept {
  location = trim_ows(location);
  if (location.empty())
    return Status::DataLoss("HTTP redirect has an empty Location");
  if (contains_url_forbidden(location) ||
      location.find('#') != std::string_view::npos) {
    return Status::PermissionDenied(
        "HTTP redirect Location is not an admitted URL");
  }

  UrlParts base;
  HYLE_RETURN_IF_ERROR(parse_url(current, base));
  OwnedChars candidate(allocator);
  UrlParts absolute;
  Status absolute_status = parse_url(location, absolute);
  const std::size_t first_path_delimiter = location.find_first_of("/?");
  const std::size_t scheme_delimiter = location.find(':');
  const bool has_absolute_scheme =
      scheme_delimiter != std::string_view::npos &&
      (first_path_delimiter == std::string_view::npos ||
       scheme_delimiter < first_path_delimiter);
  if (absolute_status.ok()) {
    if (!assign_text(candidate, location))
      return Status::ResourceExhausted("HTTP redirect URL allocation failed");
  } else if (has_absolute_scheme) {
    return Status::PermissionDenied(
        "HTTP redirect scheme is not admitted by transfer");
  } else if (location.starts_with("//")) {
    if (!assign_text(candidate, base.secure ? "https:" : "http:") ||
        !append_text(candidate, location)) {
      return Status::ResourceExhausted("HTTP redirect URL allocation failed");
    }
  } else {
    const std::size_t scheme_end = current.find("://");
    const std::size_t authority_end =
        current.find_first_of("/?", scheme_end + 3);
    const std::string_view origin = current.substr(
        0, authority_end == std::string_view::npos ? current.size()
                                                   : authority_end);
    if (!assign_text(candidate, origin))
      return Status::ResourceExhausted("HTTP redirect URL allocation failed");

    std::string_view resolved_path;
    std::string_view query;
    OwnedChars merged_path(allocator);
    if (location.front() == '/') {
      const std::size_t query_pos = location.find('?');
      resolved_path = location.substr(0, query_pos);
      if (query_pos != std::string_view::npos)
        query = location.substr(query_pos);
    } else if (location.front() == '?') {
      std::string_view base_target = base.target;
      const std::size_t query_pos = base_target.find('?');
      resolved_path = base_target.substr(0, query_pos);
      if (resolved_path.empty())
        resolved_path = "/";
      query = location;
    } else {
      std::string_view base_target = base.target;
      const std::size_t base_query = base_target.find('?');
      base_target = base_target.substr(0, base_query);
      if (base_target.empty())
        base_target = "/";
      const std::size_t slash = base_target.rfind('/');
      const std::string_view base_directory =
          slash == std::string_view::npos ? std::string_view("/")
                                          : base_target.substr(0, slash + 1);
      const std::size_t query_pos = location.find('?');
      if (!assign_text(merged_path, base_directory) ||
          !append_text(merged_path, location.substr(0, query_pos))) {
        return Status::ResourceExhausted(
            "HTTP redirect path allocation failed");
      }
      resolved_path = chars_view(merged_path);
      if (query_pos != std::string_view::npos)
        query = location.substr(query_pos);
    }

    OwnedChars normalized(allocator);
    if (!assign_text(normalized, "") ||
        !append_normalized_uri_path(normalized, resolved_path) ||
        chars_view(normalized).empty() ||
        !append_text(candidate, chars_view(normalized)) ||
        (!query.empty() && !append_text(candidate, query))) {
      return Status::ResourceExhausted("HTTP redirect path allocation failed");
    }
  }

  UrlParts next;
  HYLE_RETURN_IF_ERROR(parse_url(chars_view(candidate), next));
  if (base.secure && !next.secure && !allow_downgrade) {
    return Status::PermissionDenied(
        "HTTPS to HTTP redirect is disabled by policy");
  }
  output = std::move(candidate);
  return {};
}

Status validate_options(const TransferOptions &options) noexcept {
  if (options.limits.buffer_bytes == 0)
    return Status::InvalidArgument("transfer buffer size must be nonzero");
  if (options.limits.max_http_head_bytes == 0)
    return Status::InvalidArgument("HTTP head limit must be nonzero");
  if (options.limits.max_http_header_count == 0)
    return Status::InvalidArgument("HTTP header-count limit must be nonzero");
  // A finite implementation guard, not a benchmark shape. Redirect chains
  // beyond this are almost certainly a loop or policy error.
  if (options.limits.max_redirects > 64)
    return Status::InvalidArgument("HTTP redirect limit must not exceed 64");
  return {};
}

Result<std::uint64_t> parse_decimal(std::string_view value,
                                    const char *field) noexcept {
  value = trim_ows(value);
  if (value.empty())
    return make_unexpected(Status::DataLoss(field));
  std::uint64_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    return make_unexpected(Status::DataLoss(field));
  return result;
}

struct ContentRange {
  bool valid = false;
  bool unsatisfied = false;
  std::uint64_t first = 0;
  std::uint64_t last = 0;
  std::optional<std::uint64_t> total{};
};

Result<ContentRange> parse_content_range(std::string_view value) noexcept {
  value = trim_ows(value);
  constexpr std::string_view kPrefix = "bytes ";
  if (value.size() < kPrefix.size() ||
      !ascii_iequals(value.substr(0, kPrefix.size()), kPrefix)) {
    return make_unexpected(
        Status::DataLoss("HTTP Content-Range unit is not bytes"));
  }
  value.remove_prefix(kPrefix.size());
  ContentRange result;
  if (value.starts_with("*/")) {
    auto total =
        parse_decimal(value.substr(2), "HTTP Content-Range total is invalid");
    if (!total.has_value())
      return make_unexpected(total.error());
    result.valid = true;
    result.unsatisfied = true;
    result.total = *total;
    return result;
  }

  const std::size_t dash = value.find('-');
  const std::size_t slash = value.find('/');
  if (dash == std::string_view::npos || slash == std::string_view::npos ||
      dash == 0 || slash <= dash + 1) {
    return make_unexpected(
        Status::DataLoss("HTTP Content-Range syntax is invalid"));
  }
  auto first = parse_decimal(value.substr(0, dash),
                             "HTTP Content-Range start is invalid");
  auto last = parse_decimal(value.substr(dash + 1, slash - dash - 1),
                            "HTTP Content-Range end is invalid");
  if (!first.has_value())
    return make_unexpected(first.error());
  if (!last.has_value())
    return make_unexpected(last.error());
  if (*last < *first)
    return make_unexpected(
        Status::DataLoss("HTTP Content-Range end precedes start"));
  result.first = *first;
  result.last = *last;
  const std::string_view total_text = value.substr(slash + 1);
  if (total_text != "*") {
    auto total =
        parse_decimal(total_text, "HTTP Content-Range total is invalid");
    if (!total.has_value())
      return make_unexpected(total.error());
    if (*total == 0 || result.last >= *total)
      return make_unexpected(
          Status::DataLoss("HTTP Content-Range exceeds total size"));
    result.total = *total;
  }
  result.valid = true;
  return result;
}

bool is_redirect_status(int status) noexcept {
  return status == 301 || status == 302 || status == 303 || status == 307 ||
         status == 308;
}

Status http_status_failure(int status) noexcept {
  if (status == 401)
    return Status::Unauthenticated("HTTP transfer was not authenticated");
  if (status == 403)
    return Status::PermissionDenied("HTTP transfer was forbidden");
  if (status == 404)
    return Status::NotFound("HTTP transfer source was not found");
  if (status == 408 || status == 425 || status == 502 || status == 503 ||
      status == 504) {
    return Status::Unavailable(
        "HTTP transfer source is temporarily unavailable");
  }
  if (status == 413 || status == 429)
    return Status::ResourceExhausted("HTTP transfer request was rejected");
  if (status >= 500)
    return Status::Unavailable("HTTP transfer server failed");
  return Status::FailedPrecondition(
      "HTTP transfer returned an unusable status");
}

Status check_budget(const OperationOptions &options) noexcept {
  if (options.cancel != nullptr && options.cancel->is_cancelled())
    return Status::Cancelled("transfer cancelled");
  if (options.deadline.has_value() &&
      time::monotonic_now() >= *options.deadline) {
    return Status::DeadlineExceeded("transfer deadline elapsed");
  }
  return {};
}

} // namespace

namespace detail {

struct TransferState {
  explicit TransferState(mem::AllocatorRef backing) noexcept
      : allocator(backing), source(backing), current_url(backing),
        destination(backing), directory(backing), temp_path(backing),
        validator(backing) {}

  mem::AllocatorRef allocator{};
  TransferSourceKind kind = TransferSourceKind::kLocalFile;
  TransferOptions options{};
  OwnedChars source;
  OwnedChars current_url;
  OwnedChars destination;
  OwnedChars directory;
  OwnedChars temp_path;
  OwnedChars validator;
  platform::FileHandle partial;
  platform::FileInfo local_identity{};
  std::optional<std::uint64_t> expected_bytes{};
  std::uint64_t partial_bytes = 0;
  std::uint64_t bytes_this_attempt = 0;
  HttpValidatorKind validator_kind = HttpValidatorKind::kNone;
  bool local_identity_valid = false;
  bool temp_exists = false;
  bool resume_attempt = false;
  bool resumed = false;
  bool restarted = false;
  bool resumable = true;
};

class TransferAccess {
public:
  static TransferState *release(TransferPlan &plan) noexcept {
    return std::exchange(plan.state_, nullptr);
  }
  static TransferState *get(const TransferPlan &plan) noexcept {
    return plan.state_;
  }
  static TransferState *get(const TransferCheckpoint &checkpoint) noexcept {
    return checkpoint.state_;
  }
  static TransferState *release(TransferCheckpoint &checkpoint) noexcept {
    return std::exchange(checkpoint.state_, nullptr);
  }
  static TransferPlan make_plan(TransferState *state) noexcept {
    return TransferPlan(state);
  }
  static TransferCheckpoint make_checkpoint(TransferState *state) noexcept {
    return TransferCheckpoint(state);
  }
};

} // namespace detail

namespace {

void destroy_state(detail::TransferState *state) noexcept {
  if (state == nullptr)
    return;
  if (state->partial.valid())
    state->partial.close().IgnoreError();
  if (state->temp_exists && !state->temp_path.empty())
    platform::remove_file(state->temp_path.data()).IgnoreError();
  const mem::AllocatorRef allocator = state->allocator;
  state->~TransferState();
  allocator.deallocate(state, sizeof(detail::TransferState),
                       alignof(detail::TransferState));
}

Result<detail::TransferState *>
allocate_state(mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid()) {
    return make_unexpected(
        Status::InvalidArgument("transfer plan requires a complete allocator"));
  }
  void *storage = allocator.allocate(sizeof(detail::TransferState),
                                     alignof(detail::TransferState));
  if (storage == nullptr) {
    return make_unexpected(
        Status::ResourceExhausted("transfer plan allocation failed"));
  }
  return new (storage) detail::TransferState(allocator);
}

Result<TransferPlan> make_plan(mem::AllocatorRef allocator,
                               TransferSourceKind kind, std::string_view source,
                               std::string_view destination,
                               TransferOptions options) noexcept {
  Status valid = validate_options(options);
  if (!valid.ok())
    return make_unexpected(std::move(valid));
  if (source.empty())
    return make_unexpected(Status::InvalidArgument(
        kind == TransferSourceKind::kLocalFile ? "transfer source path is empty"
                                               : "transfer URL is empty"));
  if (contains_nul_or_control(source))
    return make_unexpected(
        Status::InvalidArgument("transfer source contains a control byte"));

  UrlParts parsed;
  if (kind == TransferSourceKind::kHttp) {
    valid = parse_url(source, parsed);
    if (!valid.ok())
      return make_unexpected(std::move(valid));
  }

  auto allocated = allocate_state(allocator);
  if (!allocated.has_value())
    return make_unexpected(allocated.error());
  detail::TransferState *state = *allocated;
  state->kind = kind;
  state->options = options;
  if (!assign_text(state->source, source) ||
      !assign_text(state->destination, destination) ||
      (kind == TransferSourceKind::kHttp &&
       !assign_text(state->current_url, source))) {
    destroy_state(state);
    return make_unexpected(
        Status::ResourceExhausted("transfer configuration allocation failed"));
  }
  valid = derive_destination_directory(destination, state->directory);
  if (!valid.ok()) {
    destroy_state(state);
    return make_unexpected(std::move(valid));
  }
  return detail::TransferAccess::make_plan(state);
}

Status close_partial(detail::TransferState &state) noexcept {
  if (!state.partial.valid())
    return {};
  Status status = state.partial.close();
  if (!status.ok())
    state.resumable = false;
  return status;
}

template <class Fn>
Status run_blocking_status(mem::AllocatorRef allocator, IoContext &io,
                           Fn function) noexcept {
  auto handle =
      sched::spawn_blocking(detail::IoContextAccess::blocking_pool(io),
                            allocator, std::move(function));
  auto joined = handle.join();
  if (!joined.has_value())
    return std::move(joined).error();
  return std::move(*joined);
}

Status remove_temp(mem::AllocatorRef allocator, IoContext &io,
                   detail::TransferState &state) noexcept {
  Status close = close_partial(state);
  if (!close.ok())
    return close;
  if (!state.temp_exists)
    return {};
  const char *path = state.temp_path.data();
  Status removed = run_blocking_status(
      allocator, io, [path]() noexcept { return platform::remove_file(path); });
  if (removed.IsNotFound())
    removed = {};
  if (removed.ok())
    state.temp_exists = false;
  return removed;
}

Status create_temp(mem::AllocatorRef allocator, IoContext &io,
                   detail::TransferState &state) noexcept {
  constexpr std::string_view kPrefix = "hyle-transfer-";
  const std::size_t directory_size = chars_view(state.directory).size();
  if (directory_size >
      (std::numeric_limits<std::size_t>::max)() - kPrefix.size() - 40) {
    return Status::OutOfRange("temporary-file path is too long");
  }
  const std::size_t capacity = directory_size + kPrefix.size() + 40;
  if (!state.temp_path.try_resize(capacity))
    return Status::ResourceExhausted("temporary-file path allocation failed");

  const char *directory = state.directory.data();
  char *output = state.temp_path.data();
  auto handle = sched::spawn_blocking(
      detail::IoContextAccess::blocking_pool(io), allocator,
      [directory, output, capacity]() noexcept {
        return platform::create_temp_file(directory, "hyle-transfer-", output,
                                          capacity);
      });
  auto joined = handle.join();
  if (!joined.has_value())
    return std::move(joined).error();
  auto created = std::move(*joined);
  if (!created.has_value())
    return std::move(created).error();
  if (created->path_size >= capacity) {
    created->file.close().IgnoreError();
    platform::remove_file(output).IgnoreError();
    return Status::Internal("temporary-file path contract was truncated");
  }
  state.partial = std::move(created->file);
  state.temp_exists = true;
  state.partial_bytes = 0;
  state.temp_path.try_resize(created->path_size + 1);
  return {};
}

Status truncate_partial(mem::AllocatorRef allocator, IoContext &io,
                        detail::TransferState &state,
                        File &destination) noexcept {
  state.partial = destination.release();
  if (!state.partial.valid()) {
    state.resumable = false;
    return Status::Internal("transfer lost its temporary-file owner");
  }
  platform::FileHandle *handle = &state.partial;
  Status resized = run_blocking_status(
      allocator, io, [handle]() noexcept { return handle->resize(0); });
  if (!resized.ok())
    return resized;
  auto adopted =
      File::adopt(io, destination.owner_worker(), std::move(state.partial));
  if (!adopted.has_value()) {
    state.resumable = false;
    return std::move(adopted).error();
  }
  destination = std::move(*adopted);
  state.partial_bytes = 0;
  state.expected_bytes.reset();
  state.validator.clear();
  state.validator_kind = HttpValidatorKind::kNone;
  state.resumed = false;
  state.restarted = true;
  return {};
}

Status notify_progress(const TransferExecution &execution,
                       const detail::TransferState &state,
                       TransferPhase phase) noexcept {
  return execution.progress.notify(TransferProgress{
      .phase = phase,
      .artifact_bytes = state.partial_bytes,
      .bytes_this_attempt = state.bytes_this_attempt,
      .total_bytes = state.expected_bytes,
      .resumed = state.resumed,
      .restarted = state.restarted,
  });
}

sched::Task<Status> hash_existing_prefix(mem::AllocatorRef allocator,
                                         detail::TransferState *state,
                                         File *destination,
                                         std::span<std::byte> buffer,
                                         TransferVerifier verifier,
                                         OperationOptions operation) noexcept {
  // The allocator-first coroutine convention owns this frame even though the
  // body needs no additional allocation.
  (void)allocator;
  if (!verifier.valid())
    co_return Status{};
  Status status = verifier.reset();
  if (!status.ok())
    co_return status;
  std::uint64_t offset = 0;
  while (offset < state->partial_bytes) {
    status = check_budget(operation);
    if (!status.ok())
      co_return status;
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), state->partial_bytes - offset));
    TransferResult read = co_await destination->read_some_at(
        offset, buffer.first(count), operation);
    if (read.bytes != 0) {
      status = verifier.update(
          std::span<const std::byte>(buffer.data(), read.bytes));
      if (!status.ok())
        co_return status;
      offset += read.bytes;
    }
    if (!read.status.ok())
      co_return std::move(read.status);
    if (read.eof || read.bytes == 0) {
      co_return Status::DataLoss(
          "retained transfer partial is shorter than its checkpoint");
    }
  }
  co_return Status{};
}

sched::Task<Result<bool>> local_prefix_matches(
    mem::AllocatorRef allocator, detail::TransferState *state, File *source,
    File *destination, std::span<std::byte> source_buffer,
    std::span<std::byte> partial_buffer, TransferVerifier verifier,
    OperationOptions operation) noexcept {
  if (verifier.valid()) {
    Status reset = verifier.reset();
    if (!reset.ok())
      co_return make_unexpected(std::move(reset));
  }
  std::uint64_t offset = 0;
  while (offset < state->partial_bytes) {
    Status budget = check_budget(operation);
    if (!budget.ok())
      co_return make_unexpected(std::move(budget));
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(
        source_buffer.size(), state->partial_bytes - offset));
    TransferResult source_read = co_await read_exact_at(
        allocator, *source, offset, source_buffer.first(count), operation);
    if (!source_read.status.ok()) {
      if (source_read.status.IsDataLoss())
        co_return false;
      co_return make_unexpected(std::move(source_read.status));
    }
    TransferResult partial_read =
        co_await read_exact_at(allocator, *destination, offset,
                               partial_buffer.first(count), operation);
    if (!partial_read.status.ok())
      co_return make_unexpected(std::move(partial_read.status));
    if (source_read.bytes != count || partial_read.bytes != count)
      co_return false;
    if (std::memcmp(source_buffer.data(), partial_buffer.data(), count) != 0)
      co_return false;
    if (verifier.valid()) {
      Status updated = verifier.update(
          std::span<const std::byte>(source_buffer.data(), count));
      if (!updated.ok())
        co_return make_unexpected(std::move(updated));
    }
    offset += count;
  }
  co_return true;
}

bool same_file_identity(const platform::FileInfo &lhs,
                        const platform::FileInfo &rhs) noexcept {
  return lhs.type == rhs.type && lhs.size == rhs.size &&
         lhs.modified_time == rhs.modified_time;
}

sched::Task<Status> run_local_transfer(
    mem::AllocatorRef allocator, IoContext *io, detail::TransferState *state,
    File *destination, std::span<std::byte> buffer,
    std::span<std::byte> partial_buffer, TransferExecution execution) noexcept {
  auto opened =
      co_await open_file(*io, execution.owner_worker, chars_view(state->source),
                         FileOpenOptions{.access = FileAccess::kRead});
  if (!opened.has_value())
    co_return std::move(opened).error();
  File source = std::move(*opened);
  auto info = co_await source.info();
  if (!info.has_value())
    co_return std::move(info).error();
  if (!info->is_regular())
    co_return Status::FailedPrecondition(
        "local transfer source must be a regular file");

  if (state->resume_attempt && state->local_identity_valid &&
      !same_file_identity(state->local_identity, *info)) {
    if (state->options.resume_mismatch == TransferResumeMismatchPolicy::kFail) {
      co_return Status::FailedPrecondition(
          "local transfer source changed since the checkpoint");
    }
    Status reset = truncate_partial(allocator, *io, *state, *destination);
    if (!reset.ok())
      co_return reset;
  }
  if (!state->local_identity_valid || state->restarted) {
    state->local_identity = *info;
    state->local_identity_valid = true;
  }
  // Some regular virtual files (notably procfs/sysfs on Linux) report a zero
  // metadata size while still yielding bytes. Treat only a non-zero size as a
  // known extent; a true empty file is resolved by the first clean EOF.
#if HYLE_OS_WINDOWS
  // Windows regular-file metadata has an authoritative end-of-file and its
  // overlapped read surface reports an empty read as ERROR_HANDLE_EOF.
  const bool size_unknown = false;
#else
  const bool size_unknown = info->size == 0;
#endif
  if (size_unknown)
    state->expected_bytes.reset();
  else
    state->expected_bytes = info->size;
  if (!size_unknown && info->size > state->options.limits.max_bytes)
    co_return Status::OutOfRange("local transfer source exceeds max_bytes");
  if (state->partial_bytes > state->options.limits.max_bytes)
    co_return Status::OutOfRange("retained transfer partial exceeds max_bytes");
  if (!size_unknown && state->partial_bytes > info->size)
    co_return Status::DataLoss(
        "retained transfer partial exceeds the local source");
  state->resumed = state->resume_attempt && state->partial_bytes != 0;

  Status status;
  if (state->partial_bytes != 0) {
    auto matches = co_await local_prefix_matches(
        allocator, state, &source, destination, buffer, partial_buffer,
        execution.verifier, execution.operation);
    if (!matches.has_value())
      co_return std::move(matches).error();
    if (!*matches) {
      if (state->options.resume_mismatch ==
          TransferResumeMismatchPolicy::kFail) {
        co_return Status::FailedPrecondition(
            "local transfer source prefix differs from the checkpoint");
      }
      status = truncate_partial(allocator, *io, *state, *destination);
      if (!status.ok())
        co_return status;
      state->local_identity = *info;
      state->local_identity_valid = true;
      if (!size_unknown)
        state->expected_bytes = info->size;
      if (execution.verifier.valid()) {
        status = execution.verifier.reset();
        if (!status.ok())
          co_return status;
      }
    }
  } else if (execution.verifier.valid()) {
    status = execution.verifier.reset();
    if (!status.ok())
      co_return status;
  }
  status = notify_progress(execution, *state, TransferPhase::kPreparing);
  if (!status.ok())
    co_return status;

  while (size_unknown || state->partial_bytes < info->size) {
    status = check_budget(execution.operation);
    if (!status.ok())
      co_return status;
    std::size_t count = 0;
    if (size_unknown) {
      const std::uint64_t remaining =
          state->options.limits.max_bytes - state->partial_bytes;
      count = buffer.size();
      if (remaining < count) {
        // Probe one byte beyond the admitted budget so an exact-size source
        // can still succeed while an oversized source is rejected without
        // publishing or writing beyond max_bytes.
        count = static_cast<std::size_t>(remaining) + 1;
      }
    } else {
      count = static_cast<std::size_t>(std::min<std::uint64_t>(
          buffer.size(), info->size - state->partial_bytes));
    }
    TransferResult read = co_await source.read_some_at(
        state->partial_bytes, buffer.first(count), execution.operation);
    std::size_t admitted = read.bytes;
    bool exceeds_limit = false;
    if (size_unknown) {
      const std::uint64_t remaining =
          state->options.limits.max_bytes - state->partial_bytes;
      if (admitted > remaining) {
        admitted = static_cast<std::size_t>(remaining);
        exceeds_limit = true;
      }
    }
    if (admitted != 0) {
      TransferResult written = co_await write_all_at(
          allocator, *destination, state->partial_bytes,
          std::span<const std::byte>(buffer.data(), admitted),
          execution.operation);
      if (written.bytes != 0) {
        if (execution.verifier.valid()) {
          status = execution.verifier.update(
              std::span<const std::byte>(buffer.data(), written.bytes));
          if (!status.ok())
            co_return status;
        }
        state->partial_bytes += written.bytes;
        state->bytes_this_attempt += written.bytes;
        status =
            notify_progress(execution, *state, TransferPhase::kTransferring);
        if (!status.ok())
          co_return status;
      }
      if (!written.status.ok())
        co_return std::move(written.status);
      if (written.bytes != admitted)
        co_return Status::DataLoss(
            "temporary file write completed without full progress");
    }
    if (exceeds_limit)
      co_return Status::OutOfRange("local transfer source exceeds max_bytes");
    if (!read.status.ok())
      co_return std::move(read.status);
    if (read.eof || read.bytes == 0) {
      if (!size_unknown && state->partial_bytes != info->size) {
        co_return Status::DataLoss(
            "local transfer source ended before its reported size");
      }
      if (size_unknown)
        state->expected_bytes = state->partial_bytes;
      break;
    }
  }

  auto final_info = co_await source.info();
  if (!final_info.has_value())
    co_return std::move(final_info).error();
  if (!same_file_identity(*info, *final_info))
    co_return Status::Aborted("local transfer source changed while copying");
  Status close = co_await source.close();
  co_return close;
}

std::string_view std_view(container::StringView value) noexcept {
  return std::string_view(value.data(), value.size());
}

std::optional<std::string_view>
header_value(HttpHeadersView headers, container::StringView name) noexcept {
  auto value = headers.find(name);
  if (!value.has_value())
    return std::nullopt;
  return trim_ows(std_view(*value));
}

Status capture_http_validator(detail::TransferState &state,
                              HttpHeadersView headers) noexcept {
  state.validator.clear();
  state.validator_kind = HttpValidatorKind::kNone;
  if (auto etag = header_value(headers, "etag");
      etag.has_value() && !etag->empty()) {
    const bool weak =
        etag->size() >= 2 && ascii_iequal((*etag)[0], 'w') && (*etag)[1] == '/';
    if (!weak) {
      if (!assign_text(state.validator, *etag))
        return Status::ResourceExhausted(
            "HTTP ETag checkpoint allocation failed");
      state.validator_kind = HttpValidatorKind::kStrongEtag;
      return {};
    }
  }
  if (auto modified = header_value(headers, "last-modified");
      modified.has_value() && !modified->empty()) {
    if (!assign_text(state.validator, *modified))
      return Status::ResourceExhausted(
          "HTTP Last-Modified checkpoint allocation failed");
    state.validator_kind = HttpValidatorKind::kLastModified;
  }
  return {};
}

Status validate_http_validator(const detail::TransferState &state,
                               HttpHeadersView headers) noexcept {
  std::optional<std::string_view> observed;
  if (state.validator_kind == HttpValidatorKind::kStrongEtag)
    observed = header_value(headers, "etag");
  else if (state.validator_kind == HttpValidatorKind::kLastModified)
    observed = header_value(headers, "last-modified");
  else
    return {};
  if (!observed.has_value() || observed->empty() ||
      *observed != chars_view(state.validator)) {
    return Status::DataLoss(
        "HTTP range response omitted or changed the checkpoint validator");
  }
  return {};
}

Result<std::optional<std::uint64_t>>
content_length(HttpHeadersView headers) noexcept {
  auto value = header_value(headers, "content-length");
  if (!value.has_value())
    return std::optional<std::uint64_t>{};
  auto parsed = parse_decimal(*value, "HTTP Content-Length is invalid");
  if (!parsed.has_value())
    return make_unexpected(parsed.error());
  return std::optional<std::uint64_t>(*parsed);
}

Status build_request_target(mem::AllocatorRef allocator, const UrlParts &parts,
                            OwnedChars &output) noexcept {
  (void)allocator;
  if (parts.target.empty()) {
    if (!assign_text(output, "/"))
      return Status::ResourceExhausted("HTTP request-target allocation failed");
  } else if (parts.target.front() == '?') {
    if (!assign_text(output, "/") || !append_text(output, parts.target))
      return Status::ResourceExhausted("HTTP request-target allocation failed");
  } else if (!assign_text(output, parts.target)) {
    return Status::ResourceExhausted("HTTP request-target allocation failed");
  }
  return {};
}

Result<container::StringView> make_range_value(std::uint64_t offset,
                                               char (&storage)[64]) noexcept {
  constexpr std::string_view prefix = "bytes=";
  std::memcpy(storage, prefix.data(), prefix.size());
  auto converted = std::to_chars(storage + prefix.size(),
                                 storage + sizeof(storage) - 1, offset, 10);
  if (converted.ec != std::errc{})
    return make_unexpected(Status::Internal("HTTP Range formatting failed"));
  *converted.ptr++ = '-';
  *converted.ptr = '\0';
  return container::StringView(
      storage, static_cast<std::uint32_t>(converted.ptr - storage), true);
}

sched::Task<Status> run_http_transfer(mem::AllocatorRef allocator,
                                      IoContext *io,
                                      detail::TransferState *state,
                                      File *destination,
                                      std::span<std::byte> prefix_buffer,
                                      TransferExecution execution) noexcept {
  // Each execution starts from the caller's stable source URL. A redirect can
  // be short-lived (for example, a signed CDN URL), so a retained checkpoint
  // must reacquire the redirect chain rather than persisting the last hop.
  if (!assign_text(state->current_url, chars_view(state->source)))
    co_return Status::ResourceExhausted(
        "HTTP transfer URL reset allocation failed");
  if (state->partial_bytes > state->options.limits.max_bytes)
    co_return Status::OutOfRange(
        "retained HTTP partial exceeds transfer max_bytes");

  if (state->resume_attempt && state->partial_bytes != 0 &&
      state->validator_kind == HttpValidatorKind::kNone) {
    if (state->options.resume_mismatch == TransferResumeMismatchPolicy::kFail) {
      co_return Status::FailedPrecondition(
          "HTTP checkpoint has no safe Range validator");
    }
    Status reset = truncate_partial(allocator, *io, *state, *destination);
    if (!reset.ok())
      co_return reset;
  }

  unsigned redirects = 0;
  bool range_restart_attempted = state->restarted;
  const auto restart_range_or_fail =
      [&](const char *failure) noexcept -> Status {
    if (state->options.resume_mismatch !=
            TransferResumeMismatchPolicy::kRestart ||
        range_restart_attempted) {
      return Status::FailedPrecondition(failure);
    }
    Status reset = truncate_partial(allocator, *io, *state, *destination);
    if (reset.ok())
      range_restart_attempted = true;
    return reset;
  };
#if defined(HYLE_IO_TLS)
  TlsContext owned_tls_context;
#endif
  for (;;) {
    Status status = check_budget(execution.operation);
    if (!status.ok())
      co_return status;
    UrlParts parts;
    status = parse_url(chars_view(state->current_url), parts);
    if (!status.ok())
      co_return status;

    TcpConnectOptions connect_options;
    connect_options.operation = execution.operation;
    auto connected =
        co_await connect_tcp(allocator, *io, execution.owner_worker, parts.host,
                             parts.port, connect_options);
    if (!connected.has_value()) {
      co_return std::move(connected).error();
    }

    HttpTransport transport;
    if (parts.secure) {
#if defined(HYLE_IO_TLS)
      const TlsContext *tls_context = execution.tls_context;
      if (tls_context == nullptr && !owned_tls_context.valid()) {
        const container::StringView protocols[] = {"http/1.1"};
        TlsClientOptions tls_options;
        tls_options.connection_allocator = allocator;
        tls_options.alpn_protocols = protocols;
        auto context = TlsContext::make_client(tls_options);
        if (!context.has_value())
          co_return std::move(context).error();
        owned_tls_context = std::move(*context);
      }
      if (tls_context == nullptr)
        tls_context = &owned_tls_context;
      TlsConnectOptions tls_connect;
      tls_connect.server_name = container::StringView(parts.host);
      auto encrypted =
          TlsStream::connect(std::move(*connected), *tls_context, tls_connect);
      if (!encrypted.has_value())
        co_return std::move(encrypted).error();
      status =
          co_await tls_handshake(allocator, *encrypted, execution.operation);
      if (!status.ok())
        co_return status;
      const container::StringView protocol = encrypted->negotiated_protocol();
      if (!protocol.empty() && protocol != "http/1.1") {
        co_return Status::FailedPrecondition(
            "HTTPS transfer TLS context negotiated a protocol other than "
            "HTTP/1.1");
      }
      transport = HttpTransport(std::move(*encrypted));
#else
      co_return Status::Unimplemented(
          "HTTPS transfer requires an IO build with TLS enabled");
#endif
    } else {
      transport = HttpTransport(std::move(*connected));
    }

    HttpLimits limits;
    limits.max_head_bytes = state->options.limits.max_http_head_bytes;
    limits.max_header_count = state->options.limits.max_http_header_count;
    limits.max_body_bytes = state->options.limits.max_bytes;
    limits.read_chunk_bytes = state->options.limits.buffer_bytes;
    auto made =
        HttpClientConnection::create(allocator, std::move(transport), limits);
    if (!made.has_value())
      co_return std::move(made).error();
    HttpClientConnection connection = std::move(*made);

    OwnedChars request_target(allocator);
    status = build_request_target(allocator, parts, request_target);
    if (!status.ok())
      co_return status;

    HttpHeader headers[3] = {
        {"Accept-Encoding", "identity"},
        {},
        {},
    };
    std::size_t header_count = 1;
    const bool requested_range =
        state->partial_bytes != 0 &&
        state->validator_kind != HttpValidatorKind::kNone;
    char range_storage[64]{};
    if (requested_range) {
      auto range = make_range_value(state->partial_bytes, range_storage);
      if (!range.has_value())
        co_return std::move(range).error();
      headers[header_count++] = {"Range", *range};
      headers[header_count++] = {"If-Range", public_view(state->validator)};
    }

    HttpRequestHead request;
    request.method = "GET";
    request.target = public_view(request_target);
    request.authority = container::StringView(parts.authority);
    request.headers = std::span<const HttpHeader>(headers, header_count);
    request.keep_alive = false;
    auto received = co_await connection.request(
        request, std::span<const std::byte>{}, execution.operation);
    if (!received.has_value())
      co_return std::move(received).error();
    HttpIncomingResponse response = std::move(*received);
    const int response_status = response.head.status();
    const HttpHeadersView response_headers = response.head.headers();
    const std::uint64_t response_start = state->partial_bytes;
    std::optional<std::uint64_t> response_extent{};

    if (is_redirect_status(response_status)) {
      if (redirects >= state->options.limits.max_redirects)
        co_return Status::OutOfRange("HTTP redirect limit exceeded");
      auto location = header_value(response_headers, "location");
      if (!location.has_value())
        co_return Status::DataLoss("HTTP redirect omitted Location");
      OwnedChars resolved(allocator);
      status = resolve_redirect(
          allocator, chars_view(state->current_url), *location,
          state->options.allow_https_to_http_redirect, resolved);
      if (!status.ok())
        co_return status;
      state->current_url = std::move(resolved);
      ++redirects;
      continue;
    }

    if (auto encoding = header_value(response_headers, "content-encoding");
        encoding.has_value() && !encoding->empty() &&
        !ascii_iequals(*encoding, "identity")) {
      co_return Status::DataLoss(
          "HTTP transfer requires an identity-encoded representation");
    }
    auto length = content_length(response_headers);
    if (!length.has_value())
      co_return std::move(length).error();

    if (response_status == 416) {
      if (!requested_range)
        co_return http_status_failure(response_status);
      auto range_header = header_value(response_headers, "content-range");
      if (!range_header.has_value())
        co_return Status::DataLoss("HTTP 416 response omitted Content-Range");
      auto range = parse_content_range(*range_header);
      if (!range.has_value())
        co_return std::move(range).error();
      if (!range->unsatisfied || !range->total.has_value())
        co_return Status::DataLoss(
            "HTTP 416 response has an invalid Content-Range");
      if (*range->total == state->partial_bytes) {
        if (state->expected_bytes.has_value() &&
            *state->expected_bytes != *range->total) {
          status = restart_range_or_fail(
              "HTTP 416 total differs from the checkpoint");
          if (!status.ok())
            co_return status;
          continue;
        }
        status = validate_http_validator(*state, response_headers);
        if (!status.ok()) {
          status = restart_range_or_fail(
              "HTTP 416 validator differs from the checkpoint");
          if (!status.ok())
            co_return status;
          continue;
        }
        state->expected_bytes = *range->total;
        state->resumed = state->partial_bytes != 0;
        status = co_await hash_existing_prefix(
            allocator, state, destination, prefix_buffer, execution.verifier,
            execution.operation);
        if (!status.ok())
          co_return status;
        status = notify_progress(execution, *state, TransferPhase::kPreparing);
        co_return status;
      }
      status = restart_range_or_fail(
          "HTTP checkpoint offset is not valid for the representation");
      if (!status.ok())
        co_return status;
      continue;
    }

    if (response_status == 206) {
      if (!requested_range)
        co_return Status::DataLoss(
            "HTTP source returned unsolicited partial content");
      auto range_header = header_value(response_headers, "content-range");
      if (!range_header.has_value())
        co_return Status::DataLoss("HTTP 206 response omitted Content-Range");
      auto range = parse_content_range(*range_header);
      if (!range.has_value())
        co_return std::move(range).error();
      if (range->unsatisfied || range->first != state->partial_bytes)
        co_return Status::DataLoss(
            "HTTP range response does not begin at the checkpoint offset");
      if (range->last == (std::numeric_limits<std::uint64_t>::max)())
        co_return Status::DataLoss("HTTP Content-Range length overflows");
      const std::uint64_t response_bytes = range->last - range->first + 1;
      response_extent = response_bytes;
      if (length->has_value() && **length != response_bytes)
        co_return Status::DataLoss(
            "HTTP range Content-Length disagrees with Content-Range");
      if (state->expected_bytes.has_value() && range->total.has_value() &&
          *state->expected_bytes != *range->total) {
        status = restart_range_or_fail(
            "HTTP range total differs from the checkpoint");
        if (!status.ok())
          co_return status;
        continue;
      }
      if (range->total.has_value())
        state->expected_bytes = *range->total;
      status = validate_http_validator(*state, response_headers);
      if (!status.ok()) {
        status = restart_range_or_fail(
            "HTTP range validator differs from the checkpoint");
        if (!status.ok())
          co_return status;
        continue;
      }
      state->resumed = true;
    } else if (response_status == 200) {
      if (requested_range) {
        if (state->options.resume_mismatch ==
            TransferResumeMismatchPolicy::kFail) {
          co_return Status::FailedPrecondition(
              "HTTP source ignored Range for this checkpoint");
        }
        status = truncate_partial(allocator, *io, *state, *destination);
        if (!status.ok())
          co_return status;
      }
      state->expected_bytes = *length;
      status = capture_http_validator(*state, response_headers);
      if (!status.ok())
        co_return status;
    } else {
      co_return http_status_failure(response_status);
    }

    if (state->expected_bytes.has_value() &&
        *state->expected_bytes > state->options.limits.max_bytes) {
      co_return Status::OutOfRange("HTTP artifact exceeds max_bytes");
    }
    status = co_await hash_existing_prefix(allocator, state, destination,
                                           prefix_buffer, execution.verifier,
                                           execution.operation);
    if (!status.ok())
      co_return status;
    status = notify_progress(execution, *state, TransferPhase::kPreparing);
    if (!status.ok())
      co_return status;

    for (;;) {
      status = check_budget(execution.operation);
      if (!status.ok())
        co_return status;
      const std::uint64_t remaining_budget =
          state->options.limits.max_bytes - state->partial_bytes;
      std::size_t request_bytes = state->options.limits.buffer_bytes;
      if (remaining_budget < request_bytes) {
        request_bytes =
            remaining_budget >= (std::numeric_limits<std::size_t>::max)() - 1
                ? request_bytes
                : static_cast<std::size_t>(remaining_budget + 1);
      }
      if (request_bytes == 0)
        request_bytes = 1;
      auto next =
          co_await response.body.read(request_bytes, execution.operation);
      if (!next.has_value())
        co_return std::move(next).error();
      const std::span<const std::byte> bytes = next->bytes;
      const std::size_t admitted = static_cast<std::size_t>(
          std::min<std::uint64_t>(bytes.size(), remaining_budget));
      if (admitted != 0) {
        TransferResult written =
            co_await write_all_at(allocator, *destination, state->partial_bytes,
                                  bytes.first(admitted), execution.operation);
        if (written.bytes != 0) {
          if (execution.verifier.valid()) {
            status = execution.verifier.update(bytes.first(written.bytes));
            if (!status.ok())
              co_return status;
          }
          state->partial_bytes += written.bytes;
          state->bytes_this_attempt += written.bytes;
          status =
              notify_progress(execution, *state, TransferPhase::kTransferring);
          if (!status.ok())
            co_return status;
        }
        if (!written.status.ok())
          co_return std::move(written.status);
        if (written.bytes != admitted)
          co_return Status::DataLoss(
              "temporary file write completed without full progress");
      }
      if (admitted != bytes.size())
        co_return Status::OutOfRange("HTTP artifact exceeds max_bytes");
      if (next->end_stream)
        break;
      if (bytes.empty())
        co_return Status::DataLoss(
            "HTTP body reader made no progress before end-of-stream");
    }

    if (response_extent.has_value() &&
        state->partial_bytes - response_start != *response_extent) {
      co_return Status::DataLoss(
          "HTTP range body size differs from Content-Range");
    }
    if (state->expected_bytes.has_value() &&
        state->partial_bytes != *state->expected_bytes) {
      co_return Status::DataLoss(
          "HTTP body size differs from the declared representation size");
    }
    if (!state->expected_bytes.has_value())
      state->expected_bytes = state->partial_bytes;
    co_return Status{};
  }
}

struct BufferOwner {
  mem::AllocatorRef allocator{};
  std::byte *data = nullptr;
  std::size_t size = 0;

  BufferOwner(mem::AllocatorRef backing, std::size_t bytes) noexcept
      : allocator(backing), size(bytes) {}

  ~BufferOwner() {
    if (data != nullptr)
      allocator.deallocate(data, size, alignof(std::max_align_t));
  }
  BufferOwner(const BufferOwner &) = delete;
  BufferOwner &operator=(const BufferOwner &) = delete;
  BufferOwner(BufferOwner &&) = delete;
  BufferOwner &operator=(BufferOwner &&) = delete;
};

TransferReport report_without_state(Status status) noexcept {
  TransferReport report;
  report.status = std::move(status);
  return report;
}

TransferReport failure_report(detail::TransferState *&state, File &destination,
                              Status status, mem::AllocatorRef allocator,
                              IoContext &io) noexcept {
  TransferReport report;
  report.status = std::move(status);
  report.bytes_this_attempt = state->bytes_this_attempt;
  report.artifact_bytes = state->partial_bytes;
  report.total_bytes = state->expected_bytes;
  report.resumed = state->resumed;
  report.restarted = state->restarted;
  if (destination.valid())
    state->partial = destination.release();
  if (!state->partial.valid())
    state->resumable = false;

  if (!state->temp_exists) {
    report.disposition = TransferDisposition::kNoArtifact;
    destroy_state(std::exchange(state, nullptr));
    return report;
  }

  if (state->options.partial_on_failure == TransferPartialPolicy::kDiscard) {
    Status removed = remove_temp(allocator, io, *state);
    if (removed.ok()) {
      report.disposition = TransferDisposition::kPartialDiscarded;
      destroy_state(std::exchange(state, nullptr));
      return report;
    }
    state->resumable = false;
  }

  report.disposition = TransferDisposition::kPartialRetained;
  report.checkpoint =
      detail::TransferAccess::make_checkpoint(std::exchange(state, nullptr));
  return report;
}

} // namespace

KeptTransferPartial::~KeptTransferPartial() { destroy_state(state_); }

KeptTransferPartial::KeptTransferPartial(KeptTransferPartial &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

KeptTransferPartial &
KeptTransferPartial::operator=(KeptTransferPartial &&other) noexcept {
  if (this != &other) {
    destroy_state(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

bool KeptTransferPartial::valid() const noexcept { return state_ != nullptr; }

container::StringView KeptTransferPartial::path() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->temp_path);
}

const char *KeptTransferPartial::c_str() const noexcept {
  return state_ == nullptr ? "" : state_->temp_path.data();
}

TransferCheckpoint::~TransferCheckpoint() { destroy_state(state_); }

TransferCheckpoint::TransferCheckpoint(TransferCheckpoint &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

TransferCheckpoint &
TransferCheckpoint::operator=(TransferCheckpoint &&other) noexcept {
  if (this != &other) {
    destroy_state(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

bool TransferCheckpoint::valid() const noexcept {
  return state_ != nullptr && state_->temp_exists;
}

bool TransferCheckpoint::resumable() const noexcept {
  return valid() && state_->resumable && state_->partial.valid();
}

TransferSourceKind TransferCheckpoint::source_kind() const noexcept {
  return state_ == nullptr ? TransferSourceKind::kLocalFile : state_->kind;
}

std::uint64_t TransferCheckpoint::partial_bytes() const noexcept {
  return state_ == nullptr ? 0 : state_->partial_bytes;
}

std::optional<std::uint64_t>
TransferCheckpoint::expected_bytes() const noexcept {
  return state_ == nullptr ? std::optional<std::uint64_t>{}
                           : state_->expected_bytes;
}

container::StringView TransferCheckpoint::source() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->source);
}

container::StringView TransferCheckpoint::destination() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->destination);
}

container::StringView TransferCheckpoint::partial_path() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->temp_path);
}

Result<KeptTransferPartial> TransferCheckpoint::keep_partial() noexcept {
  if (!valid())
    return make_unexpected(
        Status::FailedPrecondition("transfer checkpoint is empty"));
  Status close = close_partial(*state_);
  if (!close.ok())
    return make_unexpected(std::move(close));
  state_->temp_exists = false;
  return KeptTransferPartial(std::exchange(state_, nullptr));
}

Status TransferCheckpoint::discard() noexcept {
  if (state_ == nullptr)
    return {};
  Status close = close_partial(*state_);
  if (!close.ok())
    return close;
  if (state_->temp_exists) {
    Status removed = platform::remove_file(state_->temp_path.data());
    if (removed.IsNotFound())
      removed = {};
    if (!removed.ok())
      return removed;
    state_->temp_exists = false;
  }
  destroy_state(std::exchange(state_, nullptr));
  return {};
}

TransferPlan::~TransferPlan() { destroy_state(state_); }

TransferPlan::TransferPlan(TransferPlan &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

TransferPlan &TransferPlan::operator=(TransferPlan &&other) noexcept {
  if (this != &other) {
    destroy_state(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}

Result<TransferPlan> TransferPlan::local(mem::AllocatorRef allocator,
                                         LocalTransferSource source,
                                         AtomicFileDestination destination,
                                         TransferOptions options) noexcept {
  return make_plan(
      allocator, TransferSourceKind::kLocalFile,
      std::string_view(source.path.data(), source.path.size()),
      std::string_view(destination.path.data(), destination.path.size()),
      options);
}

Result<TransferPlan> TransferPlan::http(mem::AllocatorRef allocator,
                                        HttpTransferSource source,
                                        AtomicFileDestination destination,
                                        TransferOptions options) noexcept {
  return make_plan(
      allocator, TransferSourceKind::kHttp,
      std::string_view(source.url.data(), source.url.size()),
      std::string_view(destination.path.data(), destination.path.size()),
      options);
}

Result<TransferPlan>
TransferPlan::resume(TransferCheckpoint &&checkpoint) noexcept {
  detail::TransferState *state = detail::TransferAccess::get(checkpoint);
  if (state == nullptr)
    return make_unexpected(
        Status::FailedPrecondition("transfer checkpoint is empty"));
  const TransferOptions options = state->options;
  return resume(std::move(checkpoint), options);
}

Result<TransferPlan> TransferPlan::resume(TransferCheckpoint &&checkpoint,
                                          TransferOptions options) noexcept {
  detail::TransferState *state = detail::TransferAccess::get(checkpoint);
  if (state == nullptr || !checkpoint.valid())
    return make_unexpected(
        Status::FailedPrecondition("transfer checkpoint is empty"));
  if (!checkpoint.resumable())
    return make_unexpected(Status::FailedPrecondition(
        "transfer checkpoint no longer owns a resumable file"));
  Status valid = validate_options(options);
  if (!valid.ok())
    return make_unexpected(std::move(valid));
  state = detail::TransferAccess::release(checkpoint);
  state->options = options;
  state->resume_attempt = true;
  state->bytes_this_attempt = 0;
  state->resumed = false;
  state->restarted = false;
  return detail::TransferAccess::make_plan(state);
}

bool TransferPlan::valid() const noexcept { return state_ != nullptr; }

TransferSourceKind TransferPlan::source_kind() const noexcept {
  return state_ == nullptr ? TransferSourceKind::kLocalFile : state_->kind;
}

container::StringView TransferPlan::source() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->source);
}

container::StringView TransferPlan::destination() const noexcept {
  return state_ == nullptr ? container::StringView{}
                           : public_view(state_->destination);
}

const TransferOptions &TransferPlan::options() const noexcept {
  static const TransferOptions empty{};
  return state_ == nullptr ? empty : state_->options;
}

sched::Task<TransferReport> transfer(mem::AllocatorRef allocator, IoContext &io,
                                     TransferPlan plan,
                                     TransferExecution execution) noexcept {
  detail::TransferState *state = detail::TransferAccess::release(plan);
  if (state == nullptr)
    co_return report_without_state(
        Status::InvalidArgument("transfer plan is empty"));
  if (!io.valid()) {
    Status status = io.status().ok()
                        ? Status::FailedPrecondition("IO context is invalid")
                        : io.status();
    destroy_state(state);
    co_return report_without_state(std::move(status));
  }
  if (execution.owner_worker >= io.num_workers()) {
    destroy_state(state);
    co_return report_without_state(
        Status::OutOfRange("transfer owner worker is out of range"));
  }
  if (sched::current_worker_index() !=
      static_cast<int>(execution.owner_worker)) {
    destroy_state(state);
    co_return report_without_state(Status::FailedPrecondition(
        "transfer task must run on its owner worker"));
  }

  Status status = check_budget(execution.operation);
  if (!status.ok()) {
    destroy_state(state);
    co_return report_without_state(std::move(status));
  }

  const bool needs_copy_buffer =
      state->kind == TransferSourceKind::kLocalFile ||
      (state->partial_bytes != 0 && execution.verifier.valid());
  const bool needs_local_comparison =
      state->kind == TransferSourceKind::kLocalFile &&
      state->partial_bytes != 0;
  BufferOwner buffer(
      allocator, needs_copy_buffer ? state->options.limits.buffer_bytes : 0);
  BufferOwner comparison(allocator, needs_local_comparison
                                        ? state->options.limits.buffer_bytes
                                        : 0);
  if (buffer.size != 0) {
    buffer.data = static_cast<std::byte *>(
        allocator.allocate(buffer.size, alignof(std::max_align_t)));
  }
  if (comparison.size != 0) {
    comparison.data = static_cast<std::byte *>(
        allocator.allocate(comparison.size, alignof(std::max_align_t)));
  }
  if ((buffer.size != 0 && buffer.data == nullptr) ||
      (comparison.size != 0 && comparison.data == nullptr)) {
    File empty;
    TransferReport report = failure_report(
        state, empty,
        Status::ResourceExhausted("transfer buffer allocation failed"),
        allocator, io);
    co_return report;
  }

  if (!state->temp_exists) {
    status = create_temp(allocator, io, *state);
    if (!status.ok()) {
      destroy_state(state);
      co_return report_without_state(std::move(status));
    }
  }
  status = check_budget(execution.operation);
  if (!status.ok()) {
    File empty;
    co_return failure_report(state, empty, std::move(status), allocator, io);
  }

  auto adopted =
      File::adopt(io, execution.owner_worker, std::move(state->partial));
  if (!adopted.has_value()) {
    state->resumable = false;
    File empty;
    co_return failure_report(state, empty, std::move(adopted).error(),
                             allocator, io);
  }
  File destination = std::move(*adopted);
  auto destination_info = co_await destination.info();
  if (!destination_info.has_value()) {
    co_return failure_report(
        state, destination, std::move(destination_info).error(), allocator, io);
  }
  if (!destination_info->is_regular()) {
    co_return failure_report(
        state, destination,
        Status::FailedPrecondition("transfer partial is not a regular file"),
        allocator, io);
  }
  if (destination_info->size != state->partial_bytes) {
    if (!state->resume_attempt ||
        state->options.resume_mismatch == TransferResumeMismatchPolicy::kFail) {
      co_return failure_report(
          state, destination,
          Status::DataLoss(
              "transfer partial size no longer matches checkpoint"),
          allocator, io);
    }
    status = truncate_partial(allocator, io, *state, destination);
    if (!status.ok())
      co_return failure_report(state, destination, std::move(status), allocator,
                               io);
  }

  if (state->kind == TransferSourceKind::kLocalFile) {
    status = co_await run_local_transfer(
        allocator, &io, state, &destination,
        std::span<std::byte>(buffer.data, buffer.size),
        std::span<std::byte>(comparison.data, comparison.size), execution);
  } else {
    status = co_await run_http_transfer(
        allocator, &io, state, &destination,
        std::span<std::byte>(buffer.data, buffer.size), execution);
  }
  if (!status.ok())
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);

  status = notify_progress(execution, *state, TransferPhase::kVerifying);
  if (status.ok() && execution.verifier.valid())
    status = execution.verifier.finish(state->partial_bytes);
  if (status.ok())
    status = check_budget(execution.operation);
  if (!status.ok())
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);

  status = co_await destination.sync(FileSyncMode::kDataAndMetadata);
  if (status.ok())
    status = check_budget(execution.operation);
  if (!status.ok())
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);

  status = co_await destination.close();
  if (!status.ok()) {
    state->resumable = false;
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);
  }
  status = notify_progress(execution, *state, TransferPhase::kPublishing);
  if (status.ok())
    status = check_budget(execution.operation);
  if (!status.ok())
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);

  const char *temporary = state->temp_path.data();
  const char *final_path = state->destination.data();
  status =
      run_blocking_status(allocator, io, [temporary, final_path]() noexcept {
        return platform::rename_path(temporary, final_path);
      });
  if (!status.ok()) {
    state->resumable = false;
    co_return failure_report(state, destination, std::move(status), allocator,
                             io);
  }

  state->temp_exists = false;
  TransferReport report;
  report.disposition = TransferDisposition::kPublished;
  report.durability = TransferDurability::kFileDataAndMetadata;
  report.bytes_this_attempt = state->bytes_this_attempt;
  report.artifact_bytes = state->partial_bytes;
  report.total_bytes = state->expected_bytes;
  report.resumed = state->resumed;
  report.restarted = state->restarted;
  destroy_state(std::exchange(state, nullptr));
  co_return report;
}

} // namespace hyle::io
