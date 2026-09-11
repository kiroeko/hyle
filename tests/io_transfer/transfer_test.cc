#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/transfer.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

#if defined(HYLE_IO_TLS)
#include "../support/tls_test_material.h"
#endif

namespace {

using hyle::Status;
using hyle::io::AtomicFileDestination;
using hyle::io::HttpHeader;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
using hyle::io::HttpTransferSource;
using hyle::io::IoContext;
using hyle::io::LocalTransferSource;
using hyle::io::KeptTransferPartial;
using hyle::io::TcpListener;
using hyle::io::TransferCheckpoint;
using hyle::io::TransferDisposition;
using hyle::io::TransferExecution;
using hyle::io::TransferOptions;
using hyle::io::TransferPartialPolicy;
using hyle::io::TransferPlan;
using hyle::io::TransferProgress;
using hyle::io::TransferProgressObserver;
using hyle::io::TransferReport;
using hyle::io::TransferResumeMismatchPolicy;
using hyle::io::TransferSourceKind;
using hyle::io::TransferVerifier;
#if defined(HYLE_IO_TLS)
using hyle::io::tls_handshake;
using hyle::io::TlsClientOptions;
using hyle::io::TlsContext;
using hyle::io::TlsCredentialSource;
using hyle::io::TlsServerOptions;
using hyle::io::TlsStream;
#endif
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr unsigned kOwner = 0;

std::string unique_path(const char *stem) {
  static std::atomic<unsigned> sequence{0};
  return std::string(stem) + "_" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
         ".tmp";
}

Status write_file(const std::string &path, std::string_view value) {
  auto opened = hyle::platform::open_file(
      path.c_str(), hyle::platform::kOpenWrite | hyle::platform::kOpenCreate |
                        hyle::platform::kOpenTruncate);
  if (!opened.has_value())
    return std::move(opened).error();
  Status status = opened->write_all(value.data(), value.size());
  if (status.ok())
    status = opened->sync();
  Status close = opened->close();
  status.Update(std::move(close));
  return status;
}

hyle::Result<std::string> read_file(const std::string &path) {
  auto opened =
      hyle::platform::open_file(path.c_str(), hyle::platform::kOpenRead);
  if (!opened.has_value())
    return hyle::make_unexpected(opened.error());
  auto size = opened->size();
  if (!size.has_value())
    return hyle::make_unexpected(size.error());
  std::string result(static_cast<std::size_t>(*size), '\0');
  Status status = opened->read_exact(result.data(), result.size());
  if (!status.ok())
    return hyle::make_unexpected(std::move(status));
  return result;
}

void remove_if_present(const std::string &path) {
  Status status = hyle::platform::remove_file(path.c_str());
  if (!status.ok() && !status.IsNotFound())
    status.IgnoreError();
}

struct Drive {
  Task<TransferReport> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

TransferReport run_transfer(TransferPlan plan,
                            TransferExecution execution = {}) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  EXPECT_TRUE(io.valid()) << io.status().message();
  Drive drive{
      hyle::io::transfer(system_allocator(), io, std::move(plan), execution)};
  EXPECT_TRUE(drive.task.valid());
  Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry, &drive);
  EXPECT_TRUE(scheduler.spawn_on(execution.owner_worker, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(drive.task.done());
  return drive.task.result();
}

std::span<const std::byte> byte_view(std::string_view value) noexcept {
  return std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(value.data()), value.size());
}

enum class HttpScenarioMode {
  kFull,
  kStatus,
  kRedirectThenFull,
  kRedirectLoop,
  kRange,
  kRangeWithoutValidator,
  kRangeComplete,
  kChangedValidatorThenFull,
  kIgnoreRange,
  kUnsupportedRedirect,
};

struct HttpScenario {
  HttpScenarioMode mode = HttpScenarioMode::kFull;
  std::string_view payload;
  int response_status = 200;
  std::string_view etag = "\"hyle-transfer-etag\"";
  std::string_view expected_if_range = "\"hyle-transfer-etag\"";
  std::string_view content_range_override{};
  std::uint64_t range_offset = 0;
  bool saw_expected_range = false;
  bool saw_expected_if_range = false;
  bool saw_resolved_redirect = false;
  Status status{};
};

Task<int> serve_http(AllocatorRef allocator, TcpListener *listener,
                     HttpScenario *scenario) {
  const unsigned request_count =
      scenario->mode == HttpScenarioMode::kRedirectThenFull ||
              scenario->mode == HttpScenarioMode::kRedirectLoop ||
              scenario->mode == HttpScenarioMode::kChangedValidatorThenFull
          ? 2
          : 1;
  for (unsigned index = 0; index < request_count; ++index) {
    auto accepted = co_await listener->accept(kOwner);
    if (!accepted.has_value()) {
      scenario->status = std::move(accepted).error();
      co_return 0;
    }
    auto made = HttpServerConnection::create(allocator, std::move(*accepted));
    if (!made.has_value()) {
      scenario->status = std::move(made).error();
      co_return 0;
    }
    HttpServerConnection connection = std::move(*made);
    auto incoming = co_await connection.read_request();
    if (!incoming.has_value() || !incoming->has_value()) {
      if (incoming.has_value())
        scenario->status = Status::DataLoss("test HTTP request ended early");
      else
        scenario->status = std::move(incoming).error();
      co_return 0;
    }
    auto request = std::move(incoming->value());
    Status discarded = co_await request.body.discard();
    if (!discarded.ok()) {
      scenario->status = discarded;
      co_return 0;
    }

    HttpResponseHead response;
    response.keep_alive = false;
    if (scenario->mode == HttpScenarioMode::kStatus) {
      response.status = scenario->response_status;
      scenario->status = co_await connection.send_response(
          response, std::span<const std::byte>{});
      if (!scenario->status.ok())
        co_return 0;
      continue;
    }
    if ((scenario->mode == HttpScenarioMode::kRedirectThenFull && index == 0) ||
        scenario->mode == HttpScenarioMode::kRedirectLoop ||
        scenario->mode == HttpScenarioMode::kUnsupportedRedirect) {
      if (scenario->mode == HttpScenarioMode::kRedirectThenFull)
        scenario->saw_resolved_redirect = request.head.target() == "/dir/start";
      const hyle::container::StringView location =
          scenario->mode == HttpScenarioMode::kRedirectLoop
              ? hyle::container::StringView("/again")
          : scenario->mode == HttpScenarioMode::kUnsupportedRedirect
              ? hyle::container::StringView("ftp://example.invalid/model")
              : hyle::container::StringView("../final/model");
      const HttpHeader redirect_headers[] = {{"Location", location}};
      response.status = 302;
      response.headers = redirect_headers;
      scenario->status = co_await connection.send_response(
          response, std::span<const std::byte>{});
      if (!scenario->status.ok())
        co_return 0;
      continue;
    }

    if (scenario->mode == HttpScenarioMode::kRedirectThenFull)
      scenario->saw_resolved_redirect = scenario->saw_resolved_redirect &&
                                        request.head.target() == "/final/model";

    std::string content_range;
    HttpHeader response_headers[2] = {{"ETag", scenario->etag}, {}};
    std::size_t response_header_count =
        scenario->mode == HttpScenarioMode::kRangeWithoutValidator ? 0 : 1;
    std::string_view body = scenario->payload;
    const auto range = request.head.headers().find("range");
    const auto if_range = request.head.headers().find("if-range");
    if (scenario->mode == HttpScenarioMode::kRange ||
        scenario->mode == HttpScenarioMode::kRangeWithoutValidator ||
        scenario->mode == HttpScenarioMode::kRangeComplete ||
        (scenario->mode == HttpScenarioMode::kRedirectThenFull && index != 0 &&
         range.has_value()) ||
        (scenario->mode == HttpScenarioMode::kChangedValidatorThenFull &&
         index == 0)) {
      const std::string expected_range =
          "bytes=" + std::to_string(scenario->range_offset) + "-";
      scenario->saw_expected_range =
          range.has_value() &&
          std::string_view(range->data(), range->size()) == expected_range;
      scenario->saw_expected_if_range =
          if_range.has_value() &&
          std::string_view(if_range->data(), if_range->size()) ==
              scenario->expected_if_range;
      if (scenario->range_offset > scenario->payload.size()) {
        scenario->status = Status::Internal("test range offset is invalid");
        co_return 0;
      }
      if (scenario->mode == HttpScenarioMode::kRangeComplete) {
        content_range = "bytes */" + std::to_string(scenario->payload.size());
        response.status = 416;
        response_headers[response_header_count++] = {"Content-Range",
                                                     content_range};
        response.headers = std::span<const HttpHeader>(response_headers,
                                                       response_header_count);
        scenario->status = co_await connection.send_response(
            response, std::span<const std::byte>{});
        if (!scenario->status.ok())
          co_return 0;
        continue;
      }
      body.remove_prefix(static_cast<std::size_t>(scenario->range_offset));
      content_range = "bytes " + std::to_string(scenario->range_offset) + "-" +
                      std::to_string(scenario->payload.size() - 1) + "/" +
                      std::to_string(scenario->payload.size());
      if (!scenario->content_range_override.empty())
        content_range.assign(scenario->content_range_override);
      response.status = 206;
      response_headers[response_header_count++] = {"Content-Range",
                                                   content_range};
    } else if (scenario->mode == HttpScenarioMode::kIgnoreRange) {
      scenario->saw_expected_range = range.has_value();
      scenario->saw_expected_if_range = if_range.has_value();
    }
    response.headers =
        std::span<const HttpHeader>(response_headers, response_header_count);
    scenario->status =
        co_await connection.send_response(response, byte_view(body));
    if (!scenario->status.ok())
      co_return 0;
  }
  co_return 0;
}

#if defined(HYLE_IO_TLS)
Task<int> serve_https(AllocatorRef allocator, TcpListener *listener,
                      TlsContext *tls_context,
                      HttpScenario *scenario) noexcept {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value()) {
    scenario->status = std::move(accepted).error();
    co_return 0;
  }
  auto wrapped = TlsStream::accept(std::move(*accepted), *tls_context);
  if (!wrapped.has_value()) {
    scenario->status = std::move(wrapped).error();
    co_return 0;
  }
  TlsStream stream = std::move(*wrapped);
  scenario->status = co_await tls_handshake(allocator, stream);
  if (!scenario->status.ok())
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(stream));
  if (!made.has_value()) {
    scenario->status = std::move(made).error();
    co_return 0;
  }
  HttpServerConnection connection = std::move(*made);
  auto incoming = co_await connection.read_request();
  if (!incoming.has_value() || !incoming->has_value()) {
    if (incoming.has_value())
      scenario->status = Status::DataLoss("test HTTPS request ended early");
    else
      scenario->status = std::move(incoming).error();
    co_return 0;
  }
  auto request = std::move(incoming->value());
  scenario->status = co_await request.body.discard();
  if (!scenario->status.ok())
    co_return 0;
  const HttpHeader headers[] = {{"ETag", scenario->etag}};
  HttpResponseHead response;
  response.headers = headers;
  response.keep_alive = false;
  scenario->status =
      co_await connection.send_response(response, byte_view(scenario->payload));
  co_return 0;
}
#endif

struct HttpServerDrive {
  Task<int> task;
};

void server_drive_entry(void *context) noexcept {
  static_cast<HttpServerDrive *>(context)->task.resume();
}

TransferReport run_http_transfer(TransferPlan plan, TransferExecution execution,
                                 HttpScenario &scenario, std::uint16_t port) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  EXPECT_TRUE(io.valid()) << io.status().message();
  auto bound = TcpListener::bind(
      io, kOwner,
      hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4, port));
  EXPECT_TRUE(bound.has_value()) << bound.error().message();
  TcpListener listener = std::move(*bound);
  HttpServerDrive server{serve_http(system_allocator(), &listener, &scenario)};
  Drive client{
      hyle::io::transfer(system_allocator(), io, std::move(plan), execution)};
  Fiber server_fiber(hyle::sched::kDefaultStackSize, &server_drive_entry,
                     &server);
  Fiber client_fiber(hyle::sched::kDefaultStackSize, &drive_entry, &client);
  EXPECT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  EXPECT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(server.task.done());
  EXPECT_TRUE(client.task.done());
  (void)server.task.result();
  return client.task.result();
}

#if defined(HYLE_IO_TLS)
TransferReport run_https_transfer(TransferPlan plan,
                                  TransferExecution execution,
                                  HttpScenario &scenario, std::uint16_t port,
                                  TlsContext &server_context) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  EXPECT_TRUE(io.valid()) << io.status().message();
  auto bound = TcpListener::bind(
      io, kOwner,
      hyle::io::Endpoint::loopback(hyle::io::AddressFamily::kIpv4, port));
  EXPECT_TRUE(bound.has_value()) << bound.error().message();
  TcpListener listener = std::move(*bound);
  HttpServerDrive server{
      serve_https(system_allocator(), &listener, &server_context, &scenario)};
  Drive client{
      hyle::io::transfer(system_allocator(), io, std::move(plan), execution)};
  Fiber server_fiber(hyle::sched::kDefaultStackSize, &server_drive_entry,
                     &server);
  Fiber client_fiber(hyle::sched::kDefaultStackSize, &drive_entry, &client);
  EXPECT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  EXPECT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(server.task.done());
  EXPECT_TRUE(client.task.done());
  (void)server.task.result();
  return client.task.result();
}
#endif

struct StopAfter {
  std::uint64_t threshold = 0;
  unsigned calls = 0;

  Status operator()(const TransferProgress &progress) noexcept {
    ++calls;
    if (progress.artifact_bytes >= threshold && threshold != 0)
      return Status::Cancelled("test requested a retained partial");
    return {};
  }
};

struct FnvVerifier {
  std::uint64_t expected = 0;
  std::uint64_t hash = 0;
  std::uint64_t expected_size = 0;
  unsigned resets = 0;

  static std::uint64_t digest(std::string_view value) noexcept {
    std::uint64_t result = 1469598103934665603ull;
    for (const unsigned char byte : value) {
      result ^= byte;
      result *= 1099511628211ull;
    }
    return result;
  }

  Status reset() noexcept {
    ++resets;
    hash = 1469598103934665603ull;
    return {};
  }
  Status update(std::span<const std::byte> bytes) noexcept {
    for (std::byte byte : bytes) {
      hash ^= std::to_integer<unsigned char>(byte);
      hash *= 1099511628211ull;
    }
    return {};
  }
  Status finish(std::uint64_t size) noexcept {
    if (size != expected_size || hash != expected)
      return Status::DataLoss("test checksum mismatch");
    return {};
  }
};

AllocatorRef failing_allocator() noexcept {
  return AllocatorRef([](void *, std::size_t,
                         std::size_t) noexcept -> void * { return nullptr; },
                      [](void *, void *, std::size_t, std::size_t) noexcept {},
                      [](void *, void *, std::size_t, std::size_t,
                         std::size_t) noexcept -> void * { return nullptr; },
                      nullptr);
}

TEST(IoTransferPlan, OwnsConfigurationAndRejectsUnsafeUrlShapes) {
  std::string source = "source.bin";
  std::string destination = "destination.bin";
  auto plan = TransferPlan::local(system_allocator(), {source}, {destination});
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  source.assign("mutated");
  destination.assign("mutated");
  EXPECT_TRUE(plan->valid());
  EXPECT_EQ(plan->source_kind(), TransferSourceKind::kLocalFile);
  EXPECT_EQ(plan->source(), "source.bin");
  EXPECT_EQ(plan->destination(), "destination.bin");
  EXPECT_EQ(plan->options().limits.buffer_bytes, 1024u * 1024u);

  auto http = TransferPlan::http(system_allocator(),
                                 {"https://example.com/model.gguf"},
                                 {"model.gguf"});
  ASSERT_TRUE(http.has_value());
  EXPECT_EQ(http->source_kind(), TransferSourceKind::kHttp);
  TransferPlan moved = std::move(*plan);
  EXPECT_FALSE(plan->valid());
  *http = std::move(moved);
  EXPECT_FALSE(moved.valid());
  EXPECT_EQ(http->source_kind(), TransferSourceKind::kLocalFile);
  EXPECT_EQ(http->source(), "source.bin");
  auto credentials = TransferPlan::http(
      system_allocator(), {"https://token@example.com/model"}, {"model"});
  EXPECT_FALSE(credentials.has_value());
  EXPECT_TRUE(credentials.error().IsInvalidArgument());
  auto fragment = TransferPlan::http(
      system_allocator(), {"https://example.com/model#part"}, {"model"});
  EXPECT_FALSE(fragment.has_value());
  EXPECT_TRUE(fragment.error().IsInvalidArgument());

  TransferOptions invalid;
  invalid.limits.buffer_bytes = 0;
  EXPECT_FALSE(
      TransferPlan::local(system_allocator(), {"source"}, {"target"}, invalid)
          .has_value());
  auto oom = TransferPlan::local(failing_allocator(), {"source"}, {"target"});
  EXPECT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());
}

TEST(IoTransferLocal, AtomicallyReplacesOldTargetAndVerifiesWholeArtifact) {
  const std::string source = unique_path("hyle_transfer_source");
  const std::string destination = unique_path("hyle_transfer_destination");
  const std::string payload =
      "model-header-and-weights-that-cross-several-small-copy-chunks";
  ASSERT_TRUE(write_file(source, payload).ok());
  ASSERT_TRUE(
      write_file(destination, "old-model-must-survive-until-commit").ok());

  TransferOptions options;
  options.limits.buffer_bytes = 7;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  FnvVerifier verifier{.expected = FnvVerifier::digest(payload),
                       .expected_size = payload.size()};
  TransferExecution execution;
  execution.verifier = TransferVerifier::from(verifier);
  TransferReport report = run_transfer(std::move(*plan), execution);

  EXPECT_TRUE(report.published()) << report.status.message();
  EXPECT_EQ(report.disposition, TransferDisposition::kPublished);
  EXPECT_EQ(report.artifact_bytes, payload.size());
  EXPECT_EQ(report.bytes_this_attempt, payload.size());
  EXPECT_EQ(verifier.resets, 1u);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(source);
  remove_if_present(destination);
}

TEST(IoTransferLocal, RetainedCheckpointResumesAndRehashesPrefix) {
  const std::string source = unique_path("hyle_transfer_resume_source");
  const std::string destination = unique_path("hyle_transfer_resume_dest");
  const std::string payload = "0123456789abcdefghijklmnopqrstuvwxyz";
  ASSERT_TRUE(write_file(source, payload).ok());

  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  TransferReport first = run_transfer(std::move(*plan), interrupted);
  ASSERT_TRUE(first.status.IsCancelled()) << first.status.message();
  ASSERT_EQ(first.disposition, TransferDisposition::kPartialRetained);
  ASSERT_TRUE(first.checkpoint.valid());
  ASSERT_TRUE(first.checkpoint.resumable());
  EXPECT_EQ(first.checkpoint.source_kind(), TransferSourceKind::kLocalFile);
  EXPECT_EQ(first.checkpoint.source(), source);
  EXPECT_EQ(first.checkpoint.destination(), destination);
  EXPECT_FALSE(first.checkpoint.partial_path().empty());
  EXPECT_EQ(first.checkpoint.expected_bytes(),
            std::optional<std::uint64_t>{payload.size()});
  const std::uint64_t prefix = first.checkpoint.partial_bytes();
  EXPECT_GE(prefix, 8u);
  EXPECT_FALSE(hyle::platform::exists(destination.c_str()).value_or(false));

  TransferCheckpoint checkpoint;
  checkpoint = std::move(first.checkpoint);
  EXPECT_FALSE(first.checkpoint.valid());
  ASSERT_TRUE(checkpoint.resumable());
  auto resumed = TransferPlan::resume(std::move(checkpoint));
  ASSERT_TRUE(resumed.has_value()) << resumed.error().message();
  FnvVerifier verifier{.expected = FnvVerifier::digest(payload),
                       .expected_size = payload.size()};
  TransferExecution finish;
  finish.verifier = TransferVerifier::from(verifier);
  TransferReport second = run_transfer(std::move(*resumed), finish);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.resumed);
  EXPECT_FALSE(second.restarted);
  EXPECT_EQ(second.bytes_this_attempt, payload.size() - prefix);
  EXPECT_EQ(verifier.resets, 1u);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(source);
  remove_if_present(destination);
}

#if HYLE_OS_LINUX || HYLE_OS_MACOS
TEST(IoTransferLocal, ChangedPartialPrefixRestartsInsteadOfBlindlyAppending) {
  const std::string source = unique_path("hyle_transfer_prefix_source");
  const std::string destination = unique_path("hyle_transfer_prefix_dest");
  const std::string payload = "0123456789-stable-model-payload";
  ASSERT_TRUE(write_file(source, payload).ok());

  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  TransferReport first = run_transfer(std::move(*plan), interrupted);
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::string partial(first.checkpoint.partial_path().data(),
                            first.checkpoint.partial_path().size());

  auto changed =
      hyle::platform::open_file(partial.c_str(), hyle::platform::kOpenWrite);
  ASSERT_TRUE(changed.has_value()) << changed.error().message();
  ASSERT_TRUE(changed->write_all("X", 1).ok());
  ASSERT_TRUE(changed->close().ok());

  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value()) << resumed.error().message();
  TransferReport second = run_transfer(std::move(*resumed));
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.restarted);
  EXPECT_FALSE(second.resumed);
  EXPECT_EQ(second.bytes_this_attempt, payload.size());
  ASSERT_TRUE(second.total_bytes.has_value());
  EXPECT_EQ(*second.total_bytes, payload.size());
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value()) << actual.error().message();
  EXPECT_EQ(*actual, payload);
  remove_if_present(source);
  remove_if_present(destination);
}
#endif

TEST(IoTransferLocal, ChangedSourceRestartsInsteadOfAppending) {
  const std::string source = unique_path("hyle_transfer_changed_source");
  const std::string destination = unique_path("hyle_transfer_changed_dest");
  ASSERT_TRUE(write_file(source, "first-source-content").ok());

  TransferOptions options;
  options.limits.buffer_bytes = 3;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 6};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  TransferReport first = run_transfer(std::move(*plan), interrupted);
  ASSERT_TRUE(first.checkpoint.resumable());

  const std::string replacement = "a-different-and-longer-source";
  ASSERT_TRUE(write_file(source, replacement).ok());
  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value());
  TransferReport second = run_transfer(std::move(*resumed));
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.restarted);
  EXPECT_FALSE(second.resumed);
  EXPECT_EQ(second.bytes_this_attempt, replacement.size());
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, replacement);
  remove_if_present(source);
  remove_if_present(destination);
}

TEST(IoTransferLocal, FailurePolicyAndVerifierNeverDamageOldTarget) {
  const std::string source = unique_path("hyle_transfer_failure_source");
  const std::string destination = unique_path("hyle_transfer_failure_dest");
  const std::string payload = "new-model-content";
  ASSERT_TRUE(write_file(source, payload).ok());
  ASSERT_TRUE(write_file(destination, "known-good-old-model").ok());

  TransferOptions options;
  options.limits.buffer_bytes = 4;
  options.partial_on_failure = TransferPartialPolicy::kDiscard;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  FnvVerifier verifier{.expected = 1, .expected_size = payload.size()};
  TransferExecution execution;
  execution.verifier = TransferVerifier::from(verifier);
  TransferReport report = run_transfer(std::move(*plan), execution);
  EXPECT_TRUE(report.status.IsDataLoss());
  EXPECT_EQ(report.disposition, TransferDisposition::kPartialDiscarded);
  EXPECT_FALSE(report.checkpoint.valid());
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, "known-good-old-model");

  TransferOptions bounded;
  bounded.limits.max_bytes = 2;
  auto too_large =
      TransferPlan::local(system_allocator(), {source}, {destination}, bounded);
  ASSERT_TRUE(too_large.has_value());
  TransferReport bounded_report = run_transfer(std::move(*too_large));
  EXPECT_TRUE(bounded_report.status.IsOutOfRange());
  actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, "known-good-old-model");
  remove_if_present(source);
  remove_if_present(destination);
}

TEST(IoTransferLocal, EmptySourcePublishesAnEmptyArtifact) {
  const std::string source = unique_path("hyle_transfer_empty_source");
  const std::string destination = unique_path("hyle_transfer_empty_dest");
  ASSERT_TRUE(write_file(source, {}).ok());
  ASSERT_TRUE(write_file(destination, "old-model").ok());
  auto plan = TransferPlan::local(system_allocator(), {source}, {destination});
  ASSERT_TRUE(plan.has_value());

  FnvVerifier verifier{.expected = FnvVerifier::digest({}), .expected_size = 0};
  TransferExecution execution;
  execution.verifier = TransferVerifier::from(verifier);
  TransferReport report = run_transfer(std::move(*plan), execution);

  EXPECT_TRUE(report.published()) << report.status.message();
  EXPECT_EQ(report.artifact_bytes, 0u);
  EXPECT_EQ(report.bytes_this_attempt, 0u);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_TRUE(actual->empty());
  remove_if_present(source);
  remove_if_present(destination);
}

#if HYLE_OS_LINUX
TEST(IoTransferLocal, ReadsVirtualRegularFileWhoseMetadataSizeIsZero) {
  constexpr const char *source = "/proc/self/cmdline";
  auto source_info = hyle::platform::stat(source);
  ASSERT_TRUE(source_info.has_value()) << source_info.error().message();
  ASSERT_TRUE(source_info->is_regular());
  ASSERT_EQ(source_info->size, 0u);

  const std::string destination =
      unique_path("hyle_transfer_virtual_regular_dest");
  TransferOptions options;
  options.limits.buffer_bytes = 3;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  TransferReport report = run_transfer(std::move(*plan));

  EXPECT_TRUE(report.published()) << report.status.message();
  EXPECT_GT(report.artifact_bytes, 0u);
  ASSERT_TRUE(report.total_bytes.has_value());
  EXPECT_EQ(*report.total_bytes, report.artifact_bytes);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value()) << actual.error().message();
  EXPECT_EQ(actual->size(), report.artifact_bytes);
  EXPECT_FALSE(actual->empty());
  remove_if_present(destination);

  const std::string bounded_destination =
      unique_path("hyle_transfer_virtual_regular_bounded_dest");
  ASSERT_TRUE(write_file(bounded_destination, "old-model").ok());
  options.limits.max_bytes = 1;
  options.partial_on_failure = TransferPartialPolicy::kDiscard;
  auto bounded = TransferPlan::local(system_allocator(), {source},
                                     {bounded_destination}, options);
  ASSERT_TRUE(bounded.has_value()) << bounded.error().message();
  TransferReport bounded_report = run_transfer(std::move(*bounded));
  EXPECT_TRUE(bounded_report.status.IsOutOfRange())
      << bounded_report.status.message();
  auto preserved = read_file(bounded_destination);
  ASSERT_TRUE(preserved.has_value()) << preserved.error().message();
  EXPECT_EQ(*preserved, "old-model");
  remove_if_present(bounded_destination);
}
#endif

TEST(IoTransferLocal, ExpiredOrCancelledOperationCreatesNoArtifact) {
  const std::string cancelled_destination =
      unique_path("hyle_transfer_precancelled_dest");
  auto cancelled_plan = TransferPlan::local(
      system_allocator(), {"source-is-never-opened"}, {cancelled_destination});
  ASSERT_TRUE(cancelled_plan.has_value());
  hyle::sched::CancelToken cancellation =
      hyle::sched::CancelToken::root(system_allocator());
  cancellation.cancel();
  TransferExecution cancelled_execution;
  cancelled_execution.operation.cancel = &cancellation;
  TransferReport cancelled =
      run_transfer(std::move(*cancelled_plan), cancelled_execution);
  EXPECT_TRUE(cancelled.status.IsCancelled()) << cancelled.status.message();
  EXPECT_EQ(cancelled.disposition, TransferDisposition::kNoArtifact);
  EXPECT_FALSE(cancelled.checkpoint.valid());
  EXPECT_FALSE(
      hyle::platform::exists(cancelled_destination.c_str()).value_or(false));

  const std::string expired_destination =
      unique_path("hyle_transfer_expired_dest");
  auto expired_plan = TransferPlan::local(
      system_allocator(), {"source-is-never-opened"}, {expired_destination});
  ASSERT_TRUE(expired_plan.has_value());
  TransferExecution expired_execution;
  expired_execution.operation.deadline = hyle::time::monotonic_now();
  TransferReport expired =
      run_transfer(std::move(*expired_plan), expired_execution);
  EXPECT_TRUE(expired.status.IsDeadlineExceeded()) << expired.status.message();
  EXPECT_EQ(expired.disposition, TransferDisposition::kNoArtifact);
  EXPECT_FALSE(expired.checkpoint.valid());
  EXPECT_FALSE(
      hyle::platform::exists(expired_destination.c_str()).value_or(false));
}

TEST(IoTransferLocal, ResumeMismatchCanFailWithoutDamagingEitherArtifact) {
  const std::string source = unique_path("hyle_transfer_fail_mismatch_source");
  const std::string destination =
      unique_path("hyle_transfer_fail_mismatch_dest");
  ASSERT_TRUE(write_file(source, "original-source").ok());
  ASSERT_TRUE(write_file(destination, "known-good-old-model").ok());

  TransferOptions options;
  options.limits.buffer_bytes = 3;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 6};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  TransferReport first = run_transfer(std::move(*plan), interrupted);
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::uint64_t retained_bytes = first.checkpoint.partial_bytes();

  ASSERT_TRUE(write_file(source, "replacement-source-is-longer").ok());
  options.resume_mismatch = TransferResumeMismatchPolicy::kFail;
  auto resumed = TransferPlan::resume(std::move(first.checkpoint), options);
  ASSERT_TRUE(resumed.has_value());
  TransferReport second = run_transfer(std::move(*resumed));
  EXPECT_TRUE(second.status.IsFailedPrecondition()) << second.status.message();
  EXPECT_EQ(second.artifact_bytes, retained_bytes);
  ASSERT_TRUE(second.checkpoint.resumable());
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, "known-good-old-model");
  EXPECT_TRUE(second.checkpoint.discard().ok());
  remove_if_present(source);
  remove_if_present(destination);
}

TEST(IoTransferLocal, KeepPartialExplicitlyTransfersCleanupOwnership) {
  const std::string source = unique_path("hyle_transfer_keep_source");
  const std::string destination = unique_path("hyle_transfer_keep_dest");
  ASSERT_TRUE(write_file(source, "partial-model-content").ok());
  TransferOptions options;
  options.limits.buffer_bytes = 3;
  auto plan =
      TransferPlan::local(system_allocator(), {source}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 6};
  TransferExecution execution;
  execution.progress = TransferProgressObserver::from(stopper);
  TransferReport report = run_transfer(std::move(*plan), execution);
  ASSERT_TRUE(report.checkpoint.resumable());

  std::string path;
  {
    auto kept = report.checkpoint.keep_partial();
    ASSERT_TRUE(kept.has_value()) << kept.error().message();
    KeptTransferPartial owner;
    owner = std::move(*kept);
    EXPECT_FALSE(kept->valid());
    ASSERT_TRUE(owner.valid());
    ASSERT_FALSE(owner.path().empty());
    EXPECT_STREQ(owner.c_str(), owner.path().c_str());
    path.assign(owner.path().data(), owner.path().size());
  }
  EXPECT_FALSE(report.checkpoint.valid());
  // KeptTransferPartial owns only the stable path value. Once keep_partial()
  // succeeds, destruction must not reclaim the caller-owned file.
  EXPECT_TRUE(hyle::platform::exists(path.c_str()).value_or(false));
  remove_if_present(path);
  remove_if_present(source);
  remove_if_present(destination);
}

TEST(IoTransferHttp, FollowsBoundedRelativeRedirectAndPublishesStreamingBody) {
  constexpr std::uint16_t port = 54571;
  const std::string destination = unique_path("hyle_transfer_http_redirect");
  const std::string payload = "remote-model-content-through-relative-redirect";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/dir/start";
  TransferOptions options;
  options.limits.buffer_bytes = 6;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  HttpScenario scenario{.mode = HttpScenarioMode::kRedirectThenFull,
                        .payload = payload};
  TransferReport report =
      run_http_transfer(std::move(*plan), {}, scenario, port);
  EXPECT_TRUE(report.published()) << report.status.message();
  EXPECT_TRUE(scenario.status.ok()) << scenario.status.message();
  EXPECT_TRUE(scenario.saw_resolved_redirect);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, ResumeReacquiresTheOriginalRedirectChain) {
  constexpr std::uint16_t port = 54579;
  const std::string destination =
      unique_path("hyle_transfer_http_redirect_resume");
  const std::string payload = "redirected-model-content-for-resume";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/dir/start";
  TransferOptions options;
  options.limits.buffer_bytes = 5;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kRedirectThenFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.status.IsCancelled()) << first.status.message();
  ASSERT_TRUE(first.checkpoint.resumable());

  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value()) << resumed.error().message();
  HttpScenario second_server{.mode = HttpScenarioMode::kRedirectThenFull,
                             .payload = payload,
                             .range_offset = first.artifact_bytes};
  TransferReport second =
      run_http_transfer(std::move(*resumed), {}, second_server, port);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.resumed);
  EXPECT_TRUE(second_server.saw_resolved_redirect);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, RejectsUnsupportedAbsoluteRedirectScheme) {
  constexpr std::uint16_t port = 54580;
  const std::string destination =
      unique_path("hyle_transfer_http_unsupported_redirect");
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.partial_on_failure = TransferPartialPolicy::kDiscard;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  HttpScenario scenario{.mode = HttpScenarioMode::kUnsupportedRedirect,
                        .payload = {}};
  TransferReport report =
      run_http_transfer(std::move(*plan), {}, scenario, port);
  EXPECT_TRUE(report.status.IsPermissionDenied()) << report.status.message();
  EXPECT_EQ(report.disposition, TransferDisposition::kPartialDiscarded);
  EXPECT_FALSE(report.checkpoint.valid());
  EXPECT_FALSE(hyle::platform::exists(destination.c_str()).value_or(false));
}

TEST(IoTransferHttp, MapsUnusableResponseStatusWithoutPublishing) {
  struct Case {
    int http_status;
    hyle::StatusCode expected;
  };
  constexpr std::array cases{
      Case{401, hyle::StatusCode::Unauthenticated},
      Case{403, hyle::StatusCode::PermissionDenied},
      Case{404, hyle::StatusCode::NotFound},
      Case{503, hyle::StatusCode::Unavailable},
      Case{429, hyle::StatusCode::ResourceExhausted},
      Case{400, hyle::StatusCode::FailedPrecondition},
  };

  std::uint16_t port = 54582;
  for (const Case &test : cases) {
    const std::string destination =
        unique_path("hyle_transfer_http_status_failure");
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
    TransferOptions options;
    options.partial_on_failure = TransferPartialPolicy::kDiscard;
    auto plan =
        TransferPlan::http(system_allocator(), {url}, {destination}, options);
    ASSERT_TRUE(plan.has_value()) << plan.error().message();
    HttpScenario scenario{.mode = HttpScenarioMode::kStatus,
                          .payload = {},
                          .response_status = test.http_status};
    TransferReport report =
        run_http_transfer(std::move(*plan), {}, scenario, port++);
    EXPECT_EQ(report.status.code(), test.expected) << test.http_status;
    EXPECT_EQ(report.disposition, TransferDisposition::kPartialDiscarded);
    EXPECT_FALSE(report.checkpoint.valid());
    EXPECT_FALSE(hyle::platform::exists(destination.c_str()).value_or(false));
  }
}

#if defined(HYLE_IO_TLS)
TEST(IoTransferHttp, HttpsUsesBorrowedVerifiedTlsPolicy) {
  constexpr std::uint16_t port = 54578;
  const std::string cert_path = unique_path("hyle_transfer_https_cert");
  const std::string key_path = unique_path("hyle_transfer_https_key");
  const std::string destination =
      unique_path("hyle_transfer_https_destination");
  ASSERT_TRUE(
      write_file(cert_path, hyle::test_support::kTlsCertificatePem).ok());
  ASSERT_TRUE(write_file(key_path, hyle::test_support::kTlsPrivateKeyPem).ok());

  const hyle::container::StringView protocols[] = {"http/1.1"};
  TlsServerOptions server_options;
  server_options.certificate_chain = TlsCredentialSource::file(cert_path);
  server_options.private_key = TlsCredentialSource::file(key_path);
  server_options.alpn_protocols = protocols;
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server.has_value()) << made_server.error().message();
  TlsContext server_context = std::move(*made_server);

  TlsClientOptions client_options;
  client_options.certificate_authority = TlsCredentialSource::file(cert_path);
  client_options.alpn_protocols = protocols;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client.has_value()) << made_client.error().message();
  TlsContext client_context = std::move(*made_client);

  const std::string payload = "verified-https-model-content";
  const std::string url =
      "https://localhost:" + std::to_string(port) + "/model.bin";
  auto plan = TransferPlan::http(system_allocator(), {url}, {destination});
  ASSERT_TRUE(plan.has_value()) << plan.error().message();
  TransferExecution execution;
  execution.tls_context = &client_context;
  HttpScenario scenario{.payload = payload};
  TransferReport report = run_https_transfer(std::move(*plan), execution,
                                             scenario, port, server_context);

  EXPECT_TRUE(report.published()) << report.status.message();
  EXPECT_TRUE(scenario.status.ok()) << scenario.status.message();
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value()) << actual.error().message();
  EXPECT_EQ(*actual, payload);
  remove_if_present(cert_path);
  remove_if_present(key_path);
  remove_if_present(destination);
}
#endif

TEST(IoTransferHttp, RetainedPartialUsesRangeAndIfRange) {
  constexpr std::uint16_t port = 54572;
  const std::string destination = unique_path("hyle_transfer_http_resume");
  const std::string payload = "0123456789-remote-model-weight-payload";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 5;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 10};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.status.IsCancelled()) << first.status.message();
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::uint64_t prefix = first.checkpoint.partial_bytes();
  ASSERT_GE(prefix, 10u);
  ASSERT_LT(prefix, payload.size());

  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value()) << resumed.error().message();
  FnvVerifier verifier{.expected = FnvVerifier::digest(payload),
                       .expected_size = payload.size()};
  TransferExecution finish;
  finish.verifier = TransferVerifier::from(verifier);
  HttpScenario second_server{.mode = HttpScenarioMode::kRange,
                             .payload = payload,
                             .range_offset = prefix};
  TransferReport second =
      run_http_transfer(std::move(*resumed), finish, second_server, port);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.resumed);
  EXPECT_FALSE(second.restarted);
  EXPECT_EQ(second.bytes_this_attempt, payload.size() - prefix);
  EXPECT_TRUE(second_server.saw_expected_range);
  EXPECT_TRUE(second_server.saw_expected_if_range);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, IgnoredRangeSafelyRestartsFromZero) {
  constexpr std::uint16_t port = 54573;
  const std::string destination = unique_path("hyle_transfer_http_restart");
  const std::string payload = "server-ignores-range-but-full-body-is-safe";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.checkpoint.resumable());
  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value());
  HttpScenario second_server{.mode = HttpScenarioMode::kIgnoreRange,
                             .payload = payload};
  TransferReport second =
      run_http_transfer(std::move(*resumed), {}, second_server, port);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.restarted);
  EXPECT_FALSE(second.resumed);
  EXPECT_EQ(second.bytes_this_attempt, payload.size());
  EXPECT_TRUE(second_server.saw_expected_range);
  EXPECT_TRUE(second_server.saw_expected_if_range);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, CompleteCheckpointAcceptsValidatedRange416) {
  constexpr std::uint16_t port = 54574;
  const std::string destination = unique_path("hyle_transfer_http_complete");
  const std::string payload = "complete-retained-remote-model";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 5;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = payload.size()};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.status.IsCancelled()) << first.status.message();
  ASSERT_TRUE(first.checkpoint.resumable());
  ASSERT_EQ(first.checkpoint.partial_bytes(), payload.size());

  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value());
  FnvVerifier verifier{.expected = FnvVerifier::digest(payload),
                       .expected_size = payload.size()};
  TransferExecution finish;
  finish.verifier = TransferVerifier::from(verifier);
  HttpScenario complete_server{.mode = HttpScenarioMode::kRangeComplete,
                               .payload = payload,
                               .range_offset = payload.size()};
  TransferReport second =
      run_http_transfer(std::move(*resumed), finish, complete_server, port);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.resumed);
  EXPECT_EQ(second.bytes_this_attempt, 0u);
  EXPECT_TRUE(complete_server.saw_expected_range);
  EXPECT_TRUE(complete_server.saw_expected_if_range);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, MalformedRangeAndChangedValidatorAreRejected) {
  constexpr std::uint16_t malformed_port = 54575;
  const std::string malformed_destination =
      unique_path("hyle_transfer_http_bad_range");
  const std::string payload = "remote-model-for-range-validation";
  const std::string malformed_url =
      "http://127.0.0.1:" + std::to_string(malformed_port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan = TransferPlan::http(system_allocator(), {malformed_url},
                                 {malformed_destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first = run_http_transfer(std::move(*plan), interrupted,
                                           first_server, malformed_port);
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::uint64_t prefix = first.checkpoint.partial_bytes();

  options.resume_mismatch = TransferResumeMismatchPolicy::kFail;
  auto malformed_resume =
      TransferPlan::resume(std::move(first.checkpoint), options);
  ASSERT_TRUE(malformed_resume.has_value());
  HttpScenario malformed_server{
      .mode = HttpScenarioMode::kRange,
      .payload = payload,
      .content_range_override = "bytes 0-3/32",
      .range_offset = prefix,
  };
  TransferReport malformed = run_http_transfer(
      std::move(*malformed_resume), {}, malformed_server, malformed_port);
  EXPECT_TRUE(malformed.status.IsDataLoss()) << malformed.status.message();
  ASSERT_TRUE(malformed.checkpoint.resumable());

  auto validator_resume = TransferPlan::resume(std::move(malformed.checkpoint));
  ASSERT_TRUE(validator_resume.has_value());
  HttpScenario changed_validator{
      .mode = HttpScenarioMode::kRange,
      .payload = payload,
      .etag = "\"changed-etag\"",
      .range_offset = prefix,
  };
  TransferReport changed = run_http_transfer(std::move(*validator_resume), {},
                                             changed_validator, malformed_port);
  EXPECT_TRUE(changed.status.IsFailedPrecondition())
      << changed.status.message();
  ASSERT_TRUE(changed.checkpoint.resumable());
  EXPECT_TRUE(changed.checkpoint.discard().ok());
  EXPECT_FALSE(
      hyle::platform::exists(malformed_destination.c_str()).value_or(false));
  remove_if_present(malformed_destination);
}

TEST(IoTransferHttp, RangeResponseMustRepeatTheCheckpointValidator) {
  constexpr std::uint16_t port = 54581;
  const std::string destination =
      unique_path("hyle_transfer_http_missing_validator");
  const std::string payload = "remote-model-with-required-range-validator";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::uint64_t prefix = first.checkpoint.partial_bytes();

  options.resume_mismatch = TransferResumeMismatchPolicy::kFail;
  auto resumed = TransferPlan::resume(std::move(first.checkpoint), options);
  ASSERT_TRUE(resumed.has_value());
  HttpScenario missing_validator{
      .mode = HttpScenarioMode::kRangeWithoutValidator,
      .payload = payload,
      .range_offset = prefix,
  };
  TransferReport report =
      run_http_transfer(std::move(*resumed), {}, missing_validator, port);
  EXPECT_TRUE(report.status.IsFailedPrecondition()) << report.status.message();
  ASSERT_TRUE(report.checkpoint.resumable());
  EXPECT_TRUE(report.checkpoint.discard().ok());
  EXPECT_FALSE(hyle::platform::exists(destination.c_str()).value_or(false));
}

TEST(IoTransferHttp, ChangedValidatorSafelyRestartsWithDefaultPolicy) {
  constexpr std::uint16_t port = 54576;
  const std::string destination =
      unique_path("hyle_transfer_http_changed_validator");
  const std::string payload = "remote-model-with-a-new-validator";
  const std::string url =
      "http://127.0.0.1:" + std::to_string(port) + "/model.bin";
  TransferOptions options;
  options.limits.buffer_bytes = 4;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  StopAfter stopper{.threshold = 8};
  TransferExecution interrupted;
  interrupted.progress = TransferProgressObserver::from(stopper);
  HttpScenario first_server{.mode = HttpScenarioMode::kFull,
                            .payload = payload};
  TransferReport first =
      run_http_transfer(std::move(*plan), interrupted, first_server, port);
  ASSERT_TRUE(first.checkpoint.resumable());
  const std::uint64_t prefix = first.checkpoint.partial_bytes();

  auto resumed = TransferPlan::resume(std::move(first.checkpoint));
  ASSERT_TRUE(resumed.has_value());
  HttpScenario changed_server{
      .mode = HttpScenarioMode::kChangedValidatorThenFull,
      .payload = payload,
      .etag = "\"new-etag\"",
      .range_offset = prefix,
  };
  TransferReport second =
      run_http_transfer(std::move(*resumed), {}, changed_server, port);
  EXPECT_TRUE(second.published()) << second.status.message();
  EXPECT_TRUE(second.restarted);
  EXPECT_FALSE(second.resumed);
  EXPECT_EQ(second.bytes_this_attempt, payload.size());
  EXPECT_TRUE(changed_server.saw_expected_range);
  EXPECT_TRUE(changed_server.saw_expected_if_range);
  auto actual = read_file(destination);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(*actual, payload);
  remove_if_present(destination);
}

TEST(IoTransferHttp, RedirectLoopStopsAtTheConfiguredLimit) {
  constexpr std::uint16_t port = 54577;
  const std::string destination =
      unique_path("hyle_transfer_http_redirect_loop");
  const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/start";
  TransferOptions options;
  options.limits.max_redirects = 1;
  options.partial_on_failure = TransferPartialPolicy::kDiscard;
  auto plan =
      TransferPlan::http(system_allocator(), {url}, {destination}, options);
  ASSERT_TRUE(plan.has_value());
  HttpScenario scenario{.mode = HttpScenarioMode::kRedirectLoop, .payload = {}};
  TransferReport report =
      run_http_transfer(std::move(*plan), {}, scenario, port);
  EXPECT_TRUE(report.status.IsOutOfRange()) << report.status.message();
  EXPECT_EQ(report.disposition, TransferDisposition::kPartialDiscarded);
  EXPECT_FALSE(report.checkpoint.valid());
  EXPECT_FALSE(hyle::platform::exists(destination.c_str()).value_or(false));
}

} // namespace
