// Public-API vertical slice: a small REST-shaped service with a real HYLE
// network/scheduler/memory/SIMD path and a deliberately fake model backend.
// It is a consumer probe, not HYLE's future inference-engine implementation.

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/io/endpoint.h"
#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/operation.h"
#include "hyle/io/signal.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/mapping.h"
#include "hyle/obs/metrics.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/host_memory.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/simd/ops.h"
#include "hyle/time/clock.h"
#include "hyle/util/registry.h"
#include "hyle/util/scope_guard.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::container::StringView;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::HttpBodyChunk;
using hyle::io::HttpBodySource;
using hyle::io::HttpHeader;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::SignalInterest;
using hyle::io::SignalSet;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelToken;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr std::uint16_t kDefaultPort = 2048;
constexpr std::size_t kMaxInputValues = 256;
constexpr std::size_t kMaxRequestBytes = 64 * 1024;
constexpr std::size_t kMaxWorkers = 8;
constexpr std::array<float, 8> kBuiltinWeights{
    0.5F, -0.25F, 0.125F, 1.0F, -0.75F, 0.375F, 0.625F, -0.5F};

std::span<const std::byte> bytes(StringView text) noexcept {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

struct Options {
  const char* model_path = nullptr;
  const char* backend = "simd";
  std::uint16_t port = kDefaultPort;
  std::uint64_t max_connections = 0;
  bool metrics = false;
};

bool parse_unsigned(std::string_view text, std::uint64_t& value) noexcept {
  if (text.empty()) return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_options(int argc, char** argv, Options& options) noexcept {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--metrics") {
      options.metrics = true;
    } else if (arg.starts_with("--model=")) {
      options.model_path = argv[index] + std::strlen("--model=");
    } else if (arg.starts_with("--backend=")) {
      options.backend = argv[index] + std::strlen("--backend=");
    } else if (arg.starts_with("--port=")) {
      std::uint64_t port = 0;
      if (!parse_unsigned(arg.substr(std::strlen("--port=")), port) ||
          port == 0 || port > (std::numeric_limits<std::uint16_t>::max)()) {
        return false;
      }
      options.port = static_cast<std::uint16_t>(port);
    } else if (arg.starts_with("--max-connections=")) {
      if (!parse_unsigned(arg.substr(std::strlen("--max-connections=")),
                          options.max_connections)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

void print_status(const char* operation, const Status& status) noexcept {
  char text[512]{};
  status.ToString(text, sizeof(text));
  std::fprintf(stderr, "%s: %s\n", operation, text);
}

struct Model {
  hyle::mem::FileMapping mapping;
  const float* weights = kBuiltinWeights.data();
  std::size_t count = kBuiltinWeights.size();
  StringView name = "builtin-demo";
};

Result<Model> load_model(const char* path) noexcept {
  Model model;
  if (path == nullptr || *path == '\0') return model;

  auto mapped = hyle::mem::FileMapping::open(path);
  if (!mapped) {
    return hyle::make_unexpected(
        std::move(mapped).error().WithContext("mapping model"));
  }
  if (mapped->size() == 0 || mapped->size() % sizeof(float) != 0) {
    return hyle::make_unexpected(Status::InvalidArgument(
        "model file must contain a non-empty native float array"));
  }
  auto storage = mapped->bytes();
  if (!storage) return hyle::make_unexpected(std::move(storage).error());

  model.count = storage->size() / sizeof(float);
  model.weights = reinterpret_cast<const float*>(storage->data());
  model.mapping = std::move(*mapped);
  model.name = StringView(path);
  // Prefetch is an advisory optimization. A platform that cannot honor it may
  // still serve the correctly mapped model.
  (void)model.mapping.prefetch();
  return model;
}

struct Backend {
  StringView name;
  float (*infer)(const float*, const float*, std::size_t) noexcept;
};

float simd_infer(const float* weights, const float* input,
                 std::size_t count) noexcept {
  return hyle::simd::dot(weights, input, count);
}

const Backend kSimdBackend{"simd", &simd_infer};

Result<const Backend*> select_backend(StringView requested) noexcept {
  hyle::util::Registry<const Backend*> registry;
  Status registered = registry.register_creator(
      "simd", []() noexcept -> const Backend* { return &kSimdBackend; },
      hyle::util::RegistryPriority::kPreferred,
      "HYLE runtime-dispatched SIMD dot product");
  if (!registered.ok()) return hyle::make_unexpected(std::move(registered));
  const auto published = registry.freeze();
  const Backend* backend = published.create(requested);
  if (backend == nullptr) {
    return hyle::make_unexpected(Status::NotFound("backend is not registered"));
  }
  return backend;
}

void skip_space(const char*& cursor, const char* end) noexcept {
  while (cursor != end && (*cursor == ' ' || *cursor == '\t' ||
                           *cursor == '\r' || *cursor == '\n')) {
    ++cursor;
  }
}

// Deliberately narrow sample parser. It accepts exactly one JSON object member,
// `values`, whose value is a finite-number array. Production schema handling is
// an application/dependency concern, not an excuse for HYLE to grow a toy JSON
// library.
Result<std::size_t> parse_values(StringView json,
                                 std::span<float> destination) noexcept {
  const char* cursor = json.data();
  const char* const end = cursor + json.size();
  skip_space(cursor, end);
  if (cursor == end || *cursor++ != '{') {
    return hyle::make_unexpected(Status::InvalidArgument("expected JSON object"));
  }
  skip_space(cursor, end);
  constexpr std::string_view kMember = "\"values\"";
  if (static_cast<std::size_t>(end - cursor) < kMember.size() ||
      std::memcmp(cursor, kMember.data(), kMember.size()) != 0) {
    return hyle::make_unexpected(
        Status::InvalidArgument("expected a values member"));
  }
  cursor += kMember.size();
  skip_space(cursor, end);
  if (cursor == end || *cursor++ != ':') {
    return hyle::make_unexpected(Status::InvalidArgument("expected ':'"));
  }
  skip_space(cursor, end);
  if (cursor == end || *cursor++ != '[') {
    return hyle::make_unexpected(Status::InvalidArgument("expected value array"));
  }

  std::size_t count = 0;
  for (;;) {
    skip_space(cursor, end);
    if (cursor == end) {
      return hyle::make_unexpected(
          Status::InvalidArgument("unterminated value array"));
    }
    if (*cursor == ']') {
      ++cursor;
      break;
    }
    if (count == destination.size()) {
      return hyle::make_unexpected(
          Status::ResourceExhausted("too many input values"));
    }
    float value = 0.0F;
    const auto parsed =
        std::from_chars(cursor, end, value, std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr == cursor ||
        !std::isfinite(value)) {
      return hyle::make_unexpected(
          Status::InvalidArgument("input value is not a finite number"));
    }
    destination[count++] = value;
    cursor = parsed.ptr;
    skip_space(cursor, end);
    if (cursor != end && *cursor == ',') {
      ++cursor;
      continue;
    }
    if (cursor == end || *cursor != ']') {
      return hyle::make_unexpected(
          Status::InvalidArgument("expected ',' or ']'"));
    }
  }
  skip_space(cursor, end);
  if (cursor == end || *cursor++ != '}') {
    return hyle::make_unexpected(Status::InvalidArgument("expected object end"));
  }
  skip_space(cursor, end);
  if (cursor != end || count == 0) {
    return hyle::make_unexpected(
        Status::InvalidArgument("values must be the complete non-empty object"));
  }
  return count;
}

struct ServiceState {
  Scheduler* scheduler;
  const Model* model;
  const Backend* backend;
  CancelToken cancel;
  std::uint64_t max_connections;
  bool metrics_enabled;
  hyle::obs::ConcurrentCounter<std::uint64_t> accepted_connections;
  hyle::obs::ConcurrentCounter<std::uint64_t> requests;
  hyle::obs::ConcurrentCounter<std::uint64_t> inferences;
  hyle::obs::ConcurrentCounter<std::uint64_t> errors;
  hyle::obs::ConcurrentGauge<std::int64_t> active_connections;
};

OperationOptions request_budget(const ServiceState& state) noexcept {
  return {.deadline = hyle::time::deadline_after(std::chrono::seconds(30)),
          .cancel = &state.cancel};
}

Status format_score(char* output, std::size_t capacity, float score,
                    StringView backend, std::size_t count,
                    std::size_t& size) noexcept {
  const int written = std::snprintf(
      output, capacity,
      "{\"score\":%.9g,\"backend\":\"%.*s\",\"values\":%zu}\n",
      static_cast<double>(score), static_cast<int>(backend.size()),
      backend.data(), count);
  if (written < 0 || static_cast<std::size_t>(written) >= capacity) {
    return Status::ResourceExhausted("response buffer is too small");
  }
  size = static_cast<std::size_t>(written);
  return Status::Ok();
}

struct ScoreStream {
  std::array<char, 256> score_event{};
  std::size_t score_size = 0;
  unsigned index = 0;

  Task<Result<HttpBodyChunk>> next(AllocatorRef allocator,
                                   OperationOptions options) noexcept {
    return next_impl(allocator, this, options);
  }

 private:
  static Task<Result<HttpBodyChunk>> next_impl(AllocatorRef,
                                               ScoreStream* self,
                                               OperationOptions) noexcept {
    static constexpr char kDone[] = "data: [DONE]\n\n";
    if (self->index == 0) {
      ++self->index;
      co_return HttpBodyChunk{
          {reinterpret_cast<const std::byte*>(self->score_event.data()),
           self->score_size},
          {}, false};
    }
    if (self->index == 1) {
      ++self->index;
      co_return HttpBodyChunk{
          {reinterpret_cast<const std::byte*>(kDone), sizeof(kDone) - 1}, {},
          true};
    }
    co_return hyle::make_unexpected(
        Status::FailedPrecondition("SSE source read after completion"));
  }
};

Task<Status> serve_connection(AllocatorRef allocator, ServiceState* state,
                              TcpStream stream) noexcept {
  state->active_connections.add(1);
  hyle::util::ScopeGuard release_active{[state]() noexcept {
    state->active_connections.subtract(1);
  }};

  auto made = HttpServerConnection::create(allocator, std::move(stream));
  if (!made) co_return std::move(made).error();
  HttpServerConnection connection = std::move(*made);

  for (;;) {
    const OperationOptions operation = request_budget(*state);
    auto next = co_await connection.read_request(operation);
    if (!next) {
      if (state->cancel.is_cancelled() && next.error().IsCancelled()) break;
      co_return std::move(next).error();
    }
    if (!next->has_value()) break;
    auto incoming = std::move(next->value());
    auto body = co_await incoming.body.read_all(kMaxRequestBytes, operation);
    if (!body) co_return std::move(body).error();
    state->requests.add();

    static constexpr HttpHeader kJsonHeaders[] = {
        {"Content-Type", "application/json"}};
    static constexpr HttpHeader kMetricsHeaders[] = {
        {"Content-Type", "text/plain; version=0.0.4"}};
    static constexpr HttpHeader kSseHeaders[] = {
        {"Content-Type", "text/event-stream"},
        {"Cache-Control", "no-cache"}};

    HttpResponseHead response;
    response.keep_alive = incoming.head.keep_alive();
    std::array<char, 1024> output{};
    std::size_t output_size = 0;
    Status sent;

    const StringView method = incoming.head.method();
    const StringView path = incoming.head.path();
    if (method == "GET" && path == "/health") {
      response.headers = kJsonHeaders;
      constexpr StringView kHealth = "{\"status\":\"ok\"}\n";
      sent = co_await connection.send_response(response, bytes(kHealth),
                                               operation);
    } else if (method == "GET" && path == "/v1/models") {
      response.headers = kJsonHeaders;
      const int written = std::snprintf(
          output.data(), output.size(),
          "{\"data\":[{\"id\":\"%.*s\",\"weights\":%zu}]}\n",
          static_cast<int>(state->model->name.size()),
          state->model->name.data(), state->model->count);
      if (written < 0 || static_cast<std::size_t>(written) >= output.size()) {
        co_return Status::ResourceExhausted("model response is too large");
      }
      output_size = static_cast<std::size_t>(written);
      sent = co_await connection.send_response(
          response,
          {reinterpret_cast<const std::byte*>(output.data()), output_size},
          operation);
    } else if (method == "GET" && path == "/metrics" &&
               state->metrics_enabled) {
      response.headers = kMetricsHeaders;
      const int written = std::snprintf(
          output.data(), output.size(),
          "hyle_demo_connections_accepted %llu\n"
          "hyle_demo_connections_active %lld\n"
          "hyle_demo_requests_total %llu\n"
          "hyle_demo_inferences_total %llu\n"
          "hyle_demo_errors_total %llu\n",
          static_cast<unsigned long long>(
              state->accepted_connections.value()),
          static_cast<long long>(state->active_connections.value()),
          static_cast<unsigned long long>(state->requests.value()),
          static_cast<unsigned long long>(state->inferences.value()),
          static_cast<unsigned long long>(state->errors.value()));
      if (written < 0 || static_cast<std::size_t>(written) >= output.size()) {
        co_return Status::ResourceExhausted("metrics response is too large");
      }
      output_size = static_cast<std::size_t>(written);
      sent = co_await connection.send_response(
          response,
          {reinterpret_cast<const std::byte*>(output.data()), output_size},
          operation);
    } else if (method == "POST" &&
               (path == "/v1/infer" || path == "/v1/stream")) {
      std::array<float, kMaxInputValues> input{};
      auto parsed = parse_values(*body, input);
      if (!parsed || *parsed > state->model->count) {
        response.status = 400;
        response.keep_alive = false;
        response.headers = kJsonHeaders;
        constexpr StringView kBadRequest =
            "{\"error\":\"invalid values payload or model extent\"}\n";
        state->errors.add();
        sent = co_await connection.send_response(
            response, bytes(kBadRequest), operation);
      } else {
        const float score = state->backend->infer(
            state->model->weights, input.data(), *parsed);
        state->inferences.add();
        if (path == "/v1/infer") {
          response.headers = kJsonHeaders;
          Status formatted =
              format_score(output.data(), output.size(), score,
                           state->backend->name, *parsed, output_size);
          if (!formatted.ok()) co_return formatted;
          sent = co_await connection.send_response(
              response,
              {reinterpret_cast<const std::byte*>(output.data()), output_size},
              operation);
        } else {
          response.headers = kSseHeaders;
          ScoreStream source;
          const int written = std::snprintf(
              source.score_event.data(), source.score_event.size(),
              "data: {\"score\":%.9g}\n\n", static_cast<double>(score));
          if (written < 0 ||
              static_cast<std::size_t>(written) >= source.score_event.size()) {
            co_return Status::ResourceExhausted("SSE response is too large");
          }
          source.score_size = static_cast<std::size_t>(written);
          sent = co_await connection.send_response(
              response, HttpBodySource::from(source), operation);
        }
      }
    } else {
      response.status = 404;
      response.keep_alive = false;
      response.headers = kJsonHeaders;
      constexpr StringView kNotFound = "{\"error\":\"not found\"}\n";
      sent = co_await connection.send_response(response, bytes(kNotFound),
                                               operation);
    }

    if (!sent.ok()) co_return sent;
    if (!response.keep_alive) break;
  }

  const OperationOptions close_budget{
      .deadline = hyle::time::deadline_after(std::chrono::seconds(1))};
  co_return co_await connection.close(close_budget);
}

Task<void> report_service_task(AllocatorRef, Task<Status> task,
                               ServiceState* state,
                               bool report_error) noexcept {
  try {
    Status status = co_await task;
    if (report_error && !status.ok()) {
      state->errors.add();
      print_status("service task", status);
    }
  } catch (...) {
    state->errors.add();
    std::fputs("service task: uncaught exception\n", stderr);
  }
  co_return;
}

Status launch_pinned(Scheduler& scheduler, unsigned owner, Task<Status> task,
                     ServiceState* state,
                     bool report_error = true) noexcept {
  if (!task.valid()) {
    return Status::ResourceExhausted("coroutine frame allocation failed");
  }
  auto handle = hyle::sched::spawn_task_on(
      scheduler, owner, system_allocator(),
      report_service_task(system_allocator(), std::move(task), state,
                          report_error));
  // Admission failures are represented by an already-completed handle. A task
  // that genuinely completed before this check has already reported its own
  // result, so joining it is equally correct. Running tasks intentionally
  // detach: Scheduler::drain still owns their lifetime.
  if (!handle.is_finished()) {
    handle.detach();
    return Status::Ok();
  }
  auto launched = handle.join();
  if (!launched) return std::move(launched).error();
  return Status::Ok();
}

Task<Status> accept_loop(AllocatorRef allocator, TcpListener* listener,
                         ServiceState* state) noexcept {
  std::uint64_t accepted = 0;
  unsigned next_owner = 0;
  while (!state->cancel.is_cancelled() &&
         (state->max_connections == 0 ||
          accepted < state->max_connections)) {
    const unsigned owner = next_owner++ % state->scheduler->num_workers();
    auto stream = co_await listener->accept(
        owner, {.cancel = &state->cancel});
    if (!stream) {
      if (state->cancel.is_cancelled() && stream.error().IsCancelled()) break;
      co_return std::move(stream).error();
    }
    ++accepted;
    state->accepted_connections.add();
    Status launched = launch_pinned(
        *state->scheduler, owner,
        serve_connection(allocator, state, std::move(*stream)), state);
    if (!launched.ok()) co_return launched;
  }
  co_return Status::Ok();
}

Task<Status> signal_loop(AllocatorRef, SignalSet* signals,
                         ServiceState* state) noexcept {
  auto signal = co_await signals->wait();
  if (!signal) co_return std::move(signal).error();
  state->cancel.cancel();
  state->scheduler->begin_shutdown();
  co_return Status::Ok();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    std::fprintf(stderr,
                 "usage: %s [--port=2048] [--metrics] [--model=FLOATS] "
                 "[--backend=simd] [--max-connections=N]\n",
                 argv[0]);
    return 2;
  }

  auto model_result = load_model(options.model_path);
  if (!model_result) {
    print_status("load model", model_result.error());
    return 1;
  }
  Model model = std::move(*model_result);

  auto backend_result = select_backend(StringView(options.backend));
  if (!backend_result) {
    print_status("select backend", backend_result.error());
    return 1;
  }

  std::size_t workers = 1;
  auto capacity = hyle::platform::process_cpu_capacity();
  if (capacity) {
    workers = std::clamp<std::size_t>(capacity->parallelism_ceiling(), 1,
                                      kMaxWorkers);
  }
  auto memory = hyle::platform::host_memory_info();

  Status scheduler_status;
  Scheduler scheduler(static_cast<unsigned>(workers), system_allocator(), 1024,
                      &scheduler_status);
  if (!scheduler_status.ok()) {
    print_status("create scheduler", scheduler_status);
    return 1;
  }
  IoContext io(scheduler);
  if (!io.valid()) {
    print_status("create io context", io.status());
    scheduler.shutdown();
    return 1;
  }

  auto endpoint = Endpoint::parse("127.0.0.1", options.port);
  if (!endpoint) {
    print_status("parse endpoint", endpoint.error());
    scheduler.shutdown();
    return 1;
  }
  auto listener_result = TcpListener::bind(io, 0, *endpoint);
  if (!listener_result) {
    print_status("bind listener", listener_result.error());
    scheduler.shutdown();
    return 1;
  }
  TcpListener listener = std::move(*listener_result);

  auto cancel_result = CancelToken::try_create_root(system_allocator());
  if (!cancel_result) {
    print_status("create cancellation root", cancel_result.error());
    scheduler.shutdown();
    return 1;
  }
  ServiceState state{&scheduler,
                     &model,
                     *backend_result,
                     std::move(*cancel_result),
                     options.max_connections,
                     options.metrics};

  std::optional<SignalSet> signals;
  if (options.max_connections == 0) {
    auto created = SignalSet::create(
        io, 0, SignalInterest::kInterrupt | SignalInterest::kTerminate);
    if (!created) {
      print_status("create signal set", created.error());
      scheduler.shutdown();
      return 1;
    }
    signals.emplace(std::move(*created));
  }

  Status launched = launch_pinned(
      scheduler, 0, accept_loop(system_allocator(), &listener, &state), &state);
  if (!launched.ok()) {
    print_status("launch accept loop", launched);
    scheduler.shutdown();
    return 1;
  }
  if (signals) {
    launched = launch_pinned(
        scheduler, 0, signal_loop(system_allocator(), &*signals, &state),
        &state);
    if (!launched.ok()) {
      print_status("launch signal loop", launched);
      state.cancel.cancel();
      scheduler.shutdown();
      return 1;
    }
  }

  std::printf("HYLE inference-service slice listening on http://127.0.0.1:%u\n",
              static_cast<unsigned>(options.port));
  std::printf("backend=%.*s weights=%zu workers=%zu metrics=%s\n",
              static_cast<int>((*backend_result)->name.size()),
              (*backend_result)->name.data(), model.count, workers,
              options.metrics ? "on" : "off");
  if (memory) {
    std::printf("host-available=%llu bytes\n",
                static_cast<unsigned long long>(
                    memory->effective_available_bytes()));
  }

  scheduler.drain();
  scheduler.shutdown();
  return state.errors.value() == 0 ? 0 : 1;
}
