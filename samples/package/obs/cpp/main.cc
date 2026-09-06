#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include <hyle/obs/log.h>
#include <hyle/obs/metrics.h>

namespace {

struct Capture {
  std::vector<std::string> messages;
  std::uint64_t request_id = 0;
  double tokens_per_second = 0.0;
  bool saw_progress = false;

  void write(const hyle::obs::LogRecord &record) noexcept {
    messages.emplace_back(record.message.data(), record.message.size());
    if (record.category == "inference.decode" &&
        record.event == "slot.progress" && record.attributes.size() == 1 &&
        record.correlation_attributes.size() == 1) {
      tokens_per_second = record.attributes.front().value.as_floating_point();
      request_id =
          record.correlation_attributes.front().value.as_unsigned_integer();
      saw_progress = record.timestamp.has_value();
    }
  }
};

} // namespace

int main() {
  using hyle::obs::LogLevel;
  if (!hyle::obs::set_default_sink_enabled(false).ok())
    return 1;
  hyle::obs::set_level(LogLevel::Info);

  int evaluated = 0;
  HYLE_LOG_DEBUG("{}", ++evaluated);
  if (evaluated != 0)
    return 2;

  Capture capture;
  {
    auto registration = hyle::obs::SinkRegistration::create(capture);
    if (!registration)
      return 3;
    HYLE_LOG_INFO("loaded {} kernels", 3);
    if (capture.messages.size() != 1 ||
        capture.messages.front() != "loaded 3 kernels") {
      return 4;
    }

    const hyle::obs::LogAttribute attributes[] = {
        {"tokens_per_second", 72.5},
    };
    const hyle::obs::LogAttribute correlation[] = {
        {"request_id", std::uint64_t{17}},
    };
    const hyle::obs::LogRecord progress{
        .level = LogLevel::Info,
        .category = "inference.decode",
        .event = "slot.progress",
        .message = "decode progress",
        .attributes = attributes,
        .correlation_attributes = correlation,
        .source =
            hyle::obs::LogSourceLocation::from(std::source_location::current()),
        .timestamp = hyle::time::unix_timestamp(1'700'000'000, 123),
    };
    if (!hyle::obs::log_record(progress) || !capture.saw_progress ||
        capture.request_id != 17 || capture.tokens_per_second != 72.5) {
      return 5;
    }
  }

  (void)hyle::obs::log_message(LogLevel::Info, "after scope");
  if (!hyle::obs::set_default_sink_enabled(true).ok())
    return 6;
  if (capture.messages.size() != 2)
    return 7;

  hyle::obs::Counter<> decoded_tokens;
  hyle::obs::Gauge<> queue_depth;
  auto buckets = hyle::obs::FixedBuckets<3>::create({0.001, 0.01, 0.1});
  if (!buckets)
    return 8;
  hyle::obs::FixedHistogram decode_latency(*buckets);

  decoded_tokens.add(32);
  queue_depth.set(3.0);
  if (!decode_latency.observe(0.004, 32))
    return 9;
  const auto snapshot = decode_latency.snapshot();
  return decoded_tokens.value() == 32 && queue_depth.value() == 3.0 &&
                 snapshot.count() == 32 &&
                 *snapshot.quantile_upper_bound(0.99) == 0.01
             ? 0
             : 10;
}
