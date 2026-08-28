#include <string>
#include <utility>
#include <vector>

#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include <hyle/obs/log.h>

namespace {

struct Capture {
  std::vector<std::string> messages;

  void write(const hyle::obs::LogRecord &record) noexcept {
    messages.emplace_back(record.message.data(), record.message.size());
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
  }

  (void)hyle::obs::log_message(LogLevel::Info, "after scope");
  if (!hyle::obs::set_default_sink_enabled(true).ok())
    return 5;
  return capture.messages.size() == 1 ? 0 : 6;
}
