#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include <hyle/process/process.h>

int main(int argc, char** argv) {
  constexpr std::string_view kChildArgument = "--hyle-package-child";
  if (argc == 2 && std::string_view(argv[1]) == kChildArgument) return 23;
  if (argc < 1 || argv[0] == nullptr) return 1;

  auto created = hyle::process::ProcessRuntime::create();
  if (!created) return 2;
  auto runtime = *std::move(created);

  const std::array<std::string_view, 1> arguments{kChildArgument};
  hyle::process::ProcessStartOptions options;
  options.executable = argv[0];
  options.arguments = arguments;
  options.stdin_stream = hyle::process::ProcessStdio::null();
  options.stdout_stream = hyle::process::ProcessStdio::null();
  options.stderr_stream = hyle::process::ProcessStdio::null();

  auto spawned = runtime.spawn(options);
  if (!spawned) return 3;
  auto child = *std::move(spawned);
  auto exited = child.wait();
  if (!exited || exited->kind != hyle::process::ProcessExitKind::kExited ||
      exited->exit_code != 23) {
    return 4;
  }
  return runtime.shutdown().ok() ? 0 : 5;
}
