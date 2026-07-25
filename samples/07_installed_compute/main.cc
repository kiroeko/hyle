#include <hyle/compute/device.h>

#include <array>
#include <cstdio>

int main() {
  std::array<hyle::compute::Device, 16> devices{};
  const std::size_t count = hyle::compute::enumerate(devices.data(), devices.size());
  if (count == 0) {
    std::fputs("installed HYLE compute package exposed no host device\n", stderr);
    return 1;
  }
  const auto& first = devices.front();
  std::printf("installed HYLE compute package: %zu device(s), first=%s/%s\n",
              count, first.backend(), first.name());
  return 0;
}
