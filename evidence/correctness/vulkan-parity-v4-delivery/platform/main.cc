#include "hyle/compute/vulkan.h"
#include "hyle/compute/detail/vulkan_plan.h"
#include "hyle/graph/vulkan.h"
#include <iostream>
#if defined(V4_EXPECT_VULKAN)
#ifndef HYLE_COMPUTE_VULKAN
#error Selected Vulkan SDK did not expose the route
#endif
#else
#ifdef HYLE_COMPUTE_VULKAN
#error Disabled Vulkan route leaked a feature definition
#endif
#endif
int main() {
#if defined(V4_EXPECT_VULKAN)
  namespace v = hyle::compute::vulkan;
  auto ordinary = v::Context::create();
  if (ordinary) {
    std::cerr << "Expected no native GPU; ordinary context must reject software-only device\n";
    return 1;
  }
  if (!ordinary.error().IsUnavailable() && !ordinary.error().IsUnimplemented()) {
    std::cerr << ordinary.error().message() << '\n'; return 2;
  }
  auto diagnostic = v::Context::create({.allow_software_device = true});
  if (!diagnostic) { std::cerr << diagnostic.error().message() << '\n'; return 3; }
  if (!diagnostic->environment().software_device) return 4;
  const auto &e = diagnostic->environment();
  std::cout << "SOFTWARE_ONLY " << e.device_name << " loader=" << e.loader_version
            << " api=" << e.api_version << " default=REJECTED numeric_gpu=UNAVAILABLE\n";
  if (!diagnostic->close().ok()) return 5;
#else
  auto builder = hyle::graph::GraphBuilder::create();
  if (!builder) return 6;
  std::cout << "VULKAN_OFF headers/common_graph=PASS\n";
#endif
}
