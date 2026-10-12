#include <hyle/compute/vulkan_batch.h>
#include <hyle/graph/vulkan.h>
#include <cstdio>

int main() {
  auto context = hyle::compute::vulkan::Context::create({.allow_software_device = true});
  if (!context) {
    char message[768]{};
    context.error().ToString(message, sizeof(message));
    std::fprintf(stderr, "public Vulkan package loader: %s\n", message);
    return 1;
  }
  if (!context->environment().software_device) {
    std::fputs("software-only package cell unexpectedly selected a hardware GPU\n", stderr);
    return 1;
  }
  // This cell proves actual loading/context identity and public batch symbols.
  // It does not execute or certify paged/ragged mathematics on a hardware GPU.
  hyle::compute::vulkan::MaterializedBatchInvocation direct;
  const auto graph_factory = &hyle::graph::vulkan::prepare_materialized_batch;
  (void)graph_factory;
  auto closed = context->close();
  if (!closed.ok()) return 1;
  std::puts("public Vulkan batch SDK software context/loader PASS; actual-GPU math NOT_APPLICABLE");
  return 0;
}
