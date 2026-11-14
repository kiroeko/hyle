#include <gtest/gtest.h>

#include <array>

#include "hyle/core/environment.h"
#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(push, 0)
#pragma warning(disable : 4576)
#endif
#include "iree/hal/drivers/vulkan/device_plan.h"
#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(pop)
#endif

namespace {
TEST(IreeQueuePlan, WrappedRolesShareAFrontierExactlyWhenNativeQueuesMatch) {
  iree_hal_vulkan_physical_device_snapshot_t snapshot{};
  snapshot.properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  snapshot.properties2.properties.apiVersion = VK_API_VERSION_1_3;
  snapshot.features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  snapshot.features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  snapshot.features12.timelineSemaphore = VK_TRUE;
  snapshot.features12.scalarBlockLayout = VK_TRUE;
  snapshot.features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  snapshot.features13.synchronization2 = VK_TRUE;
  std::array<VkQueueFamilyProperties2, 2> families{};
  for (auto &family : families) {
    family.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    family.queueFamilyProperties.queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    family.queueFamilyProperties.queueCount = 2;
    family.queueFamilyProperties.timestampValidBits = 64;
  }
  snapshot.queue_family_count = static_cast<uint32_t>(families.size());
  snapshot.queue_families = families.data();
  // Inferred transfer, explicit alias, another index, another family. Merely
  // having spare hardware queues does not make an imported alias independent.
  for (unsigned mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    iree_hal_vulkan_device_options_t options;
    iree_hal_vulkan_device_options_initialize(&options);
    iree_hal_vulkan_external_device_params_t external{};
    external.enabled_features = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE;
    external.compute_queue_set = {0, 1};
    if (mode) external.transfer_queue_set = {mode == 3 ? 1u : 0u, mode == 2 ? 2ull : 1ull};
    iree_hal_vulkan_device_plan_t plan;
    auto status = iree_hal_vulkan_device_plan_initialize_for_wrap(&snapshot, &options, &external, &plan);
    const auto code = iree_status_code(status);
    iree_status_ignore(status);
    ASSERT_EQ(code, IREE_STATUS_OK);
    const bool independent = mode >= 2;
    EXPECT_EQ(plan.queue_assignment.queue_count, independent ? 2u : 1u);
    EXPECT_EQ(plan.queue_assignment.compute.affinity, 1u);
    EXPECT_EQ(plan.queue_assignment.transfer.affinity, independent ? 2u : 1u);
    EXPECT_EQ(plan.queue_assignment.transfer.family_index, mode == 3 ? 1u : 0u);
    EXPECT_EQ(plan.queue_assignment.transfer.queue_index, mode == 2 ? 1u : 0u);
  }
}
} // namespace
