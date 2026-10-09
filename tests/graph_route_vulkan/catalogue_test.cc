#define HYLE_TEST_GRAPH_ROUTE_VULKAN 1
#include "../compute_route_vulkan/catalogue_test_support.h"

namespace {
class VulkanGraphCataloguePlain : public ::testing::TestWithParam<cpu_c5_plain_inputs::Case> {};
TEST_P(VulkanGraphCataloguePlain, ActualStorageRoleAndReplay) {
  vulkan_catalogue_test::plain(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CurrentDemand,VulkanGraphCataloguePlain,
    ::testing::ValuesIn(cpu_c5_plain_inputs::cases),
    [](const auto &item){return std::string(item.param.id);});
class VulkanGraphCataloguePacked : public ::testing::TestWithParam<cpu_c5_non_gguf_inputs::Case> {};
TEST_P(VulkanGraphCataloguePacked, ActualPackingColdPreparationAndReplay) {
  vulkan_catalogue_test::packed(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CurrentDemand,VulkanGraphCataloguePacked,
    ::testing::ValuesIn(cpu_c5_non_gguf_inputs::cases),
    [](const auto &item){return std::string(item.param.id);});
} // namespace
