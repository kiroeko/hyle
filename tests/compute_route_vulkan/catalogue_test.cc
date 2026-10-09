#include "catalogue_test_support.h"

namespace {
class VulkanCataloguePlain : public ::testing::TestWithParam<cpu_c5_plain_inputs::Case> {};
TEST_P(VulkanCataloguePlain, ActualStorageRoleAndReplay) {
  vulkan_catalogue_test::plain(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CurrentDemand,VulkanCataloguePlain,
    ::testing::ValuesIn(cpu_c5_plain_inputs::cases),
    [](const auto &item){return std::string(item.param.id);});
class VulkanCataloguePacked : public ::testing::TestWithParam<cpu_c5_non_gguf_inputs::Case> {};
TEST_P(VulkanCataloguePacked, ActualPackingColdPreparationAndReplay) {
  vulkan_catalogue_test::packed(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CurrentDemand,VulkanCataloguePacked,
    ::testing::ValuesIn(cpu_c5_non_gguf_inputs::cases),
    [](const auto &item){return std::string(item.param.id);});
} // namespace
