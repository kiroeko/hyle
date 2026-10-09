#include "catalogue_test_support.h"
#include "catalogue_gguf_test_support.h"
#include "current_state_test_support.h"
#include "current_state_tests.h"
#include "catalogue_parameter_tests.h"

namespace {
class VulkanCatalogueGguf : public ::testing::TestWithParam<cpu_c5_inputs::Case> {};
TEST_P(VulkanCatalogueGguf, OriginalBytesActualRoleWidthAndReplay) {
  vulkan_catalogue_gguf::check(GetParam());
}
INSTANTIATE_TEST_SUITE_P(CurrentDemand,VulkanCatalogueGguf,
    ::testing::ValuesIn(cpu_c5_inputs::cases),
    [](const auto &item){return std::string(item.param.id);});
TEST(VulkanCatalogueGguf, All104ExactArtifactIdentitiesRunActualProjection) {
  vulkan_catalogue_gguf::identities();
}
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
