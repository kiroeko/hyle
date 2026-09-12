#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "provider_internal.h"

namespace {

struct ProviderProbe final {
  std::array<std::size_t, 16> created{};
  std::array<std::size_t, 16> destroyed{};
  std::array<std::size_t, 16> prepared{};
  std::size_t create_count = 0;
  std::size_t destroy_count = 0;
  std::size_t prepare_count = 0;
  std::size_t fail_index = static_cast<std::size_t>(-1);
};

struct ProviderPayload final {
  ProviderProbe *probe = nullptr;
  std::size_t index = 0;
};

hyle::Status create_provider_context(const hyle::compute::Provider &provider,
                                     hyle::compute::Device, void *,
                                     hyle::mem::AllocatorRef,
                                     void **context) noexcept {
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->created[payload.probe->create_count++] = payload.index;
  if (payload.index == payload.probe->fail_index) {
    *context = nullptr;
    return hyle::Status::InvalidArgument("injected provider failure");
  }
  *context = const_cast<hyle::compute::Provider *>(&provider);
  return hyle::Status::Ok();
}

void destroy_provider_context(void *context, hyle::mem::AllocatorRef) noexcept {
  const auto &provider = *static_cast<hyle::compute::Provider *>(context);
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->destroyed[payload.probe->destroy_count++] = payload.index;
}

hyle::Status
prepare_provider(void *context, hyle::compute::Operation,
                 std::span<const hyle::compute::TensorDesc>,
                 std::span<const hyle::compute::TensorDesc>,
                 hyle::mem::AllocatorRef,
                 hyle::compute::detail::ProviderPrepared *) noexcept {
  const auto &provider = *static_cast<hyle::compute::Provider *>(context);
  const auto payload =
      hyle::compute::detail::ProviderAccess::payload<ProviderPayload>(provider);
  payload.probe->prepared[payload.probe->prepare_count++] = payload.index;
  return hyle::Status::Unimplemented("probe provider declines operation");
}

const hyle::compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = {0x5445535400000001ULL, 0x50524f4245000001ULL},
                 .name = "test-probe",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .prepare = &prepare_provider,
};

hyle::compute::Provider make_provider(ProviderProbe &probe,
                                      std::size_t index) noexcept {
  return hyle::compute::detail::ProviderAccess::make(
      &kProviderVTable, ProviderPayload{&probe, index});
}

TEST(RuntimeProviders, ContextCopiesAnArbitraryOrderedSequence) {
  ProviderProbe probe;
  std::array<hyle::compute::Provider, 6> providers{};
  for (std::size_t index = 0; index < providers.size(); ++index)
    providers[index] = make_provider(probe, index);

  {
    auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                  {.providers = providers});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    providers.fill({});

    const std::array<std::int64_t, 1> shape{8};
    auto description = hyle::compute::TensorDesc::dense(
        hyle::compute::DataType::kFloat32, shape);
    ASSERT_TRUE(description.has_value());
    auto prepared = hyle::compute::prepare(*created, hyle::compute::RmsNorm{},
                                           *description, *description);
    ASSERT_FALSE(prepared.has_value());
    EXPECT_TRUE(prepared.error().IsUnimplemented());
    ASSERT_EQ(probe.prepare_count, providers.size());
    for (std::size_t index = 0; index < providers.size(); ++index)
      EXPECT_EQ(probe.prepared[index], index);
  }

  ASSERT_EQ(probe.destroy_count, providers.size());
  for (std::size_t index = 0; index < providers.size(); ++index)
    EXPECT_EQ(probe.destroyed[index], providers.size() - index - 1);
}

TEST(RuntimeProviders, ContextCreationRollsBackInReverseOrder) {
  ProviderProbe probe;
  probe.fail_index = 2;
  const std::array providers{make_provider(probe, 0), make_provider(probe, 1),
                             make_provider(probe, 2), make_provider(probe, 3)};

  auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                {.providers = providers});
  ASSERT_FALSE(created.has_value());
  EXPECT_TRUE(created.error().IsInvalidArgument());
  ASSERT_EQ(probe.create_count, 3u);
  EXPECT_EQ(probe.created[0], 0u);
  EXPECT_EQ(probe.created[1], 1u);
  EXPECT_EQ(probe.created[2], 2u);
  ASSERT_EQ(probe.destroy_count, 2u);
  EXPECT_EQ(probe.destroyed[0], 1u);
  EXPECT_EQ(probe.destroyed[1], 0u);
}

} // namespace
