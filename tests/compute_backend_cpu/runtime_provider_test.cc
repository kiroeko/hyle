#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/detail/provider_extension.h"
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

constexpr hyle::compute::ProviderExtensionId kProbeExtensionId{
    0x5445535400000001ULL, 0x455854454e53494fULL, 1};

struct ProbeExtensionInterface final {
  std::uint32_t marker = 0;
};

constexpr ProbeExtensionInterface kProbeExtension{0x48594c45U};

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

const void *
query_provider_extension(void *,
                         hyle::compute::ProviderExtensionId id) noexcept {
  return id == kProbeExtensionId ? &kProbeExtension : nullptr;
}

const hyle::compute::detail::ProviderVTable kProviderVTable{
    .identity = {.id = {0x5445535400000001ULL, 0x50524f4245000001ULL},
                 .name = "test-probe",
                 .version = "1"},
    .create_context = &create_provider_context,
    .destroy_context = &destroy_provider_context,
    .prepare = &prepare_provider,
    .query_extension = &query_provider_extension,
};

hyle::compute::Provider make_provider(ProviderProbe &probe,
                                      std::size_t index) noexcept {
  return hyle::compute::detail::ProviderAccess::make(
      &kProviderVTable, ProviderPayload{&probe, index});
}

hyle::Status observe_probe_extension(void *, const void *interface_value,
                                     void *call_context) noexcept {
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::Status::Ok();
}

hyle::Result<hyle::compute::Completion>
submit_probe_extension(void *, const void *interface_value, void *call_context,
                       hyle::compute::Queue &) noexcept {
  const auto &interface =
      *static_cast<const ProbeExtensionInterface *>(interface_value);
  *static_cast<std::uint32_t *>(call_context) = interface.marker;
  return hyle::compute::Completion::completed();
}

TEST(RuntimeProviders, ExtensionIsVersionedContextLocalAndRetainsItsOwner) {
  ProviderProbe probe;
  hyle::compute::detail::ProviderExtensionRef retained;
  {
    const auto configured = make_provider(probe, 3);
    auto created = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                  configured);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    EXPECT_EQ(created->provider_count(), 1u);
    ASSERT_TRUE(created->provider_identity(0).has_value());
    EXPECT_EQ(created->provider_identity(0)->id, kProviderVTable.identity.id);
    EXPECT_TRUE(created->provider_identity(1).error().IsOutOfRange());
    EXPECT_TRUE(
        hyle::compute::detail::ProviderExtensionAccess::query(*created, 0, {})
            .error()
            .IsInvalidArgument());
    EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::query(
                    *created, 0,
                    hyle::compute::ProviderExtensionId{
                        kProbeExtensionId.namespace_id,
                        kProbeExtensionId.extension_id, 2})
                    .error()
                    .IsUnimplemented());
    retained = hyle::compute::detail::ProviderExtensionAccess::query(
                   *created, 0, kProbeExtensionId)
                   .value();
    EXPECT_EQ(retained.id(), kProbeExtensionId);
    EXPECT_EQ(retained.provider().id, kProviderVTable.identity.id);

    std::uint32_t observed = 0;
    EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::call(
                    retained, &observe_probe_extension, &observed)
                    .ok());
    EXPECT_EQ(observed, kProbeExtension.marker);
  }
  EXPECT_EQ(probe.destroy_count, 0u);
  retained = {};
  EXPECT_EQ(probe.destroy_count, 1u);
}

TEST(RuntimeProviders, ExtensionSubmissionRequiresItsExactOpenContext) {
  ProviderProbe first_probe;
  ProviderProbe second_probe;
  const auto first_provider = make_provider(first_probe, 0);
  const auto second_provider = make_provider(second_probe, 1);
  auto first = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                              first_provider)
                   .value();
  auto second = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                               second_provider)
                    .value();
  auto extension = hyle::compute::detail::ProviderExtensionAccess::query(
                       first, 0, kProbeExtensionId)
                       .value();
  auto first_queue = first.create_queue().value();
  auto second_queue = second.create_queue().value();
  std::uint32_t observed = 0;

  auto wrong = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, second_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_FALSE(wrong.has_value());
  EXPECT_TRUE(wrong.error().IsInvalidArgument());

  auto accepted = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, first_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_TRUE(accepted.has_value()) << accepted.error().message();
  EXPECT_TRUE(accepted->wait().ok());
  EXPECT_EQ(observed, kProbeExtension.marker);

  ASSERT_TRUE(first.close().ok());
  EXPECT_TRUE(hyle::compute::detail::ProviderExtensionAccess::call(
                  extension, &observe_probe_extension, &observed)
                  .IsFailedPrecondition());
  auto closed = hyle::compute::detail::ProviderExtensionAccess::submit(
      extension, first_queue, {}, {}, &submit_probe_extension, &observed);
  ASSERT_FALSE(closed.has_value());
  EXPECT_TRUE(closed.error().IsFailedPrecondition());
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
