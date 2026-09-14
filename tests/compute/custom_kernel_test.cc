#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include "custom_kernel_fixture.h"

namespace {
using namespace custom_test;

struct FaultAllocator final {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t attempts = 0, fail = SIZE_MAX, live = 0;
  hyle::mem::AllocatorRef ref() noexcept {
    return {[](void *self, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (s.attempts++ == s.fail)
                return nullptr;
              auto *p = s.backing.allocate(size, alignment);
              s.live += p != nullptr;
              return p;
            },
            [](void *self, void *p, std::size_t size,
               std::size_t alignment) noexcept {
              auto &s = *static_cast<FaultAllocator *>(self);
              s.live -= p != nullptr;
              s.backing.deallocate(p, size, alignment);
            },
            [](void *self, void *p, std::size_t old_size, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (size != 0 && s.attempts++ == s.fail)
                return nullptr;
              auto *next = s.backing.reallocate(p, old_size, size, alignment);
              if (size == 0)
                s.live -= p != nullptr;
              else if (next != nullptr && p == nullptr)
                ++s.live;
              return next;
            },
            this};
  }
};

TEST(CustomKernel, OwnsMoveOnlyFactoryAndCopiesIdentityText) {
  Probe probe;
  {
    Factory factory{std::make_unique<Config>(probe)};
    std::string name = "owned-name", version = "owned-version";
    auto created = make_custom_provider({identity.id, name, version},
                                        cpu::kBackendId, std::move(factory));
    ASSERT_TRUE(created);
    EXPECT_EQ(factory.config, nullptr);
    name.assign(500, 'x');
    version.clear();
    Provider copy = *created;
    Provider assigned;
    assigned = copy;
    assigned = assigned;
    copy = std::move(copy);
    Provider moved = std::move(*created);
    EXPECT_FALSE(*created);
    *created = {};
    moved = {};
    EXPECT_EQ(copy.name(), "owned-name");
    EXPECT_EQ(assigned.version(), "owned-version");
    EXPECT_EQ(probe.configurations, 1);
  }
  EXPECT_EQ(probe.configurations, 0);
}

TEST(CustomKernel, InvalidOrAllocationFailureDoesNotConsumeNamedFactory) {
  Probe probe;
  Factory factory{std::make_unique<Config>(probe)};
  auto invalid = make_custom_provider({}, cpu::kBackendId, std::move(factory));
  EXPECT_FALSE(invalid);
  ASSERT_NE(factory.config, nullptr);
  invalid = make_custom_provider(identity, {}, std::move(factory));
  EXPECT_FALSE(invalid);
  ASSERT_NE(factory.config, nullptr);
  FaultAllocator allocator;
  allocator.fail = 0;
  auto failed = make_custom_provider(identity, cpu::kBackendId,
                                     std::move(factory), allocator.ref());
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsResourceExhausted());
  ASSERT_NE(factory.config, nullptr);
  EXPECT_EQ(allocator.live, 0u);
  allocator.fail = SIZE_MAX;
  {
    auto recovered = make_custom_provider(identity, cpu::kBackendId,
                                          std::move(factory), allocator.ref());
    ASSERT_TRUE(recovered);
    EXPECT_EQ(factory.config, nullptr);
    EXPECT_EQ(allocator.live, 1u);
  }
  EXPECT_EQ(allocator.live, 0u);
  EXPECT_EQ(probe.configurations, 0);
}

TEST(CustomKernel, ConstCopyableFactoryIsCapturedOnceNotCopiedByHandles) {
  struct CopyableFactory final {
    std::shared_ptr<const Factory> backing;
    Result<KernelProposal>
    propose(const KernelRequest &request, std::span<ResourceRequirement> inputs,
            std::span<ResourceRequirement> outputs) const noexcept {
      return backing->propose(request, inputs, outputs);
    }
    KernelPreparation
    prepare(const KernelPrepareRequest &request,
            hyle::mem::AllocatorRef allocator) const noexcept {
      return backing->prepare(request, allocator);
    }
  };
  Probe probe;
  const CopyableFactory factory{
      std::make_shared<Factory>(Factory{std::make_unique<Config>(probe)})};
  ASSERT_EQ(factory.backing.use_count(), 1);
  {
    auto first =
        make_custom_provider(identity, cpu::kBackendId, factory).value();
    EXPECT_EQ(factory.backing.use_count(), 2);
    auto second = first;
    auto context = Context::create(cpu::device(), second).value();
    EXPECT_EQ(factory.backing.use_count(), 2);
  }
  EXPECT_EQ(factory.backing.use_count(), 1);
}

TEST(CustomKernel, SerialArtifactDeclaresDirectCallerCompletionConstraint) {
  Probe probe;
  probe.concurrent = false;
  Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  auto queue = context.create_queue().value();
  auto prepared = prepare(context, Operation::custom(Affine{3}),
                          std::array{tensors.in}, std::array{tensors.out})
                      .value();
  EXPECT_FALSE(prepared.runtime_properties().supports_concurrent_invocations);
  auto first = prepared.create_invocation().value();
  auto second = prepared.create_invocation().value();
  for (auto *invocation : {&first, &second}) {
    auto done = invocation->submit(queue, tensors.in, tensors.out);
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->wait().ok());
  }
  EXPECT_EQ(tensors.read(), (std::array<float, 4>{5, 7, 9, 11}));
}

TEST(CustomKernel, WrongBackendRejectedAndProviderOwnerRemainsValid) {
  Probe probe;
  auto wrong = make_custom_provider(identity, BackendId{1, 2},
                                    Factory{std::make_unique<Config>(probe)})
                   .value();
  auto context = Context::create(cpu::device(), wrong);
  ASSERT_FALSE(context);
  EXPECT_TRUE(context.error().IsUnimplemented());
  EXPECT_TRUE(wrong);
  EXPECT_EQ(probe.configurations, 1);
}

TEST(CustomKernel, DirectPublicJourneyKeepsFactoryUntilLastExecutionOwner) {
  Probe probe;
  Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  {
    auto factory = provider(probe).value();
    auto context = Context::create(cpu::device(), factory).value();
    auto queue = context.create_queue().value();
    auto prepared = prepare(context, Operation::custom(Affine{3}),
                            std::array{tensors.in}, std::array{tensors.out})
                        .value();
    auto invocation = prepared.create_invocation().value();
    EXPECT_TRUE(prepared.alias_support(0, 0)->supported);
    factory = {};
    prepared = {};
    EXPECT_EQ(probe.configurations, 1);
    EXPECT_EQ(probe.kernels, 1);
    const int preparations = probe.preparations, proposals = probe.proposals;
    for (int run = 0; run < 3; ++run) {
      auto done = invocation.submit(queue, tensors.in, tensors.out);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{5, 7, 9, 11}));
    }
    EXPECT_EQ(probe.preparations, preparations);
    EXPECT_EQ(probe.proposals, proposals);
    context = {};
    EXPECT_EQ(probe.configurations, 1);
    invocation = {};
    EXPECT_EQ(probe.kernels, 0);
    EXPECT_EQ(probe.invocations, 0);
    EXPECT_EQ(probe.configurations, 1); // Queue still owns ContextState
  }
  EXPECT_EQ(probe.configurations, 0);
}

TEST(CustomKernel,
     RejectedAndAcceptedFailuresHaveDifferentCompletionBoundaries) {
  for (auto fault : {Fault::kRejected, Fault::kAcceptedError}) {
    Probe probe;
    probe.fault = fault;
    Tensors tensors;
    ASSERT_TRUE(tensors.fill().ok());
    auto context =
        Context::create(cpu::device(), provider(probe).value()).value();
    auto queue = context.create_queue().value();
    auto prepared = prepare(context, Operation::custom(Affine{}),
                            std::array{tensors.in}, std::array{tensors.out})
                        .value();
    auto invocation = prepared.create_invocation().value();
    auto done = invocation.submit(queue, tensors.in, tensors.out);
    if (fault == Fault::kRejected) {
      ASSERT_FALSE(done);
      EXPECT_TRUE(done.error().IsDataLoss());
      EXPECT_EQ(probe.executions, 0);
    } else {
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().IsDataLoss());
      EXPECT_EQ(probe.executions, 1);
    }
  }
}

TEST(CustomKernel,
     PreparationFailuresDestroyPartialKernelsAndDoNotGuessFallback) {
  for (auto fault :
       {Fault::kUnsupported, Fault::kResourceDecline, Fault::kFatalUnsupported,
        Fault::kInvalidOutcome, Fault::kMissingKernel, Fault::kRuntimeMismatch,
        Fault::kDelegatedProposal}) {
    SCOPED_TRACE(static_cast<int>(fault));
    Probe first, fallback;
    first.fault = fault;
    const std::array providers{provider(first).value(),
                               provider(fallback).value()};
    auto context =
        Context::create(cpu::device(), {.providers = providers}).value();
    const std::array descriptions{description()};
    auto result = prepare(context, Operation::custom(Affine{}), descriptions,
                          descriptions, hyle::mem::system_resource());
    EXPECT_EQ(result.has_value(), fault == Fault::kUnsupported);
    EXPECT_EQ(fallback.preparations, fault == Fault::kUnsupported ? 1 : 0);
    EXPECT_EQ(first.kernels, 0);
    if (fault == Fault::kFatalUnsupported)
      EXPECT_TRUE(result.error().IsUnimplemented());
    if (fault == Fault::kResourceDecline)
      EXPECT_TRUE(result.error().IsResourceExhausted());
  }
}

TEST(CustomKernel,
     InvalidPropertiesDoNotMoveKernelAndInvocationErrorRollsBack) {
  Probe probe;
  Config config(probe);
  Kernel kernel(config, 0);
  auto invalid = make_custom_kernel(std::move(kernel),
                                    {.workspace = {0, 3}, .runtime = runtime});
  ASSERT_FALSE(invalid);
  EXPECT_NE(kernel.config, nullptr);
  FaultAllocator allocator;
  allocator.fail = 0;
  auto failed = make_custom_kernel(std::move(kernel), {.runtime = runtime},
                                   allocator.ref());
  ASSERT_FALSE(failed);
  EXPECT_NE(kernel.config, nullptr);
  EXPECT_EQ(allocator.live, 0u);
  probe.fault = Fault::kInvocation;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  const std::array descriptions{description()};
  auto prepared = prepare(context, Operation::custom(Affine{}), descriptions,
                          descriptions, hyle::mem::system_resource())
                      .value();
  EXPECT_FALSE(prepared.create_invocation());
  EXPECT_EQ(probe.invocations, 0);
}

TEST(CustomKernel, EveryColdAllocationFailureReleasesAllOwnedObjects) {
  auto attempt = [](FaultAllocator &allocator, Probe &probe) -> Status {
    auto factory = provider(probe, allocator.ref());
    if (!factory)
      return std::move(factory).error();
    const std::array providers{std::move(*factory)};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = allocator.ref(),
                                        .providers = providers});
    if (!context)
      return std::move(context).error();
    const std::array descriptions{description()};
    auto prepared = prepare(*context, Operation::custom(Affine{}), descriptions,
                            descriptions, hyle::mem::system_resource());
    if (!prepared)
      return std::move(prepared).error();
    auto invocation = prepared->create_invocation();
    if (!invocation)
      return std::move(invocation).error();
    return Status::Ok();
  };
  FaultAllocator allocator;
  Probe probe;
  ASSERT_TRUE(attempt(allocator, probe).ok());
  const auto count = allocator.attempts;
  EXPECT_EQ(allocator.live, 0u);
  for (std::size_t failure = 0; failure < count; ++failure) {
    SCOPED_TRACE(failure);
    allocator.attempts = 0;
    allocator.fail = failure;
    EXPECT_TRUE(attempt(allocator, probe).IsResourceExhausted());
    EXPECT_EQ(allocator.live, 0u);
    EXPECT_EQ(probe.configurations, 0);
    EXPECT_EQ(probe.kernels, 0);
    EXPECT_EQ(probe.invocations, 0);
    allocator.fail = SIZE_MAX;
    ASSERT_TRUE(attempt(allocator, probe).ok());
    EXPECT_EQ(allocator.live, 0u);
  }
}

TEST(CustomKernel, IndependentInvocationsShareImmutableKernelConcurrently) {
  Probe probe;
  auto context =
      Context::create(cpu::device(), provider(probe).value()).value();
  const std::array descriptions{description()};
  auto prepared = prepare(context, Operation::custom(Affine{3}), descriptions,
                          descriptions, hyle::mem::system_resource())
                      .value();
  auto work = [&] {
    Tensors tensors;
    ASSERT_TRUE(tensors.fill().ok());
    auto queue = context.create_queue().value();
    auto invocation = prepared.create_invocation().value();
    for (int index = 0; index < 10; ++index) {
      auto done = invocation.submit(queue, tensors.in, tensors.out);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{5, 7, 9, 11}));
    }
  };
  std::thread first(work), second(work);
  first.join();
  second.join();
  EXPECT_EQ(probe.executions, 20);
  EXPECT_EQ(probe.invocations, 0);
  EXPECT_EQ(probe.kernels, 1);
}

} // namespace
