#include "attention_test_support.h"
#include "hyle/compute/batch_execution.h"
#include <cstring>

#if defined(HYLE_FLASHINFER_FAULT_INJECTION)
namespace {
thread_local int fail_attribute_at = 0;
thread_local bool fail_graph_launch = false;
}
extern "C" cudaError_t __real_cudaFuncSetAttribute(const void *,
                                                   cudaFuncAttribute, int);
extern "C" cudaError_t __wrap_cudaFuncSetAttribute(const void *function,
                                                   cudaFuncAttribute attribute,
                                                   int value) {
  if (fail_attribute_at > 0 && --fail_attribute_at == 0)
    return cudaErrorInvalidValue;
  return __real_cudaFuncSetAttribute(function, attribute, value);
}
extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
  if (fail_graph_launch) { fail_graph_launch=false;return cudaErrorInvalidValue; }
  return __real_cudaGraphLaunch(executable,stream);
}
#endif

namespace {
using namespace flashinfer_test;

struct MetadataAllocator final {
  mem::AllocatorRef backing = mem::system_allocator();
  std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  mem::AllocatorRef ref() noexcept {
    return {[](void *opaque, std::size_t bytes,
               std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (self.attempts++ == self.fail_at)
                return nullptr;
              auto *pointer = self.backing.allocate(bytes, alignment);
              if (pointer)
                ++self.live;
              return pointer;
            },
            [](void *opaque, void *pointer, std::size_t bytes,
               std::size_t alignment) noexcept {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (pointer)
                --self.live;
              self.backing.deallocate(pointer, bytes, alignment);
            },
            [](void *opaque, void *pointer, std::size_t old_size,
               std::size_t bytes, std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (bytes && self.attempts++ == self.fail_at)
                return nullptr;
              auto *result =
                  self.backing.reallocate(pointer, old_size, bytes, alignment);
              if (!pointer && result)
                ++self.live;
              else if (pointer && !bytes)
                --self.live;
              return result;
            },
            this};
  }
};

template <class Create>
void check_oom(MetadataAllocator &allocator, Create create) {
  const auto before = allocator.attempts;
  const auto live = allocator.live;
  {
    auto result = create();
    ASSERT_TRUE(result) << result.error().message();
  }
  const auto attempts = allocator.attempts - before;
  ASSERT_GT(attempts, 0u);
  ASSERT_LT(attempts, 32u);
  EXPECT_EQ(allocator.live, live);
  for (std::size_t offset = 0; offset < attempts; ++offset) {
    SCOPED_TRACE(offset);
    allocator.fail_at = allocator.attempts + offset;
    auto result = create();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsResourceExhausted());
    EXPECT_EQ(allocator.live, live);
  }
  allocator.fail_at = SIZE_MAX;
}

TEST(Flashinfer, ColdAllocationFailuresRollBack) {
  MetadataAllocator allocator;
  const std::array providers{
      hyle::compute::ProviderBinding{compute::flashinfer::provider(), 100}};
  const auto device = cuda::device(0).value();
  auto create = [&] {
    return compute::Context::create(
        device,
        {.metadata_allocator = allocator.ref(), .providers = providers});
  };
  check_oom(allocator, create);
  {
    auto context = create().value();
    auto memory =
        cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice)
            .value();
    const auto q =
        compute::TensorDesc::dense(compute::DataType::kFloat16,
                                   std::array<std::int64_t, 4>{2, 1, 8, 256})
            .value();
    const auto kv =
        compute::TensorDesc::dense(compute::DataType::kFloat16,
                                   std::array<std::int64_t, 4>{2, 257, 2, 256})
            .value();
    auto prepare = [&] {
      return compute::prepare(
          context,
          compute::Operation(
              compute::ScaledDotProductAttention{.causal = true}),
          std::array{q, kv, kv}, std::array{q}, memory.ref());
    };
    check_oom(allocator, prepare);
    auto prepared = prepare().value();
    const auto requirement = prepared.workspace_requirement();
    auto workspace = mem::Buffer::create(memory.ref(), requirement.size,
                                         requirement.alignment)
                         .value();
    check_oom(allocator,
              [&] { return prepared.create_invocation(workspace.view()); });
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(Flashinfer, UnqualifiedSemanticsDeclineDuringPrepare) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::flashinfer::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto q =
      compute::TensorDesc::dense(compute::DataType::kFloat16,
                                 std::array<std::int64_t, 4>{1, 17, 8, 256})
          .value();
  const auto kv =
      compute::TensorDesc::dense(compute::DataType::kFloat16,
                                 std::array<std::int64_t, 4>{1, 31, 2, 256})
          .value();
  auto noncausal = compute::prepare(
      context, compute::Operation(compute::ScaledDotProductAttention{}),
      std::array{q, kv, kv}, std::array{q}, memory.ref());
  ASSERT_FALSE(noncausal);
  EXPECT_TRUE(noncausal.error().IsUnimplemented());
  const auto other_heads =
      compute::TensorDesc::dense(compute::DataType::kFloat16,
                                 std::array<std::int64_t, 4>{1, 17, 4, 256})
          .value();
  auto grouped = compute::prepare(
      context,
      compute::Operation(compute::ScaledDotProductAttention{.causal = true}),
      std::array{other_heads, kv, kv}, std::array{other_heads}, memory.ref());
  ASSERT_FALSE(grouped);
  EXPECT_TRUE(grouped.error().IsUnimplemented());
}

#if defined(HYLE_FLASHINFER_FAULT_INJECTION)
void batch_capture_fault(bool encoded,bool launch_failure) {
  MetadataAllocator allocator;
  {
    auto context=compute::Context::create(cuda::device(0).value(),compute::flashinfer::provider()).value();
    auto memory=cuda::MemoryResource::create(context,cuda::MemoryKind::kDevice).value();
    auto host=cuda::MemoryResource::create(context,cuda::MemoryKind::kPinnedHost).value();
    auto queue=context.create_queue().value();
    const auto type=compute::DataType::kFloat16;
    const auto storage=encoded?compute::CacheStorage::float8(2.F).value():compute::CacheStorage::plain(type).value();
    const compute::BatchCapacity cap{.storage=compute::BatchStorage::kRagged,.query_type=type,.output_type=type,
      .key_storage=storage,.value_storage=storage,.max_requests=1,.max_query_rows=1,.max_kv_rows=2,
      .max_pages=0,.max_page_refs=0,.query_heads=2,.kv_heads=1,.qk_head_dim=64,.value_head_dim=encoded?128u:64u,.page_size=1};
    auto made_schema=compute::BatchSchema::create(cap);ASSERT_TRUE(made_schema)<<made_schema.error().message();auto schema=std::move(*made_schema);
    const std::array<std::int32_t,2> qp{0,1},kp{0,2};auto made_layout=schema.version({qp,kp,{},{}});ASSERT_TRUE(made_layout)<<made_layout.error().message();auto layout=std::move(*made_layout);
    auto program=compute::BatchProgram::prepare(context,schema,{.resource=memory.ref(),.host_workspace=host.ref(),
      .max_float_workspace_bytes=8u<<20,.max_int_workspace_bytes=16384,.max_invocations=2,.scale=.125F,
      .window_left=encoded?16:-1,.allocator=allocator.ref()}).value();
    auto pool=compute::LeasePool::create(memory.ref(),4096).value();
    const auto load=[&](std::size_t bytes,bool value){
      auto writer=pool.allocate(bytes).value();std::vector<std::byte> payload(bytes);
      if(value){if(encoded)std::fill(payload.begin(),payload.end(),std::byte{0x20});
        else{const std::uint16_t quarter=0x3400;for(std::size_t i=0;i<bytes;i+=2)std::memcpy(payload.data()+i,&quarter,2);}}
      EXPECT_TRUE(writer.view()->write(payload).ok());return writer.publish().value();
    };
    const auto kd=compute::batch_cache_description(cap,compute::BatchCacheRole::kKey).value();
    const auto vd=compute::batch_cache_description(cap,compute::BatchCacheRole::kValue).value();
    const std::array inputs{load(256,false),load(kd.required_bytes().value(),false),load(vd.required_bytes().value(),true)};
    {
      auto slot=compute::BatchInvocation::create(program,queue).value();auto prepared=slot.prepare_batch(layout).value();
      hyle::Status error;
      if(launch_failure){
        ASSERT_TRUE(prepared.prepare_replay(inputs).ok());fail_graph_launch=true;
        auto submitted=prepared.run(inputs);const bool consumed=!fail_graph_launch;fail_graph_launch=false;
        ASSERT_TRUE(consumed);if(submitted)error=submitted->wait();else error=submitted.error();
      }else{
        fail_attribute_at=1;error=prepared.prepare_replay(inputs);const auto remaining=fail_attribute_at;fail_attribute_at=0;
        ASSERT_EQ(remaining,0);
      }
      ASSERT_FALSE(error.ok());EXPECT_EQ(error.native_error().domain,hyle::NativeErrorDomain::CudaRuntime);
      EXPECT_EQ(error.native_error().code,static_cast<std::int64_t>(cudaErrorInvalidValue));
      EXPECT_FALSE(slot.output());EXPECT_FALSE(prepared.run(inputs));slot.drain().IgnoreError();
    }
    // A failed captured slot cannot be reused; the original Context and a fresh
    // independently owned slot still execute and return the independent mean.
    {
      auto slot=compute::BatchInvocation::create(program,queue).value();auto prepared=slot.prepare_batch(layout).value();
      ASSERT_TRUE(prepared.prepare_replay(inputs).ok());auto done=prepared.run(inputs).value();ASSERT_TRUE(done.wait().ok());
      auto output=slot.output().value();std::vector<std::uint16_t> actual(2*cap.value_head_dim);
      ASSERT_TRUE(output.view()->read(std::as_writable_bytes(std::span(actual))).ok());
      for(auto word:actual)EXPECT_EQ(word,0x3400);
    }
    ASSERT_TRUE(context.drain().ok());
  }
  EXPECT_EQ(allocator.live,0u);
}
TEST(FlashinferBatchReplay, NativeCaptureFailureKeepsOriginalErrorAndRetires) {
  for(bool encoded:{false,true})batch_capture_fault(encoded,false);
}
TEST(FlashinferBatchReplay, NativeReplayLaunchFailureKeepsOriginalErrorAndRetires) {
  for(bool encoded:{false,true})batch_capture_fault(encoded,true);
}
// A link-time CUDA resource failure leaves the actual first native batch in
// flight. No product hooks or replacement Provider are involved.
TEST(Flashinfer, PartialNativeFailureRetainsCompletionAndRetiresInvocation) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::flashinfer::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  Tensor q(compute::DataType::kFloat16, {2, 1, 8, 256}, memory.ref());
  Tensor k(compute::DataType::kFloat16, {2, 17, 2, 256}, memory.ref());
  Tensor v(compute::DataType::kFloat16, {2, 17, 2, 256}, memory.ref());
  Tensor output(compute::DataType::kFloat16, {2, 1, 8, 256}, memory.ref());
  const auto q_values = payload<__half>(q, 0);
  const auto k_values = payload<__half>(k, 5);
  const auto v_values = payload<__half>(v, 13);
  std::vector<__half> result(2 * 8 * 256, encode<__half>(42));
  ASSERT_TRUE(
      output.storage.view().write(std::as_bytes(std::span(result))).ok());
  const std::array inputs{q.view.as_const(), k.view.as_const(),
                          v.view.as_const()};
  const std::array outputs{output.view};
  auto prepared =
      compute::prepare(context,
                       compute::Operation(compute::ScaledDotProductAttention{
                           .scale = 0.0625F, .causal = true}),
                       inputs, outputs, memory.ref())
          .value();
  const auto requirement = prepared.workspace_requirement();
  auto workspace =
      mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment)
          .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  fail_attribute_at = 2;
  auto completion = invocation.submit(queue, inputs, outputs);
  EXPECT_EQ(fail_attribute_at, 0);
  fail_attribute_at = 0;
  ASSERT_TRUE(completion) << completion.error().message();
  auto terminal = completion->wait();
  EXPECT_EQ(terminal.native_error().domain,
            hyle::NativeErrorDomain::CudaRuntime);
  EXPECT_EQ(terminal.native_error().code,
            static_cast<std::int64_t>(cudaErrorInvalidValue));
  EXPECT_FALSE(invocation.submit(queue, inputs, outputs));
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (i < 8 * 256)
      EXPECT_NE(decode(result[i]), 42.0F);
    else
      EXPECT_EQ(decode(result[i]), 42.0F);
  }
  auto recovered = prepared.create_invocation(workspace.view()).value();
  auto done = recovered.submit(queue, inputs, outputs).value();
  ASSERT_TRUE(done.wait().ok());
  check_output(output, q_values, k_values, v_values, 1, 17, 0.002);
}
#endif

} // namespace
