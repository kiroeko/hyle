#include "test_support.h"

namespace {
using namespace ggml_cuda_test;

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

TEST(GgmlCuda, ColdAllocationsRollBackWithoutRetainingNativeResources) {
  MetadataAllocator allocator;
  const std::array providers{compute::ggml_cuda::provider()};
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
    const auto description =
        compute::TensorDesc::dense(compute::DataType::kFloat32,
                                   std::array<std::int64_t, 2>{4, 128})
            .value();
    auto prepare = [&] {
      return compute::prepare(context, compute::RmsNorm{}, description,
                              memory.ref());
    };
    check_oom(allocator, prepare);
    auto prepared = prepare().value();
    check_oom(allocator, [&] { return prepared.create_invocation(); });
  }
  EXPECT_EQ(allocator.live, 0u);
}
} // namespace
