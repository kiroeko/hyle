#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <new>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../src/compute_provider_ggml_metal/native_program.h"
#include "hyle/compute/model_ops.h"
#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"

extern "C" {
void hyle_metal_fault_arm(int);
int hyle_metal_fault_disarm(void);
int hyle_metal_fault_live(void);
void hyle_metal_fault_completion(std::int64_t);
}

namespace {
thread_local int new_left = -1;
thread_local bool new_failed = false;
thread_local bool map_only = false;
thread_local int map_depth = 0;
}
extern "C" void hyle_metal_fault_map_enter(void) { ++map_depth; }
extern "C" void hyle_metal_fault_map_leave(void) { --map_depth; }
void *operator new(std::size_t size) {
  if ((!map_only || map_depth != 0) && new_left > 0 && --new_left == 0) {
    new_left = -1;
    new_failed = true;
    throw std::bad_alloc();
  }
  if (auto *pointer = std::malloc(size == 0 ? 1 : size))
    return pointer;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try { return ::operator new(size); } catch (const std::bad_alloc &) { return nullptr; }
}
void operator delete(void *p, const std::nothrow_t &) noexcept { std::free(p); }

namespace {
namespace c = hyle::compute;
namespace m = c::ggml_metal;
std::array<c::TensorDesc, 6> specs() {
  const auto q = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 4>{1, 2, 1, 32}).value();
  const auto gate = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 4>{1, 2, 1, 1}).value();
  const auto state = c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 4>{1, 1, 32, 32}).value();
  return {q, q, q, gate, gate, state};
}

void check_cold_allocations(c::Operation operation,
                            std::span<const c::TensorDesc> inputs) {
  { auto warm = m::NativeProgram::create(operation, inputs); ASSERT_TRUE(warm); }
  const auto baseline = hyle_metal_fault_live();
  int failed_c = 0;
  bool completed = false;
  for (int index = 1; index <= 32; ++index) {
    hyle_metal_fault_arm(index);
    auto result = m::NativeProgram::create(operation, inputs);
    const bool injected = hyle_metal_fault_disarm();
    if (!injected) { ASSERT_TRUE(result); completed = true; break; }
    ++failed_c;
    ASSERT_FALSE(result);
    // The first two failures are in the native backend constructor, whose
    // null return has no error category. Later failures are our workspace.
    EXPECT_TRUE(index <= 2 ? result.error().IsUnavailable()
                          : result.error().IsResourceExhausted());
    EXPECT_EQ(hyle_metal_fault_live(), baseline);
  }
  EXPECT_TRUE(completed);
  EXPECT_GE(failed_c, 2); // context and event; C++ collection below is separate
  int failed_cpp = 0;
  completed = false;
  for (int index = 1; index <= 48; ++index) {
    new_left = index;
    new_failed = false;
    auto result = m::NativeProgram::create(operation, inputs);
    new_left = -1;
    if (!new_failed) { ASSERT_TRUE(result); completed = true; break; }
    ++failed_cpp;
    ASSERT_FALSE(result);
    // ggml catches a backend-context collection failure and returns null;
    // C++ allocation failures escaping its buffer wrapper retain their type.
    EXPECT_TRUE(result.error().IsResourceExhausted() ||
                (result.error().IsUnavailable() &&
                 result.error().message() == "ggml Metal backend unavailable"));
    EXPECT_EQ(hyle_metal_fault_live(), baseline);
  }
  EXPECT_TRUE(completed);
  EXPECT_GE(failed_cpp, 3);
}

TEST(GgmlMetalFaults, ColdNativeOwnerAllocationsRollBack) {
  check_cold_allocations(c::Operation(c::GatedDeltaNet{}), specs());
}

TEST(GgmlMetalFaults, CompleteGateColdWorkspaceAllocationsRollBack) {
  const auto tensor = [](std::array<std::int64_t, 4> shape) {
    return c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
  };
  const auto q = tensor({1, 1, 16, 128});
  const auto gate = tensor({1, 1, 16, 1});
  const auto state = tensor({1, 16, 128, 128});
  const std::array inputs{q, q, q, gate, gate, gate, gate, state};
  check_cold_allocations(c::model_ops::recurrent_gated_delta_net(), inputs);
}

TEST(GgmlMetalFaults, BfloatActivationWorkspaceAllocationsRollBack) {
  const auto activation = c::TensorDesc::dense(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{64, 128}).value();
  const auto weight = c::TensorDesc::dense(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{32, 128}).value();
  const std::array inputs{activation, weight};
  check_cold_allocations(c::model_ops::bfloat_linear_f32(), inputs);
}

TEST(GgmlMetalFaults, PipelineCacheFailureUnlocksAndReleasesNativeOwner) {
  auto backend = ggml_backend_metal_init();
  ASSERT_NE(backend, nullptr);
  const auto device = static_cast<ggml_metal_device_t>(backend->device->context);
  const auto library = ggml_metal_device_get_library(device);
  ASSERT_NE(library, nullptr);
  const auto baseline = hyle_metal_fault_live();
  const std::string absent = "hyle-resource-test-long-absent-pipeline-name";
  struct MapScope {
    MapScope() { map_only = true; }
    ~MapScope() { new_left = -1; map_only = false; }
  } scope;
  new_left = 1;
  new_failed = false;
  auto lookup = ggml_metal_library_get_pipeline(library, absent.c_str());
  new_left = -1;
  ASSERT_TRUE(new_failed);
  EXPECT_TRUE(lookup.allocation_failed);
  EXPECT_EQ(lookup.pipeline, nullptr);
  // Would deadlock if the failed C++ key allocation escaped the ObjC lock.
  EXPECT_FALSE(ggml_metal_library_get_pipeline(library, absent.c_str()).allocation_failed);

  auto constants = ggml_metal_cv_init();
  ASSERT_NE(constants, nullptr);
  ggml_metal_cv_set_int16(constants, 32, FC_GATED_DELTA_NET + 0);
  ggml_metal_cv_set_int16(constants, 1, FC_GATED_DELTA_NET + 1);
  ggml_metal_cv_set_int16(constants, 1, FC_GATED_DELTA_NET + 2);
  const auto before = hyle_metal_fault_live();
  int failures = 0;
  for (int index = 1; index <= 8; ++index) {
    const std::string name = "hyle-resource-test-cold-pipeline-" + std::to_string(index);
    new_left = index;
    new_failed = false;
    auto pipeline = ggml_metal_library_compile_pipeline(
        library, "kernel_gated_delta_net_f32_1", name.c_str(), constants);
    new_left = -1;
    if (!new_failed) { ASSERT_NE(pipeline.pipeline, nullptr); break; }
    ++failures;
    EXPECT_TRUE(pipeline.allocation_failed);
    EXPECT_EQ(pipeline.pipeline, nullptr);
    EXPECT_EQ(hyle_metal_fault_live(), before);
  }
  EXPECT_GE(failures, 2);
  const auto before_c = hyle_metal_fault_live();
  hyle_metal_fault_arm(1);
  auto pipeline = ggml_metal_library_compile_pipeline(
      library, "kernel_gated_delta_net_f32_1", "hyle-resource-test-c-pipeline", constants);
  ASSERT_TRUE(hyle_metal_fault_disarm());
  EXPECT_TRUE(pipeline.allocation_failed);
  EXPECT_EQ(pipeline.pipeline, nullptr);
  EXPECT_EQ(hyle_metal_fault_live(), before_c);
  ggml_metal_cv_free(constants);
  EXPECT_GE(hyle_metal_fault_live(), baseline); // one successful process cache entry
  ggml_backend_free(backend);
}

TEST(GgmlMetalFaults, ImportRollbackEncodingFailureAndCheckedCompletionRetire) {
  @autoreleasepool {
    const auto inputs = specs();
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    ASSERT_NE(device, nil);
    std::array<id<MTLBuffer>, 7> owners{};
    std::array<m::BufferView, 7> ports;
    const auto output_bytes = inputs[2].required_bytes().value() + inputs[5].required_bytes().value();
    for (std::size_t i = 0; i < owners.size(); ++i) {
      const auto bytes = i == 6 ? output_bytes : inputs[i].required_bytes().value();
      owners[i] = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      ASSERT_NE(owners[i], nil);
      auto *data = static_cast<float *>(owners[i].contents);
      for (std::size_t j = 0; j < bytes / sizeof(float); ++j) data[j] = i == 3 ? -0.1F : 0.05F;
      ports[i] = {(__bridge void *)owners[i], 0};
    }
    for (int index = 1; index <= 7; ++index) {
      auto program = m::NativeProgram::create(inputs).value();
      const auto baseline = hyle_metal_fault_live();
      hyle_metal_fault_arm(index);
      const auto failed = program->run(ports);
      ASSERT_TRUE(hyle_metal_fault_disarm());
      EXPECT_TRUE(failed.IsResourceExhausted());
      EXPECT_EQ(hyle_metal_fault_live(), baseline);
      EXPECT_TRUE(program->run(ports).ok());
    }
    auto program = m::NativeProgram::create(inputs).value();
    ASSERT_TRUE(program->run(ports).ok()); // cache the foreign wrappers first
    hyle_metal_fault_arm(1); // original encoder wrapper
    const auto encoded = program->run(ports);
    ASSERT_TRUE(hyle_metal_fault_disarm());
    EXPECT_TRUE(encoded.IsResourceExhausted());
    EXPECT_TRUE(program->run(ports).IsFailedPrecondition());
    program.reset();
    program = m::NativeProgram::create(inputs).value();
    hyle_metal_fault_completion(9);
    const auto completed = program->run(ports);
    EXPECT_FALSE(completed.ok());
    EXPECT_TRUE(completed.has_native_error());
    EXPECT_EQ(completed.native_error().code, 9);
    EXPECT_TRUE(program->run(ports).IsFailedPrecondition());
    EXPECT_TRUE(std::isfinite(static_cast<float *>(owners[6].contents)[0]));
    program.reset();
    for (auto owner : owners) [owner release];
    [device release];
  }
}
} // namespace
