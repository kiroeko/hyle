#include "C:/public/project-hyle/hyle/.scratch/cross-backend-closeout-20261114/vulkan-run-arena51/component/source/src/compute_provider_ggml_vulkan/native_device.h"
#include "C:/public/project-hyle/hyle/.scratch/cross-backend-closeout-20261114/vulkan-run-arena51/component/source/src/compute_provider_ggml_vulkan/native_graph.h"

#include <array>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "C:/public/project-hyle/hyle/.scratch/cross-backend-closeout-20261114/windows-current/vulkan-sdk/src/compute_ggml_runtime/source-f36c8fffb0df7df9/include/ggml-backend.h"
#include "C:/public/project-hyle/hyle/.scratch/cross-backend-closeout-20261114/windows-current/vulkan-sdk/src/compute_ggml_runtime/source-f36c8fffb0df7df9/include/ggml.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeGraph;

struct GraphStorage {
  NativeDevice device;
  NativeGraph storage;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *weight = nullptr;
  ggml_tensor *output = nullptr;
};

enum class GraphMath { kOriginal, kAdd, kAddRms, kView };

hyle::Result<std::shared_ptr<GraphStorage>> make_graph(const NativeDevice &device,
                                                     GraphMath math = GraphMath::kOriginal) {
  auto result = std::make_shared<GraphStorage>();
  result->device = device;
  auto active = device.activate();
  HYLE_ASSIGN_OR_RETURN(auto arena, MetadataArena::create(
      24 * ggml_tensor_overhead() + ggml_graph_overhead_custom(24, false),
      hyle::mem::system_allocator()));
  auto *context = arena.get();
  result->input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  result->weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
  if (math == GraphMath::kView) {
    result->output = ggml_view_1d(context, result->input, 2048, 0);
  } else if (math == GraphMath::kOriginal) {
    auto *norm = ggml_rms_norm(context, result->input, 1e-6f);
    auto *scaled = ggml_mul(context, norm, result->weight);
    result->output = ggml_add(context, ggml_silu(context, scaled), result->input);
  } else {
    result->output = ggml_add(context, result->input, result->weight);
    if (math == GraphMath::kAddRms)
      result->output = ggml_rms_norm(context, result->output, 1e-6f);
  }
  ggml_set_input(result->input);
  ggml_set_input(result->weight);
  ggml_set_output(result->output);
  result->graph = ggml_new_graph_custom(context, 24, false);
  ggml_build_forward_expand(result->graph, result->output);
  HYLE_ASSIGN_OR_RETURN(result->storage,
      NativeGraph::allocate(device, std::move(arena), result->graph));
  return result;
}

hyle::Result<std::shared_ptr<GraphStorage>>
make_bfloat_producer(const NativeDevice &device, bool attention) {
  auto result = std::make_shared<GraphStorage>();
  result->device = device;
  auto active = device.activate();
  HYLE_ASSIGN_OR_RETURN(auto arena, MetadataArena::for_graph(
      8, 16, hyle::mem::system_allocator()));
  auto *context = arena.get();
  result->input = attention
      ? ggml_new_tensor_3d(context, GGML_TYPE_F32, 64, 31, 2)
      : ggml_new_tensor_2d(context, GGML_TYPE_F32, 2048, 3);
  result->output = ggml_cast(context, result->input, GGML_TYPE_BF16);
  ggml_set_input(result->input);
  ggml_set_output(result->output);
  result->graph = ggml_new_graph_custom(context, 16, false);
  ggml_build_forward_expand(result->graph, result->output);
  HYLE_ASSIGN_OR_RETURN(result->storage,
      NativeGraph::allocate(device, std::move(arena), result->graph));
  return result;
}

hyle::Result<std::shared_ptr<GraphStorage>> make_bfloat_consumer(
    const NativeDevice &device, const std::shared_ptr<GraphStorage> &producer,
    bool attention) {
  auto result = std::make_shared<GraphStorage>();
  result->device = device;
  auto active = device.activate();
  HYLE_ASSIGN_OR_RETURN(auto arena, MetadataArena::for_graph(
      12, 24, hyle::mem::system_allocator()));
  auto *context = arena.get();
  result->input = ggml_dup_tensor(context, producer->output);
  auto *root = producer->output;
  while (root->view_src) root = root->view_src;
  const auto offset = reinterpret_cast<std::uintptr_t>(producer->output->data) -
      reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(root->buffer));
  HYLE_ASSIGN_OR_RETURN(auto allocation,
      device.borrow(device.buffer_handle(root->buffer),
          ggml_backend_buffer_get_size(root->buffer), producer));
  auto imported = std::shared_ptr<ggml_backend_buffer>(allocation,
      [owner = device](ggml_backend_buffer *buffer) {
        auto active = owner.activate();
        ggml_backend_buffer_free(buffer);
      });
  if (ggml_backend_tensor_alloc(imported.get(), result->input,
          static_cast<char *>(ggml_backend_buffer_get_base(imported.get())) +
              offset) != GGML_STATUS_SUCCESS)
    return std::unexpected(hyle::Status::Internal("test BF16 input binding"));
  ggml_set_input(result->input);
  auto *converted = ggml_cast(context, result->input, GGML_TYPE_F32);
  if (attention) {
    auto *query = ggml_view_4d(context, converted, 64, 1, 2, 1,
        converted->nb[1], converted->nb[2], converted->nb[3], 0);
    result->output = ggml_flash_attn_ext(context, query, converted, converted,
        nullptr, .125F, 0.F, 0.F);
    if (!ggml_prec_set_acc(result->output, GGML_PREC_F32))
      return std::unexpected(hyle::Status::Internal("test flash precision"));
    namespace bridge = hyle::compute::ggml_vulkan::bridge;
    result->output->op_params[bridge::kFlashAttentionMathSlot] =
        bridge::kFlashAttentionFullF32;
  } else {
    result->output = converted;
  }
  ggml_set_output(result->output);
  result->graph = ggml_new_graph_custom(context, 24, false);
  ggml_build_forward_expand(result->graph, result->output);
  HYLE_ASSIGN_OR_RETURN(result->storage, NativeGraph::allocate(
      device, std::move(arena), result->graph, imported));
  return result;
}

void check_bfloat_handoff(bool attention) {
  auto device = NativeDevice::create().value();
  auto producer = make_bfloat_producer(device, attention).value();
  auto consumer = make_bfloat_consumer(device, producer, attention).value();
  auto writer = device.create_work().value();
  auto reader = device.create_work().value();
  std::vector<float> input(ggml_nelements(producer->input));
  std::vector<float> output(ggml_nelements(consumer->output));
  for (std::size_t step = 0; step < 6; ++step) {
    SCOPED_TRACE(step);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(step) * .125F +
          (attention ? float(i / (64 * 31)) * .25F : float(i % 7) * .03125F);
    {
      auto active = device.activate();
      ggml_backend_tensor_set(producer->input, input.data(), 0,
          input.size() * sizeof(float));
    }
    // Both original independent Works are accepted before any wait, poll,
    // readback or further upload. The shared BF16 bytes come from the writer.
    const auto produced = writer.submit(producer->graph, producer);
    ASSERT_TRUE(produced.status.ok()) << produced.status.message();
    ASSERT_TRUE(produced.pending);
    const auto consumed = reader.submit(consumer->graph, consumer);
    ASSERT_TRUE(consumed.status.ok()) << consumed.status.message();
    ASSERT_TRUE(consumed.pending);
    const auto read_done = reader.wait();
    ASSERT_TRUE(read_done.status.ok()) << read_done.status.message();
    ASSERT_FALSE(read_done.pending);
    const auto write_done = writer.wait();
    ASSERT_TRUE(write_done.status.ok()) << write_done.status.message();
    ASSERT_FALSE(write_done.pending);
    if (attention && step > 0)
      ASSERT_TRUE(reader.recording_ready());
    if (!attention) EXPECT_FALSE(reader.recording_ready());
    {
      auto active = device.activate();
      ggml_backend_tensor_get(consumer->output, output.data(), 0,
          output.size() * sizeof(float));
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
      const auto expected = attention
          ? float(step) * .125F + float(i / 64) * .25F : input[i];
      ASSERT_NEAR(output[i], expected, 1.e-5F) << i;
    }
  }
}

TEST(GgmlVulkanWork, IndependentBfloatProducerAndConsumerUseOriginalQueueDependency) {
  check_bfloat_handoff(false);
}

TEST(GgmlVulkanWork, RecordedFlashRetainsBfloatHandoffDependencyOnReplay) {
  check_bfloat_handoff(true);
}

TEST(GgmlVulkanWork, ViewOnlyGraphRetiresAndReusesWithoutCreatingComputeWork) {
  auto device = NativeDevice::create().value();
  auto frame_result = make_graph(device, GraphMath::kView);
  ASSERT_TRUE(frame_result) << frame_result.error().message();
  auto frame = *frame_result;
  ASSERT_EQ(ggml_graph_n_nodes(frame->graph), 1);
  auto work = device.create_work().value();
  std::vector<float> expected(2048), actual(2048);
  for (int step = 0; step < 3; ++step) {
    for (std::size_t i = 0; i < expected.size(); ++i)
      expected[i] = float(step) * .25F + float(i % 11) * .03125F;
    {
      auto active = device.activate();
      ggml_backend_tensor_set(frame->input, expected.data(), 0,
          expected.size() * sizeof(float));
    }
    const auto admitted = work.submit(frame->graph, frame);
    ASSERT_TRUE(admitted.status.ok()) << admitted.status.message();
    const auto retired = work.wait();
    ASSERT_TRUE(retired.status.ok()) << retired.status.message();
    EXPECT_FALSE(retired.pending);
    EXPECT_FALSE(work.recording_ready());
    auto active = device.activate();
    ggml_backend_tensor_get(frame->output, actual.data(), 0,
        actual.size() * sizeof(float));
    EXPECT_EQ(actual, expected);
  }
}

TEST(GgmlVulkanWork, RepeatedIndependentSlotsRetainTheirResourceLeases) {
#if defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
               SEM_NOOPENFILEERRORBOX);
#endif
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto first = device->create_work();
  auto second = device->create_work();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto a = make_graph(*device);
  auto b = make_graph(*device);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  const std::array frames{*a, *b};
  const std::array slots{&*first, &*second};
  std::array<std::vector<float>, 2> inputs, weights;
  for (std::size_t slot = 0; slot < 2; ++slot) {
    inputs[slot].resize(2048);
    weights[slot].assign(2048, 0.75f + 0.25f * static_cast<float>(slot));
  }
  for (int iteration = 0; iteration < 3; ++iteration) {
    std::array<std::weak_ptr<std::shared_ptr<GraphStorage>>, 2> leases;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      for (std::size_t i = 0; i < 2048; ++i)
        inputs[slot][i] = static_cast<float>(static_cast<int>(i % 31) - 15) *
                         (0.1f + 0.05f * iteration + 0.02f * static_cast<float>(slot));
      {
        auto active = device->activate();
        ggml_backend_tensor_set(frames[slot]->input, inputs[slot].data(), 0, 2048 * sizeof(float));
        ggml_backend_tensor_set(frames[slot]->weight, weights[slot].data(), 0, 2048 * sizeof(float));
      }
    }
    // Finish synchronous uploads before admitting either region, so uploading
    // the second input cannot accidentally wait for the first execution.
    for (std::size_t slot = 0; slot < 2; ++slot) {
      auto lease = std::make_shared<std::shared_ptr<GraphStorage>>(frames[slot]);
      leases[slot] = lease;
      const auto submitted = slots[slot]->submit(frames[slot]->graph, std::move(lease));
      ASSERT_TRUE(submitted.status.ok());
      EXPECT_TRUE(submitted.pending);
      EXPECT_FALSE(leases[slot].expired());
      const auto rejected = slots[slot]->submit(frames[slot]->graph, frames[slot]);
      EXPECT_FALSE(rejected.status.ok());
      EXPECT_TRUE(rejected.pending);
    }
    // Both slots have been admitted before either owning observation.
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const auto done = slots[slot]->wait();
      ASSERT_TRUE(done.status.ok());
      EXPECT_FALSE(done.pending);
      EXPECT_TRUE(leases[slot].expired());
      std::vector<float> actual(2048);
      auto active = device->activate();
      ggml_backend_tensor_get(frames[slot]->output, actual.data(), 0, actual.size() * sizeof(float));
      double sum = 0;
      for (const auto value : inputs[slot]) sum += static_cast<double>(value) * value;
      const double inverse = 1.0 / std::sqrt(sum / 2048 + 1e-6);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double z = inputs[slot][i] * inverse * weights[slot][i];
        const double expected = z / (1 + std::exp(-z)) + inputs[slot][i];
        ASSERT_NEAR(actual[i], expected, 6e-6 * (1 + std::abs(expected)));
      }
      EXPECT_FALSE(slots[slot]->poll().pending);
    }
  }
}

TEST(GgmlVulkanWork, AlternatingAddAndAddRmsGraphsKeepOriginalMathAndRetirement) {
  auto device = NativeDevice::create().value();
  auto work = device.create_work().value();
  const std::array frames{make_graph(device, GraphMath::kAdd).value(),
                          make_graph(device, GraphMath::kAddRms).value()};
  std::vector<float> input(2048), weight(2048), sum(2048), actual(2048);
  for (std::size_t iteration = 0; iteration < 6; ++iteration) {
    const auto &frame = frames[iteration % 2];
    double squares = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = float(int(i % 31) - 15) * (0.1F + float(iteration) * .02F);
      weight[i] = float(i % 7) * .125F;
      sum[i] = input[i] + weight[i];
      squares += double(sum[i]) * sum[i];
    }
    {
      auto active = device.activate();
      ggml_backend_tensor_set(frame->input, input.data(), 0, input.size() * sizeof(float));
      ggml_backend_tensor_set(frame->weight, weight.data(), 0, weight.size() * sizeof(float));
    }
    const auto admitted = work.submit(frame->graph, frame);
    ASSERT_TRUE(admitted.status.ok()) << admitted.status.message();
    const auto retired = work.wait();
    ASSERT_TRUE(retired.status.ok()) << retired.status.message();
    EXPECT_FALSE(retired.pending);
    {
      auto active = device.activate();
      ggml_backend_tensor_get(frame->output, actual.data(), 0, actual.size() * sizeof(float));
    }
    const double scale = iteration % 2 ? 1.0 / std::sqrt(squares / sum.size() + 1e-6) : 1.0;
    for (std::size_t i = 0; i < actual.size(); ++i)
      ASSERT_NEAR(actual[i], sum[i] * scale, 6e-6 * (1.0 + std::abs(sum[i] * scale)));
  }
}

TEST(GgmlVulkanWork, ManyIndependentBindingsSurvivePartialOwnerRetirementAndReuse) {
  auto device = NativeDevice::create().value();
  using Work = hyle::compute::ggml_vulkan::detail::NativeWork;
  std::vector<Work> works;
  std::vector<std::shared_ptr<GraphStorage>> frames;
  constexpr std::size_t first_count = 260, replacement_count = 32;
  works.reserve(first_count + replacement_count);
  frames.reserve(first_count + replacement_count);
  for (std::size_t index = 0; index < first_count + replacement_count; ++index) {
    works.push_back(device.create_work().value());
    frames.push_back(make_graph(device, GraphMath::kAdd).value());
    const std::vector<float> a(2048, float(index) + .25F), b(2048, .5F);
    auto active = device.activate();
    ggml_backend_tensor_set(frames.back()->input, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(frames.back()->weight, b.data(), 0, b.size() * sizeof(float));
  }
  const auto submit = [&](std::size_t index) {
    const auto submitted = works[index].submit(frames[index]->graph, frames[index]);
    ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
    EXPECT_TRUE(submitted.pending);
  };
  const auto observe = [&](std::size_t index) {
    const auto retired = works[index].wait();
    ASSERT_TRUE(retired.status.ok()) << retired.status.message();
    EXPECT_FALSE(retired.pending);
    std::vector<float> output(2048);
    auto active = device.activate();
    ggml_backend_tensor_get(frames[index]->output, output.data(), 0, output.size() * sizeof(float));
    for (const float value : output) ASSERT_EQ(value, float(index) + .75F);
  };
  // All real uploads precede admission; they cannot serialize away overlap.
  for (std::size_t i = 0; i < first_count; ++i) submit(i);
  for (std::size_t i = 0; i < first_count; ++i) observe(i);
  // Keep odd owners alive while retired even owners return their native
  // descriptor capacity. New bindings must never replace an odd owner's sets.
  for (std::size_t i = 0; i < first_count; i += 2) works[i] = Work{};
  // The fully owned native fixture permits in-place Add. As in the repeated
  // slot case above, restore every input before admitting the next wave.
  for (std::size_t i = 1; i < first_count; i += 2) {
    const std::vector<float> a(2048, float(i) + .25F), b(2048, .5F);
    auto active = device.activate();
    ggml_backend_tensor_set(frames[i]->input, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(frames[i]->weight, b.data(), 0, b.size() * sizeof(float));
  }
  for (std::size_t i = first_count; i < works.size(); ++i) submit(i);
  for (std::size_t i = 1; i < first_count; i += 2) submit(i);
  for (std::size_t i = first_count; i < works.size(); ++i) observe(i);
  for (std::size_t i = 1; i < first_count; i += 2) observe(i);
}

TEST(GgmlVulkanWork, MovedPendingOwnerDrainsBeforeDroppingItsLastGraphLease) {
  auto device = NativeDevice::create();
  ASSERT_TRUE(device);
  auto graph = make_graph(*device);
  ASSERT_TRUE(graph);
  std::weak_ptr<GraphStorage> lifetime = *graph;
  {
    auto work = device->create_work();
    ASSERT_TRUE(work);
    auto active = device->activate();
    const std::vector<float> values(2048, 0.5f);
    ggml_backend_tensor_set((*graph)->input, values.data(), 0, 2048 * sizeof(float));
    ggml_backend_tensor_set((*graph)->weight, values.data(), 0, 2048 * sizeof(float));
    ASSERT_TRUE(work->submit((*graph)->graph, *graph).status.ok());
    graph->reset();
    EXPECT_FALSE(lifetime.expired());
    auto moved = std::move(*work);
    EXPECT_FALSE(work->poll().status.ok());
    // Emergency destruction observes native retirement before the graph and
    // payload lease disappear. Ordinary public abandonment retains this owner.
  }
  EXPECT_TRUE(lifetime.expired());
}
} // namespace


namespace {
void actual_arena_device51(const NativeDevice &device) {
  const auto &native=device.native();
  auto properties_fn=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      native.get_instance_proc_addr(native.instance,"vkGetPhysicalDeviceProperties"));
  ASSERT_NE(properties_fn,nullptr); VkPhysicalDeviceProperties properties{};
  properties_fn(native.physical,&properties);
  std::printf("ARENA51_ACTUAL_DEVICE %s vendor=%u type=%u\n",properties.deviceName,
      properties.vendorID,static_cast<unsigned>(properties.deviceType));
  ASSERT_EQ(properties.vendorID,0x10deu);
  ASSERT_EQ(properties.deviceType,VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
#if defined(_WIN32)
  const auto module=GetModuleHandleW(L"hyle_compute_provider_ggml_vulkan.dll");
  ASSERT_NE(module,nullptr); wchar_t name[32768]{};
  ASSERT_GT(GetModuleFileNameW(module,name,32768),0u);
  std::printf("ARENA51_ACTUAL_PROVIDER %ls\n",name);
#endif
}
struct RunArenaPorts51 {
  NativeDevice device;
  std::vector<std::shared_ptr<ggml_backend_buffer>> buffers;
  std::shared_ptr<RunArenaPorts51> previous;
};
struct RunArenaFrame51 {
  std::shared_ptr<RunArenaPorts51> ports;
  NativeGraph storage;
  ggml_tensor *input = nullptr, *weight = nullptr, *temporary = nullptr, *output = nullptr;
};
hyle::Result<std::shared_ptr<RunArenaFrame51>> make_run_arena_frame51(
    const NativeDevice &device,
    const std::shared_ptr<hyle::compute::ggml_vulkan::bridge::RunArena> &scratch,
    const std::shared_ptr<RunArenaFrame51> &previous = {}) {
  auto frame = std::make_shared<RunArenaFrame51>();
  frame->ports = std::make_shared<RunArenaPorts51>();
  frame->ports->device = device;
  auto active = device.activate();
  HYLE_ASSIGN_OR_RETURN(auto metadata, MetadataArena::for_graph(
      8, 12, hyle::mem::system_allocator()));
  auto *ctx = metadata.get();
  frame->input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
  frame->weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
  frame->temporary = ggml_add(ctx, frame->input, frame->weight);
  frame->output = ggml_neg(ctx, frame->temporary);
  ggml_set_input(frame->input); ggml_set_input(frame->weight);
  ggml_set_output(frame->output);
  const auto bind_owned = [&](ggml_tensor *tensor) -> hyle::Status {
    HYLE_ASSIGN_OR_RETURN(auto raw, device.allocate_buffer(ggml_nbytes(tensor)));
    std::shared_ptr<ggml_backend_buffer> buffer(raw, [device](auto *p) {
      auto active = device.activate(); ggml_backend_buffer_free(p);
    });
    if (ggml_backend_tensor_alloc(buffer.get(), tensor,
          ggml_backend_buffer_get_base(buffer.get())) != GGML_STATUS_SUCCESS)
      return hyle::Status::Internal("arena fixture original port binding");
    frame->ports->buffers.push_back(std::move(buffer));
    return hyle::Status::Ok();
  };
  if (previous) {
    frame->ports->previous = previous->ports;
    if (ggml_backend_tensor_alloc(previous->output->buffer, frame->input,
          previous->output->data) != GGML_STATUS_SUCCESS)
      return std::unexpected(hyle::Status::Internal("arena fixture complete original handoff"));
  } else { HYLE_RETURN_IF_ERROR(bind_owned(frame->input)); }
  HYLE_RETURN_IF_ERROR(bind_owned(frame->weight));
  HYLE_RETURN_IF_ERROR(bind_owned(frame->output));
  auto *graph = ggml_new_graph_custom(ctx, 12, false);
  ggml_build_forward_expand(graph, frame->output);
  HYLE_ASSIGN_OR_RETURN(frame->storage, NativeGraph::allocate_for_run(
      device, std::move(metadata), graph, frame->ports, scratch));
  return frame;
}
} // namespace

TEST(GgmlVulkanWork, RunArenaSharesTwoNonzeroPrivateGraphsAndPreservesPorts) {
  auto created = NativeDevice::create();
  if (!created) GTEST_SKIP() << created.error().message();
  auto device = *created;
  actual_arena_device51(device);
  auto planned = device.create_run_arena(); ASSERT_TRUE(planned) << planned.error().message();
  auto scratch = *planned;
  auto left_result = make_run_arena_frame51(device, scratch); ASSERT_TRUE(left_result);
  auto left = *left_result;
  auto right_result = make_run_arena_frame51(device, scratch, left); ASSERT_TRUE(right_result);
  auto right = *right_result;
  EXPECT_GT(left->storage.storage_bytes(), 0u);
  EXPECT_GT(right->storage.storage_bytes(), 0u);
  EXPECT_EQ(left->storage.get(), nullptr);
  EXPECT_FALSE(left->storage.materialize_run_storage().ok()); // Not sealed.
  ASSERT_TRUE(device.seal_run_arena(scratch).ok());
  ASSERT_TRUE(left->storage.materialize_run_storage().ok());
  ASSERT_TRUE(right->storage.materialize_run_storage().ok());
  const auto backing = device.buffer_handle(left->temporary->buffer);
  ASSERT_NE(backing, VK_NULL_HANDLE);
  ASSERT_EQ(backing, device.buffer_handle(right->temporary->buffer));
  EXPECT_NE(backing, device.buffer_handle(left->output->buffer));
  EXPECT_NE(backing, device.buffer_handle(right->output->buffer));
  auto a = device.create_work(); ASSERT_TRUE(a);
  auto b = device.create_work(); ASSERT_TRUE(b);
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<float> x(2048, float(replay) + .25F), w(2048, .5F), z(2048, .75F);
    { auto active = device.activate();
      ggml_backend_tensor_set(left->input, x.data(), 0, x.size()*sizeof(float));
      ggml_backend_tensor_set(left->weight, w.data(), 0, w.size()*sizeof(float));
      ggml_backend_tensor_set(right->weight, z.data(), 0, z.size()*sizeof(float)); }
    const auto first = a->submit(left->storage.get(), left->storage.lease());
    ASSERT_TRUE(first.status.ok()); ASSERT_TRUE(first.pending);
    const auto second = b->submit(right->storage.get(), right->storage.lease());
    ASSERT_TRUE(second.status.ok()); ASSERT_TRUE(second.pending); // No wait between stages.
    ASSERT_TRUE(b->wait().status.ok()); ASSERT_TRUE(a->wait().status.ok());
    std::vector<float> retained(2048), output(2048);
    { auto active = device.activate();
      ggml_backend_tensor_get(left->output, retained.data(), 0, retained.size()*sizeof(float));
      ggml_backend_tensor_get(right->output, output.data(), 0, output.size()*sizeof(float)); }
    for (std::size_t i=0;i<x.size();++i) {
      ASSERT_EQ(retained[i], -(x[i]+.5F));
      ASSERT_EQ(output[i], x[i]-.25F);
    }
    EXPECT_EQ(backing, device.buffer_handle(left->temporary->buffer));
  }
}

TEST(GgmlVulkanWork, RunArenasKeepSlotsSeparateAndRetainPendingGraphOnFailure) {
  auto created = NativeDevice::create();
  if (!created) GTEST_SKIP() << created.error().message();
  auto device = *created;
  actual_arena_device51(device);
  auto first = device.create_run_arena(); ASSERT_TRUE(first);
  auto second = device.create_run_arena(); ASSERT_TRUE(second);
  auto a_result=make_run_arena_frame51(device,*first); ASSERT_TRUE(a_result);
  auto b_result=make_run_arena_frame51(device,*second); ASSERT_TRUE(b_result);
  auto a_frame=*a_result; auto b_frame=*b_result;
  ASSERT_TRUE(device.seal_run_arena(*first).ok());
  ASSERT_TRUE(device.seal_run_arena(*second).ok());
  ASSERT_TRUE(a_frame->storage.materialize_run_storage().ok());
  ASSERT_TRUE(b_frame->storage.materialize_run_storage().ok());
  ASSERT_NE(device.buffer_handle(a_frame->temporary->buffer),
            device.buffer_handle(b_frame->temporary->buffer));
  auto work=device.create_work(); ASSERT_TRUE(work);
  std::vector<float> input(2048,2.F),weight(2048,.5F);
  { auto active=device.activate();
    ggml_backend_tensor_set(a_frame->input,input.data(),0,input.size()*sizeof(float));
    ggml_backend_tensor_set(a_frame->weight,weight.data(),0,weight.size()*sizeof(float)); }
  const auto accepted=work->submit(a_frame->storage.get(),a_frame->storage.lease());
  ASSERT_TRUE(accepted.status.ok()); ASSERT_TRUE(accepted.pending);
  const auto rejected=work->submit(b_frame->storage.get(),b_frame->storage.lease());
  EXPECT_FALSE(rejected.status.ok()); EXPECT_TRUE(rejected.pending);
  // Work's original lease alone keeps graph metadata and scratch alive.
  // Retain only independently owned public IO for the post-retirement read.
  auto io=a_frame->ports; const auto output=*a_frame->output;
  a_frame.reset(); a_result->reset(); first->reset();
  const auto retired=work->wait(); ASSERT_TRUE(retired.status.ok()); EXPECT_FALSE(retired.pending);
  std::vector<float> observed(2048);
  { auto active=device.activate();
    ggml_backend_tensor_get(&output,observed.data(),0,observed.size()*sizeof(float)); }
  for (const float value:observed) ASSERT_EQ(value,-2.5F);
}
