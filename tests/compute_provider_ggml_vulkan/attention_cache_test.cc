#include "native_graph.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include <gtest/gtest.h>

namespace {
namespace n = hyle::compute::ggml_vulkan::detail;
namespace b = hyle::compute::ggml_vulkan::bridge;

TEST(GgmlVulkanAttention,
     RecordedFlashReadsCurrentPayloadAcross64StepsAndNewGraphOwner) {
  auto device = n::NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  auto work = device->create_work();
  ASSERT_TRUE(work) << work.error().message();
  for (int owner : {0, 1}) {
    auto arena =
        n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
    ASSERT_TRUE(arena) << arena.error().message();
    auto *context = arena->get();
    auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
    auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
    auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
    auto *output =
        ggml_flash_attn_ext(context, q, k, v, nullptr, .125F, 0.F, 0.F);
    ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
    output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
    for (auto *input : {q, k, v})
      ggml_set_input(input);
    // Persistent ready ports live outside gallocr's reusable workspace.
    auto imports =
        std::make_shared<std::array<std::shared_ptr<ggml_backend_buffer>, 3>>();
    const std::array<ggml_tensor *, 3> inputs{q, k, v};
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      auto allocation = device->allocate_buffer(ggml_nbytes(inputs[port]));
      ASSERT_TRUE(allocation) << allocation.error().message();
      (*imports)[port] = std::shared_ptr<ggml_backend_buffer>(
          *allocation, [owner = *device](ggml_backend_buffer *buffer) {
            auto active = owner.activate();
            ggml_backend_buffer_free(buffer);
          });
      ASSERT_EQ(ggml_backend_tensor_alloc(
                    (*imports)[port].get(), inputs[port],
                    ggml_backend_buffer_get_base((*imports)[port].get())),
                GGML_STATUS_SUCCESS);
    }
    ggml_set_output(output);
    auto *graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto native =
        n::NativeGraph::allocate(*device, std::move(*arena), graph, imports);
    ASSERT_TRUE(native) << native.error().message();
    const std::vector<float> query(128, 0.F), keys(31 * 128, 0.F);
    ggml_backend_tensor_set(q, query.data(), 0, query.size() * sizeof(float));
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(float));
    for (int step = 0; step < 64; ++step) {
      const auto expected = float(owner * 64 + step) * .03125F + .125F;
      const std::vector<float> values(31 * 128, expected);
      ggml_backend_tensor_set(v, values.data(), 0,
                              values.size() * sizeof(float));
      if (step == 32) {
        ASSERT_TRUE(work->invalidate_recording().ok());
        EXPECT_FALSE(work->recording_ready());
      }
      auto submitted = work->submit(native->get(), native->lease());
      ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
      EXPECT_FALSE(work->recording_ready());
      auto completed = work->wait();
      ASSERT_TRUE(completed.status.ok()) << completed.status.message();
      // A scalar flash may need no scratch growth and cache immediately.
      // Once storage has settled, every replay and explicit invalidation must
      // recover the cache; changing graph owners must preserve that behavior.
      if (owner != 0 || step != 0) {
        ASSERT_TRUE(work->recording_ready())
            << "stable original flash recording was not retained at owner "
            << owner << " step " << step;
      }
      std::array<float, 128> actual{};
      ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
      for (const auto value : actual) {
        ASSERT_TRUE(std::isfinite(value));
        EXPECT_NEAR(value, expected, 1.e-5F);
      }
    }
  }
}

TEST(GgmlVulkanAttention,
     RecordedTwoFlashConcatReadsCurrentPayloadAcross64Steps) {
  auto device=n::NativeDevice::create();ASSERT_TRUE(device)<<device.error().message();
  auto active=device->activate();auto work=device->create_work();ASSERT_TRUE(work);
  auto arena=n::MetadataArena::for_graph(32,64,hyle::mem::system_allocator());ASSERT_TRUE(arena);
  auto *context=arena->get();
  std::array<ggml_tensor*,6> inputs{};
  std::array<ggml_tensor*,2> parts{};
  for(unsigned request=0;request<2;++request){
    const auto rows=request?17:1,keys=request?47:31;
    auto *q=inputs[3*request]=ggml_new_tensor_4d(context,GGML_TYPE_F32,64,rows,8,1);
    auto *k=inputs[3*request+1]=ggml_new_tensor_4d(context,GGML_TYPE_F32,64,keys,2,1);
    auto *v=inputs[3*request+2]=ggml_new_tensor_4d(context,GGML_TYPE_F32,64,keys,2,1);
    auto *attention=ggml_flash_attn_ext(context,q,k,v,nullptr,.125F,0.F,0.F);
    ASSERT_TRUE(ggml_prec_set_acc(attention,GGML_PREC_F32));
    attention->op_params[b::kFlashAttentionMathSlot]=b::kFlashAttentionFullF32;
    parts[request]=ggml_reshape_2d(context,attention,64*8,rows);
  }
  auto imports=std::make_shared<std::array<std::shared_ptr<ggml_backend_buffer>,6>>();
  for(unsigned port=0;port<inputs.size();++port){
    ggml_set_input(inputs[port]);auto buffer=device->allocate_buffer(ggml_nbytes(inputs[port]));ASSERT_TRUE(buffer);
    (*imports)[port]=std::shared_ptr<ggml_backend_buffer>(*buffer,[owner=*device](ggml_backend_buffer*p){auto active=owner.activate();ggml_backend_buffer_free(p);});
    ASSERT_EQ(ggml_backend_tensor_alloc((*imports)[port].get(),inputs[port],ggml_backend_buffer_get_base((*imports)[port].get())),GGML_STATUS_SUCCESS);
    const std::vector<float> zero(ggml_nelements(inputs[port]),0.F);ggml_backend_tensor_set(inputs[port],zero.data(),0,zero.size()*sizeof(float));
  }
  auto *output=ggml_concat(context,parts[0],parts[1],1);ggml_set_output(output);
  auto *graph=ggml_new_graph_custom(context,64,false);ggml_build_forward_expand(graph,output);
  auto native=n::NativeGraph::allocate(*device,std::move(*arena),graph,imports);ASSERT_TRUE(native)<<native.error().message();
  for(unsigned step=0;step<64;++step){
    for(unsigned request=0;request<2;++request){const std::vector<float> values(ggml_nelements(inputs[3*request+2]),float(step)*.015625F+float(request)*.25F);ggml_backend_tensor_set(inputs[3*request+2],values.data(),0,values.size()*sizeof(float));}
    if(step==32){ASSERT_TRUE(work->invalidate_recording().ok());EXPECT_FALSE(work->recording_ready());}
    auto submitted=work->submit(native->get(),native->lease());ASSERT_TRUE(submitted.status.ok())<<submitted.status.message();
    ASSERT_TRUE(work->wait().status.ok());
    if(step==0)EXPECT_FALSE(work->recording_ready())<<"cold scratch allocation must reject the initial recording";
    else ASSERT_TRUE(work->recording_ready())<<"stable multi flash/Concat recording was not retained at step "<<step;
    std::vector<float> actual(18*512);ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
    for(std::size_t i=0;i<actual.size();++i)EXPECT_NEAR(actual[i],float(step)*.015625F+(i<512?0.F:.25F),1.e-5F);
  }
}

TEST(GgmlVulkanAttention,
     MixedLongKvScratchGrowthRejectsColdRecordingAndThenReplays) {
  auto device = n::NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  auto work = device->create_work();
  ASSERT_TRUE(work) << work.error().message();
  auto arena =
      n::MetadataArena::for_graph(32, 64, hyle::mem::system_allocator());
  ASSERT_TRUE(arena) << arena.error().message();
  auto *context = arena->get();
  constexpr int depth = 256, heads = 8, kv_heads = 2;
  constexpr std::array<int, 2> query_rows{1, 17}, key_rows{1024, 768};
  std::array<ggml_tensor *, 6> inputs{};
  std::array<ggml_tensor *, 2> parts{};
  for (std::size_t request = 0; request < parts.size(); ++request) {
    auto *q = inputs[3 * request] = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, depth, query_rows[request], heads, 1);
    auto *k = inputs[3 * request + 1] = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, depth, key_rows[request], kv_heads, 1);
    auto *v = inputs[3 * request + 2] = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, depth, key_rows[request], kv_heads, 1);
    auto *attention = ggml_flash_attn_ext(context, q, k, v, nullptr,
                                         .0625F, 0.F, 0.F);
    ASSERT_TRUE(ggml_prec_set_acc(attention, GGML_PREC_F32));
    attention->op_params[b::kFlashAttentionMathSlot] =
        b::kFlashAttentionFullF32;
    parts[request] = ggml_reshape_2d(
        context, attention, depth * heads, query_rows[request]);
  }
  auto imports =
      std::make_shared<std::array<std::shared_ptr<ggml_backend_buffer>, 6>>();
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    ggml_set_input(inputs[port]);
    auto buffer = device->allocate_buffer(ggml_nbytes(inputs[port]));
    ASSERT_TRUE(buffer) << buffer.error().message();
    (*imports)[port] = std::shared_ptr<ggml_backend_buffer>(
        *buffer, [owner = *device](ggml_backend_buffer *p) {
          auto active = owner.activate();
          ggml_backend_buffer_free(p);
        });
    ASSERT_EQ(ggml_backend_tensor_alloc(
                  (*imports)[port].get(), inputs[port],
                  ggml_backend_buffer_get_base((*imports)[port].get())),
              GGML_STATUS_SUCCESS);
    const std::vector<float> zeros(ggml_nelements(inputs[port]), 0.F);
    ggml_backend_tensor_set(inputs[port], zeros.data(), 0,
                            zeros.size() * sizeof(float));
  }
  auto *output = ggml_concat(context, parts[0], parts[1], 1);
  ggml_set_output(output);
  auto *graph = ggml_new_graph_custom(context, 64, false);
  ggml_build_forward_expand(graph, output);
  auto native =
      n::NativeGraph::allocate(*device, std::move(*arena), graph, imports);
  ASSERT_TRUE(native) << native.error().message();
  for (int step = 0; step < 4; ++step) {
    std::array<double, 2> row_means{};
    for (std::size_t request = 0; request < parts.size(); ++request) {
      std::vector<float> values(ggml_nelements(inputs[3 * request + 2]));
      for (int head = 0; head < kv_heads; ++head)
        for (int row = 0; row < key_rows[request]; ++row) {
          const auto row_value = float((row % 5) - 2) * .015625F;
          if (head == 0)
            row_means[request] += double(row_value) / key_rows[request];
          for (int column = 0; column < depth; ++column)
            values[(head * key_rows[request] + row) * depth + column] =
                float(step) * .125F + float(request) * .25F + row_value +
                float(head) * .0625F + float(column % 7) * .00390625F;
        }
      ggml_backend_tensor_set(inputs[3 * request + 2], values.data(), 0,
                              values.size() * sizeof(float));
    }
    auto submitted = work->submit(native->get(), native->lease());
    ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
    EXPECT_FALSE(work->recording_ready());
    auto completed = work->wait();
    ASSERT_TRUE(completed.status.ok()) << completed.status.message();
    if (step == 0)
      EXPECT_FALSE(work->recording_ready())
          << "scratch growth must reject saved commands from the cold execution";
    else
      ASSERT_TRUE(work->recording_ready())
          << "stable scratch must permit recording/replay at step " << step;
    std::vector<float> actual(18 * heads * depth);
    ggml_backend_tensor_get(output, actual.data(), 0,
                            actual.size() * sizeof(float));
    // Zero Q/K makes attention uniform. This independent mean includes the
    // different logical KV extents, GQA head mapping, and changing V payload.
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const std::size_t request = i < heads * depth ? 0 : 1;
      const auto head = int(i / depth) % heads;
      const auto column = int(i % depth);
      const double expected = double(step) * .125 + double(request) * .25 +
          row_means[request] + double(head / (heads / kv_heads)) * .0625 +
          double(column % 7) * .00390625;
      ASSERT_TRUE(std::isfinite(actual[i]));
      EXPECT_NEAR(actual[i], expected, 1.e-5) << "step " << step << " element " << i;
    }
  }
}

TEST(GgmlVulkanAttention,
     TaggedNativeDomainRejectsUnqualifiedDepthBeforeExecution) {
  auto device = n::NativeDevice::create();
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  for (const auto depth : {8, 72, 512}) {
    auto arena =
        n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
    ASSERT_TRUE(arena) << arena.error().message();
    auto *context = arena->get();
    auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 1, 2, 1);
    auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 31, 2, 1);
    auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, depth, 31, 2, 1);
    auto *output =
        ggml_flash_attn_ext(context, q, k, v, nullptr, 0.125F, 0.F, 0.F);
    ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
    output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
    for (auto *input : {q, k, v})
      ggml_set_input(input);
    ggml_set_output(output);
    auto *graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto rejected = n::NativeGraph::allocate(*device, std::move(*arena), graph);
    EXPECT_FALSE(rejected) << "native full-F32 tag admitted depth " << depth;
    if (!rejected)
      EXPECT_EQ(rejected.error().code(), hyle::StatusCode::Unimplemented);
  }
}

TEST(GgmlVulkanAttention,
     TaggedColdResourceExhaustionRollsBackOwnedGraphStorage) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  n::NativeOptions options;
  options.quota = &quota;
  auto device = n::NativeDevice::create(options);
  ASSERT_TRUE(device) << device.error().message();
  auto active = device->activate();
  auto arena =
      n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
  ASSERT_TRUE(arena) << arena.error().message();
  auto *context = arena->get();
  auto *q = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
  auto *k = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
  auto *v = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
  auto *output =
      ggml_flash_attn_ext(context, q, k, v, nullptr, 0.125F, 0.F, 0.F);
  ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
  output->op_params[b::kFlashAttentionMathSlot] = b::kFlashAttentionFullF32;
  for (auto *input : {q, k, v})
    ggml_set_input(input);
  ggml_set_output(output);
  auto *graph = ggml_new_graph_custom(context, 16, false);
  ggml_build_forward_expand(graph, output);
  auto rejected = n::NativeGraph::allocate(*device, std::move(*arena), graph);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::ResourceExhausted);
  EXPECT_EQ(quota.used(), 0U);
}

TEST(GgmlVulkanAttention,
     FullF32AndOriginalScalarPipelinesStayDistinctInBothOrders) {
  // N=1/MHA makes both modes select the same scalar tiles. Only the precision
  // key can distinguish the original half-internal shader from full F32.
  for (bool strict_first : {false, true}) {
    auto device = n::NativeDevice::create();
    ASSERT_TRUE(device) << device.error().message();
    auto active = device->activate();
    std::vector<float> normal_before;
    for (bool strict :
         {strict_first, !strict_first, strict_first, !strict_first}) {
      auto arena =
          n::MetadataArena::for_graph(8, 16, hyle::mem::system_allocator());
      ASSERT_TRUE(arena) << arena.error().message();
      auto *context = arena->get();
      auto *query = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 1, 2, 1);
      auto *keys = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
      auto *values = ggml_new_tensor_4d(context, GGML_TYPE_F32, 64, 31, 2, 1);
      auto *output = ggml_flash_attn_ext(context, query, keys, values, nullptr,
                                         0.125F, 0.F, 0.F);
      ASSERT_TRUE(ggml_prec_set_acc(output, GGML_PREC_F32));
      if (strict)
        output->op_params[b::kFlashAttentionMathSlot] =
            b::kFlashAttentionFullF32;
      for (auto *input : {query, keys, values})
        ggml_set_input(input);
      ggml_set_output(output);
      auto *graph = ggml_new_graph_custom(context, 16, false);
      ggml_build_forward_expand(graph, output);
      auto native = n::NativeGraph::allocate(*device, std::move(*arena), graph);
      ASSERT_TRUE(native) << native.error().message();
      auto work = device->create_work();
      ASSERT_TRUE(work) << work.error().message();
      const std::vector<float> q(128, 0.F), k(31 * 128, 0.F),
          v(31 * 128, 131072.F);
      ggml_backend_tensor_set(query, q.data(), 0, q.size() * sizeof(float));
      ggml_backend_tensor_set(keys, k.data(), 0, k.size() * sizeof(float));
      ggml_backend_tensor_set(values, v.data(), 0, v.size() * sizeof(float));
      ASSERT_TRUE(work->submit(native->get(), native->lease()).status.ok());
      ASSERT_TRUE(work->wait().status.ok());
      std::vector<float> result(128);
      ggml_backend_tensor_get(output, result.data(), 0,
                              result.size() * sizeof(float));
      if (strict) {
        for (const auto value : result) {
          ASSERT_TRUE(std::isfinite(value));
          EXPECT_NEAR(value, 131072.F, 0.1F);
        }
      } else if (normal_before.empty()) {
        normal_before = result;
      } else {
        EXPECT_EQ(std::memcmp(result.data(), normal_before.data(),
                              result.size() * sizeof(float)),
                  0);
      }
    }
  }
}
} // namespace
