#include "apple_encoded_page_reference.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "general-q8-source.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/fast_primitives.h>
#include <mlx/ops.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Exact original parameter structs/constant indices, authenticated by the
// source materializer. No candidate kernel or native private pimpl is used.
#include "ggml-metal-impl.h"

namespace hyle::bench::resident {
namespace c = compute;
namespace n = c::apple;
namespace mx = mlx::core;

struct OriginalMetalCompletion::State final {
  std::mutex mutex;
  std::vector<NS::SharedPtr<MTL::CommandBuffer>> commands;
  void observe(MTL::CommandBuffer *command) {
    if (!command) throw std::runtime_error("original Metal peer has no command buffer");
    std::lock_guard lock(mutex);
    if (std::none_of(commands.begin(), commands.end(),
        [command](const auto &value) { return value.get() == command; }))
      commands.push_back(NS::RetainPtr(command));
  }
};
OriginalMetalCompletion::OriginalMetalCompletion() : state_(std::make_unique<State>()) {}
OriginalMetalCompletion::~OriginalMetalCompletion() = default;
OriginalMetalCompletion::State &OriginalMetalCompletion::state() noexcept { return *state_; }
Status OriginalMetalCompletion::check_completed() {
  std::lock_guard lock(state_->mutex);
  if (state_->commands.empty())
    return Status::FailedPrecondition("original Metal peer encoded no observed work");
  for (const auto &command : state_->commands) {
    if (command->status() != MTL::CommandBufferStatusCompleted || command->error())
      return Status::Internal("original Metal peer did not complete successfully");
  }
  state_->commands.clear();
  return Status::Ok();
}

namespace {
constexpr int kQueries = 8, kCache = 64, kSimdgroups = 4;
constexpr const char *kLibrary = "hyle_peer_ggml456_general_q8_embed1";
int pad(int value, int multiple) { return (value + multiple - 1) / multiple * multiple; }

struct Request final {
  int queries, keys, query_heads, kv_heads, dk, dv, query_pad;
  bool mask;
  float scale;
};

class OriginalQ8 final : public mx::fast::Custom {
public:
  OriginalQ8(mx::Stream stream, Request request,
             std::shared_ptr<OriginalMetalCompletion> completion)
      : Custom(stream, [](std::vector<mx::array>) -> std::vector<mx::array> {
          throw std::runtime_error("original Q8 peer does not support transformations");
        }), request_(request), completion_(std::move(completion)) {}
  const char *name() const override { return "OriginalGgml456GeneralQ8"; }
  bool is_equivalent(const mx::Primitive &) const override { return false; }
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array> &) override {
    return {{request_.queries, request_.query_heads, request_.dv}};
  }
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::runtime_error("original Metal peer requires GPU");
  }
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    const auto &r = request_;
    if (inputs.size() != 4 || outputs.size() != 1)
      throw std::invalid_argument("original Metal peer port count");
    auto &device = mx::metal::device(stream().device);
    for (const auto &input : inputs) {
      auto *buffer = static_cast<const MTL::Buffer *>(input.buffer().ptr());
      if (!input.flags().row_contiguous || !buffer || input.offset() < 0 ||
          buffer->device() != device.mtl_device() ||
          std::uint64_t(input.offset()) > buffer->length() ||
          input.nbytes() > buffer->length() - std::uint64_t(input.offset()))
        throw std::invalid_argument("original Metal peer actual buffer/range/device");
    }
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    const int key_row = r.dk / 32 * 34, value_row = r.dv / 32 * 34;
    const bool kvpad = r.keys % kCache != 0;
    const int pad_bytes = kCache * ((key_row + value_row) * r.kv_heads +
                                   (r.mask ? 2 * r.query_pad : 0));
    mx::array padding({kvpad ? pad_bytes : 1}, mx::uint8, nullptr, {});
    padding.set_data(mx::allocator::malloc(padding.nbytes()));
    mx::array blocks({r.mask ? (r.queries + 7) / 8 * ((r.keys + 63) / 64) : 1},
                     mx::uint8, nullptr, {});
    blocks.set_data(mx::allocator::malloc(blocks.nbytes()));
    auto library = device.get_library(kLibrary, [] { return std::string(kOriginalMetalQ8Source); });
    auto &encoder = mx::metal::get_command_encoder(stream());
    // Retain every scratch owner before any prefix can be accepted. Pipeline
    // lookup, launch qualification or observer allocation may throw later.
    encoder.add_temporary(padding); encoder.add_temporary(blocks);
    completion_->state().observe(encoder.get_command_buffer());

    if (kvpad) {
      const bool has_mask = r.mask;
      const int cache = kCache;
      const mx::metal::MTLFCList constants{
        {&has_mask, MTL::DataTypeBool, FC_FLASH_ATTN_EXT_PAD},
        {&cache, MTL::DataTypeInt, FC_FLASH_ATTN_EXT_PAD + 25}};
      auto pipeline = device.get_kernel("kernel_flash_attn_ext_pad", library,
          std::string(kLibrary) + "_pad_" + std::to_string(has_mask), constants);
      ggml_metal_kargs_flash_attn_ext_pad args{};
      args.ne11 = r.keys; args.ne_12_2 = r.kv_heads; args.ne_12_3 = 1;
      args.nb11 = key_row; args.nb12 = key_row * r.keys; args.nb13 = args.nb12 * r.kv_heads;
      args.nb21 = value_row; args.nb22 = value_row * r.keys; args.nb23 = args.nb22 * r.kv_heads;
      args.ne31 = r.query_pad; args.ne32 = 1; args.ne33 = 1;
      args.nb31 = 2 * r.keys; args.nb32 = args.nb31 * r.query_pad; args.nb33 = args.nb32;
      encoder.set_compute_pipeline_state(pipeline); encoder.set_bytes(args, 0);
      encoder.set_input_array(inputs[1], 1); encoder.set_input_array(inputs[2], 2);
      encoder.set_input_array(inputs[3], 3); encoder.set_output_array(padding, 4);
      encoder.dispatch_threadgroups(MTL::Size(kCache, r.kv_heads, 1), MTL::Size(32, 1, 1));
    }
    if (r.mask) {
      const int queries = kQueries, cache = kCache;
      const mx::metal::MTLFCList constants{
        {&queries, MTL::DataTypeInt, FC_FLASH_ATTN_EXT_BLK + 24},
        {&cache, MTL::DataTypeInt, FC_FLASH_ATTN_EXT_BLK + 25}};
      auto pipeline = device.get_kernel("kernel_flash_attn_ext_blk", library,
                                        std::string(kLibrary) + "_blk_q8_c64", constants);
      ggml_metal_kargs_flash_attn_ext_blk args{};
      args.ne01 = r.queries; args.ne30 = r.keys; args.ne31 = r.query_pad;
      args.ne32 = 1; args.ne33 = 1;
      args.nb31 = 2 * r.keys; args.nb32 = args.nb31 * r.query_pad; args.nb33 = args.nb32;
      encoder.set_compute_pipeline_state(pipeline); encoder.set_bytes(args, 0);
      encoder.set_input_array(inputs[3], 1); encoder.set_output_array(blocks, 2);
      encoder.dispatch_threadgroups(MTL::Size((r.keys + 63) / 64, (r.queries + 7) / 8, 1),
                                    MTL::Size(32, 1, 1));
    }
    const bool no = false, has_mask = r.mask;
    const int ns10 = r.dk / 32, ns20 = r.dv / 32, simdgroups = kSimdgroups;
    const mx::metal::MTLFCList constants{
      {&has_mask, MTL::DataTypeBool, FC_FLASH_ATTN_EXT},
      {&no, MTL::DataTypeBool, FC_FLASH_ATTN_EXT + 1},
      {&no, MTL::DataTypeBool, FC_FLASH_ATTN_EXT + 2},
      {&no, MTL::DataTypeBool, FC_FLASH_ATTN_EXT + 3},
      {&kvpad, MTL::DataTypeBool, FC_FLASH_ATTN_EXT + 4},
      {&no, MTL::DataTypeBool, FC_FLASH_ATTN_EXT + 10},
      {&ns10, MTL::DataTypeInt, FC_FLASH_ATTN_EXT + 20},
      {&ns20, MTL::DataTypeInt, FC_FLASH_ATTN_EXT + 21},
      {&simdgroups, MTL::DataTypeInt, FC_FLASH_ATTN_EXT + 22}};
    const auto function = "kernel_flash_attn_ext_q8_0_dk" + std::to_string(r.dk) + "_dv" + std::to_string(r.dv);
    const auto key = std::string(kLibrary) + function + "_mask" + std::to_string(has_mask) +
                     "_pad" + std::to_string(kvpad) + "_nsg4_bcm0_ns" +
                     std::to_string(ns10) + "_" + std::to_string(ns20);
    auto pipeline = device.get_kernel(function, library, key, constants);
    const auto smem = std::size_t(pad(2 * (8 * (r.dk + 2 * pad(r.dv, 64) + 256) + 512 * 4), 16));
    if (pipeline->threadExecutionWidth() != 32 || pipeline->maxTotalThreadsPerThreadgroup() < 128 ||
        smem + pipeline->staticThreadgroupMemoryLength() > device.mtl_device()->maxThreadgroupMemoryLength())
      throw std::runtime_error("original Q8 peer pipeline launch qualification");
    ggml_metal_kargs_flash_attn_ext args{};
    args.ne01 = r.queries; args.ne02 = r.query_heads; args.ne03 = 1;
    args.nb01 = 4 * r.dk; args.nb02 = args.nb01 * r.queries; args.nb03 = args.nb02 * r.query_heads;
    args.ne11 = r.keys; args.ne_12_2 = r.kv_heads; args.ne_12_3 = 1;
    args.ns10 = ns10; args.nb11 = key_row; args.nb12 = key_row * r.keys; args.nb13 = args.nb12 * r.kv_heads;
    args.ns20 = ns20; args.nb21 = value_row; args.nb22 = value_row * r.keys; args.nb23 = args.nb22 * r.kv_heads;
    args.ne31 = r.query_pad; args.ne32 = 1; args.ne33 = 1;
    args.nb31 = 2 * r.keys; args.nb32 = args.nb31 * r.query_pad; args.nb33 = args.nb32;
    args.ne1 = r.query_heads; args.ne2 = r.queries; args.ne3 = 1;
    args.scale = r.scale;
    encoder.set_compute_pipeline_state(pipeline); encoder.set_bytes(args, 0);
    encoder.set_input_array(inputs[0], 1); encoder.set_input_array(inputs[1], 2);
    encoder.set_input_array(inputs[2], 3); encoder.set_input_array(inputs[3], 4);
    encoder.set_input_array(inputs[0], 5); encoder.set_input_array(padding, 6);
    encoder.set_input_array(blocks, 7); encoder.set_output_array(outputs[0], 8);
    encoder.set_threadgroup_memory_length(smem, 0);
    encoder.dispatch_threadgroups(MTL::Size((r.queries + 7) / 8, r.query_heads, 1), MTL::Size(32, 4, 1));
  }
private:
  Request request_;
  std::shared_ptr<OriginalMetalCompletion> completion_;
};
}

Result<n::Program> encoded_page_reference_program(const AppleEncodedPeerContext &peer,
    const c::NativePagedBatchProgram &recipe, c::BatchLayoutVersion layout,
    const c::WorkloadContext &workload) {
  if (!peer.context || !peer.completion || !recipe.valid() || !recipe.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("original Metal peer context/schema"));
  const auto cap = recipe.schema().capacity();
  if (cap.key_storage != c::CacheStorage::q8_0().value() || cap.value_storage != cap.key_storage ||
      cap.query_type != c::DataType::kFloat16 || cap.output_type != cap.query_type || cap.page_size != 16)
    return std::unexpected(Status::Unimplemented("original Metal peer finite Q8/F16 domain"));
  std::array<n::InputSpec, 4> specs;
  for (unsigned i = 0; i < 4; ++i) { HYLE_ASSIGN_OR_RETURN(specs[i], n::input_spec(recipe.descriptions()[i])); }
  return n::detail::PagedBatchAccess::prepare_runtime_program(*peer.context,layout,specs,
      [recipe, completion = peer.completion](const c::BatchLayoutVersion& layout,
      const std::vector<mx::array> &inputs, mx::Stream stream) -> Result<std::vector<mx::array>> {
    const auto cap = recipe.schema().capacity(); const auto host = layout.host();
    const int dk = int(cap.qk_head_dim), dv = int(cap.value_head_dim);
    std::vector<mx::array> parts;
    for (unsigned request = 0; request < layout.facts().requests; ++request) {
      const int nq = host.query_indptr[request + 1] - host.query_indptr[request]; if (!nq) continue;
      const int begin = host.kv_indptr[request], end = host.kv_indptr[request + 1];
      const int nk = (end - begin - 1) * 16 + host.last_page_lengths[request];
      std::vector<std::int32_t> rows(std::size_t(nk), 0);
      for (int row = 0; row < nk; ++row) rows[std::size_t(row)] = host.page_indices[begin + row / 16] * 16 + row % 16;
      const mx::array indices(rows.data(), {nk}, mx::int32);
      std::vector<mx::array> native;
      auto q = mx::slice(inputs[0], {host.query_indptr[request], 0, 0},
          {host.query_indptr[request + 1], int(cap.query_heads), dk}, stream);
      native.push_back(mx::contiguous(mx::transpose(mx::astype(q, mx::float32, stream), {1, 0, 2}, stream), false, stream));
      for (unsigned port = 1; port < 3; ++port) {
        const int row_bytes = (port == 1 ? dk : dv) / 32 * 34;
        const auto root = mx::reshape(inputs[port], {int(cap.max_pages * 16), int(cap.kv_heads), row_bytes}, stream);
        const auto gathered = mx::take(root, indices, 0, stream);
        native.push_back(mx::contiguous(mx::transpose(gathered, {1, 0, 2}, stream), false, stream));
      }
      const int query_pad = pad(nq, 8);
      std::vector<float> mask(std::size_t(query_pad) * nk, -65504.F);
      if (recipe.causal()) for (int row = 0; row < nq; ++row) for (int j = 0; j < nk; ++j) {
        const int position = nk - nq + row;
        if (j <= position && (recipe.window_left() < 0 || j >= std::max(0, position - recipe.window_left())))
          mask[std::size_t(row) * nk + j] = 0;
      }
      if (recipe.causal()) native.push_back(mx::astype(
          mx::array(mask.data(), mx::Shape{query_pad, nk}, mx::float32), mx::float16, stream));
      else native.emplace_back(std::uint8_t(0), mx::uint8);
      const Request shape{nq, nk, int(cap.query_heads), int(cap.kv_heads), dk, dv, query_pad,
                          recipe.causal(), recipe.scale()};
      auto result = mx::array({nq, int(cap.query_heads), dv}, mx::float32,
          std::make_shared<OriginalQ8>(stream, shape, completion), native);
      parts.push_back(mx::astype(result, mx::float16, stream));
    }
    if (parts.empty()) return std::unexpected(Status::InvalidArgument("empty original Metal peer batch"));
    if (parts.size() == 1) return parts;
    return std::vector{mx::concatenate(parts, 0, stream)};
  }, workload, recipe.causal(), recipe.window_left());
}
}
