#pragma once

// Qwen-private tensor storage and cold source/binding API. This borrows the
// original Backend; it owns no Model, Session, request or alternate executor.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../inference/evaluation_failure.h"
#include "hyle/compute/file_stream.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/mem/buffer.h"
#include "cache_precision.h"
#include "ffn_placement.h"
#include "projection_binding.h"
#include "session_state.h"

namespace hyle::inference { class Backend; }

namespace qwen35 {
inline void check(hyle::Status status) {
  if (!status.ok()) throw hyle::inference::EvaluationFailure(std::move(status));
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw hyle::inference::EvaluationFailure(static_cast<hyle::Status>(result.error()));
  return static_cast<T &&>(*result);
}
inline hyle::compute::TensorDesc dense(hyle::compute::DataType type, std::initializer_list<std::int64_t> shape) {
  return take(hyle::compute::TensorDesc::dense(type, std::span(shape.begin(), shape.size())));
}
struct Tensor final {
  hyle::compute::TensorDesc description;
  hyle::mem::Buffer buffer;
  hyle::compute::LeasePool pool;
  hyle::compute::WriteLease writer;
  hyle::compute::ReadLease ready;
  Tensor() = default;
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      // Drop the previous payload and its original domain together, before
      // adopting a replacement. Escaped readers keep that domain's charge.
      ready.reset(); writer.reset(); pool = {}; buffer = {};
      description = other.description;
      buffer = std::move(other.buffer); pool = std::move(other.pool);
      writer = std::move(other.writer); ready = std::move(other.ready);
    }
    return *this;
  }
  Tensor(hyle::mem::MemoryResourceRef resource, hyle::compute::TensorDesc desc, bool caller = true,
         const hyle::compute::ResourceBudget &budget = {})
      : description(desc) {
    if (!caller) return;
    const auto bytes = take(desc.required_bytes());
    if (budget && bytes) {
      pool = take(hyle::compute::LeasePool::create(resource, bytes, {.max_blocks = 1, .resource_budget = budget}));
      writer = take(pool.allocate(bytes));
    } else buffer = take(hyle::mem::Buffer::create(resource, bytes, 256));
  }
  bool caller_owned() const noexcept { return bool(buffer) || bool(writer); }
  hyle::mem::BufferView storage() { return writer ? take(writer.view()) : buffer.view(); }
  hyle::compute::TensorView view() { return take(hyle::compute::TensorView::create(description, storage())); }
  void freeze(const hyle::inference::Backend &backend);
  hyle::compute::ConstTensorView read() {
    return ready ? take(ready.tensor(description)) : view().as_const();
  }
  void zero() { check(storage().fill(std::byte{})); }
  template<class T> void write(std::span<const T> values) {
    check(storage().write(std::as_bytes(values)));
  }
  void dump(const std::filesystem::path &path) {
    std::vector<std::byte> bytes(take(description.required_bytes()));
    check(storage().read(bytes));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write trace: " + path.string());
  }
};

// Source-reader annotations belong only to a loaded physical weight, never
// to activation/literal/state storage. Bindings borrow its stable tensor child.
struct LoadedWeight final {
  Tensor tensor;
  prepared::ProjectionCodec codec = prepared::ProjectionCodec::kUnspecified;
  std::uint32_t group_size = 0;
};

using WeightMeanings = std::map<std::string, prepared::ParameterMeaning>;
struct WeightLoadReport final {
  std::size_t bytes = 0;
  hyle::compute::FileStreamReport stream;
  std::string identity;
  WeightMeanings meanings;
  std::size_t device_bytes = 0, host_bytes = 0;
  std::size_t physical_text_roles = 0, physical_mtp_roles = 0;
  std::size_t canonical_bytes = 0, fp8_groups = 0, import_peak_bytes = 0;
  bool float8_policy = false;
  std::size_t ct_channel_groups = 0, ct_local_groups = 0, ct_block_groups = 0, omitted_cache_roles = 0;
  std::size_t nf4_groups = 0, nf4_original_bytes = 0, nf4_input_copy_bytes = 0, nf4_known_group_peak_bytes = 0;
};
enum class WeightSelection { kText, kTextAndMtp };
struct PreparedWeightSource {
  prepared::Roles roles;
  std::vector<CacheStoragePair> cache_storage;
  cache::KvRowBytes cache_row_bytes;
  double metadata_prepare_seconds = 0;
};
namespace projection_detail {
std::array<std::int64_t, 2> logical_shape(const ProjectionBinding &binding);
}

ModelGeometry read_weight_geometry(const std::filesystem::path &manifest);
prepared::Roles read_weight_roles(const std::filesystem::path &manifest, bool include_mtp);
PreparedWeightSource prepare_weight_source(const std::filesystem::path &manifest, bool include_mtp);
hyle::mem::MemoryResourceRef weight_resource(const hyle::inference::Backend &backend, bool host);
WeightLoadReport load_selected_weight_roles(const prepared::Roles &roles,
    hyle::inference::Backend &backend, std::map<std::string, LoadedWeight> &weights,
    std::vector<hyle::compute::LeasePool> &weight_pools, const FfnPlacement &placement,
    const hyle::compute::ResourceBudget &budget = {});
WeightLoadReport load_weight_bank(const std::filesystem::path &manifest,
    hyle::inference::Backend &backend, std::map<std::string, LoadedWeight> &weights,
    std::vector<hyle::compute::LeasePool> &weight_pools,
    const ModelGeometry &geometry, WeightSelection selection = WeightSelection::kText,
    std::size_t cpu_ffn_layers = 0);
} // namespace qwen35
