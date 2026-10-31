#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/storage_owner.h"
#include "hyle/compute/workload.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"
#include "../../inference/evaluation_failure.h"

// Fixed Kimi-K3 KDA0/NoPE MLA3 component input; no downloader/model registry.
namespace hyle::inference::kimi_k3::roles {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace detail {
template<class T> T take(hyle::Result<T> result,const char *where) {
  if(!result)throw hyle::inference::EvaluationFailure(result.error().WithContext(where));
  return std::move(*result);
}
}
struct Identity final { std::string_view repository,revision,variant,file; };
struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  const Identity *identity=nullptr;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    HYLE_ASSIGN_OR_RETURN(auto tensor,lease.tensor(description));
    return g::ConstantBinding{.value=value,.tensor=tensor,.owner=owner,.lease=lease};
  }
};
inline constexpr std::string_view repository = "moonshotai/Kimi-K3";
inline constexpr std::string_view revision = "f831ab66814297da540d832a5235f8e904f29d06";
inline constexpr std::string_view kda_file = "model-00001-of-000096.safetensors";
inline constexpr std::string_view mla_file = "model-00004-of-000096.safetensors";
inline constexpr c::ModelArtifactProfileDescription profile{
    .model_publisher = "moonshotai",
    .model_repository = "moonshotai/Kimi-K3",
    .model_family = "kimi_linear",
    .model_name = "Kimi-K3",
    .model_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .configuration_digest = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
    .artifact_repository = "moonshotai/Kimi-K3",
    .artifact_revision = "f831ab66814297da540d832a5235f8e904f29d06",
    .variant = "official-mxfp4",
    .manifest_digest = "ca7a26533de68addb8d51194afb9841e1ccb9b5b8e6769b3972bb59c996b602f",
    .format_family = "safetensors",
};
enum class Component { kKda, kMla };
enum class Role : std::size_t { kKdaA, kKdaBeta, kKdaDt, kKdaFa, kKdaFb, kKdaGate, kKdaKConv, kKdaK, kKdaNorm, kKdaOutput, kKdaQConv, kKdaQ, kKdaVConv, kKdaV, kMlaInputNorm, kMlaGate, kMlaKvNorm, kMlaKvA, kMlaKvB, kMlaOutput, kMlaQNorm, kMlaQa, kMlaQb, kCount };
struct Spec final {
  std::string_view name;
  c::DataType type;
  std::array<std::int64_t,3> shape;
  std::size_t rank,part,offset,bytes;
  c::TensorDesc description() const {
    return detail::take(c::TensorDesc::dense(type,{shape.data(),rank}),"original Kimi raw role description");
  }
};
extern const std::array<Spec,23> specs;
struct Pins final {
  std::array<std::string_view,4> unions;
  std::array<std::string_view,23> roles;
};
inline constexpr std::array<std::size_t,4> part_bytes{640000,887160832,14336,464392192};
inline constexpr std::array<std::string_view,4> part_names{
    "union-00.bin","union-01.bin","union-02.bin","union-03.bin"};

namespace layout {
inline constexpr std::size_t heads=96,depth=128,latent=512,width=heads*depth,taps=4;
std::vector<std::byte> qkv_convolution(std::span<const std::byte>,std::span<const std::byte>,std::span<const std::byte>);
struct AbsorbedKv final {std::vector<std::byte> key,value;};
AbsorbedKv split_absorbed_kv(std::span<const std::byte>);
}
class Data final {
 public:
  Data(const Data&)=delete; Data& operator=(const Data&)=delete;
  Data(Data&&)=delete; Data& operator=(Data&&)=delete;
  static std::shared_ptr<const Data> load(const std::filesystem::path &,Component,const Pins &);
  Component component() const noexcept { return component_; }
  std::span<const std::byte> raw(Role) const;
  const Identity &identity() const noexcept;
 private:
  explicit Data(Component);
  bool contains(std::size_t role) const noexcept { return (role<14)==(component_==Component::kKda); }
  Component component_;
  std::array<m::FileMapping,4> parts_;
};
struct KdaView final {
  const ReadonlyRole &q, &k, &v, &f_a, &f_b, &beta, &gate, &output;
  const ReadonlyRole &convolution, &a_log, &dt_bias, &output_norm;
};
struct MlaView final {
  const ReadonlyRole &input_norm, &q_a, &q_norm, &q_b, &kv_a, &kv_norm;
  const ReadonlyRole &absorbed_key, &absorbed_value, &gate, &output;
};

struct Kda final {
  std::shared_ptr<const Data> original;
  std::array<ReadonlyRole,12> operands;
  KdaView view() const noexcept;
};
struct Mla final {
  std::shared_ptr<const Data> original;
  std::array<ReadonlyRole,10> operands;
  MlaView view() const noexcept;
};
Kda prepare_kda(std::shared_ptr<const Data>,const c::StorageOwner &);
Mla prepare_mla(std::shared_ptr<const Data>,const c::StorageOwner &);
void validate(const KdaView &);
void validate(const MlaView &);

struct Input final {
  std::filesystem::path directory;
  std::array<std::string,4> union_sha;
  std::array<std::string,23> role_sha;
  Pins pins() const noexcept;
};
Input load_input(const std::filesystem::path &directory);
} // namespace hyle::inference::kimi_k3::roles
