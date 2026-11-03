#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "dspark_component_profiles.h"
#include "dspark_original_draft_fragment.h"
#include "dspark_original_formal_roles.h"

// Named formal parameter wiring into one ordinary Fragment. Source ownership
// and ready device leases are shared by target/draft/vocabulary; construction
// neither uploads weights nor executes model operations. The caller retains
// the existing four-phase router cut, native runner and StateHead.
namespace dspark_original_draft_program {
namespace roles=dspark_original_formal_roles;
namespace math=dspark_original_draft_fragment;
namespace common=dspark_formal_fragment;
namespace sem=math::sem;
namespace c=math::c;
namespace base=dspark_component_profiles;
using Value=math::Value;
struct Constant final {Value value;roles::ReadonlyRole role;};
template<class Visit> void materialize(sem::Fragment &f,Visit visit) {
  const auto actual_values=f.descriptions.size();
  std::vector<std::optional<Value>> copies(actual_values);
  for(auto &value:f.outputs) {
    auto &ready=copies.at(value.ordinal());
    if(!ready)ready=common::copy(f,value);
    value=*ready;
  }
  const auto observe=[&](Value &value) {
    const auto original=value.ordinal();if(original>=actual_values)return;
    auto &ready=copies[original];
    if(!ready) {ready=common::copy(f,value);f.outputs.push_back(*ready);}
    value=*ready;
  };
  visit(observe);
}
Value immutable(sem::Fragment &f,std::vector<Constant> &constants,const roles::ReadonlyRole &role,
    const c::TensorDesc &expected);
const c::ModelArtifactProfileDescription *checked_profile(const roles::Data &data,
    const c::ModelArtifactProfileDescription *profile);
struct Preparation final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10001,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
};
Preparation prepare_output_a(const roles::Bound &bound,std::uint64_t region=0x0ca10001,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile);
struct Recipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10100,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  math::Inputs inputs;
  math::Result result;
};
inline constexpr std::size_t kSemanticInputs=6,kInputs=18,kSemanticOutputs=7;
// Exactly the older complete-draft semantic port order, with original storage
// boundaries visible in each output descriptor rather than silently widened.
Recipe build(const roles::Bound &bound,const std::array<roles::ReadonlyRole,3> &prepared_oa,
    std::uint64_t region=0x0ca10100,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile);

struct MainRecipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10200,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  std::array<Value,3> captures;
  math::MainPorts ports;
};
MainRecipe build_main_projection(const roles::Bound &bound,std::int64_t rows,std::uint64_t region=0x0ca10200,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile);
struct HistoryRecipe final {
  sem::Fragment fragment{.profile=&base::pro_profile,.region=0x0ca10300,.quantized=true};
  std::shared_ptr<const roles::Data> source;
  std::vector<Constant> constants;
  Value main,positions,write_rows;
  std::array<Value,3> histories;
  math::MainHistoryPorts ports;
};
HistoryRecipe build_main_history(const roles::Bound &bound,std::int64_t rows,std::uint64_t region=0x0ca10300,
    const c::ModelArtifactProfileDescription *profile=&base::pro_profile);
} // namespace dspark_original_draft_program
