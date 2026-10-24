#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "qwen_bfloat_original_reference.h"
#include "dspark_original_expert_reference.h"

// Independent original-byte BF16 certificates. The executed precision policy
// is explicit BF16 activation storage, not a claim of every peer's F32-A path.
// MXFP4 stays on dspark_original_expert_reference with actual global IDs.
namespace dspark_bfloat_sidecar_reference {
namespace bf=qwen_bfloat_original_reference;
inline std::vector<std::uint16_t> stored_input(std::span<const float> actual) {
  return bf::store(actual);
}
inline std::vector<float> decoded_reference(std::span<const std::byte> original,
    std::size_t outputs,std::size_t width,std::span<const float> actual_input) {
  if(!outputs||!width||outputs>std::numeric_limits<std::size_t>::max()/width/2||
      original.size()!=outputs*width*2||actual_input.empty()||actual_input.size()%width||
      actual_input.size()/width>std::numeric_limits<std::size_t>::max()/outputs)
    throw std::runtime_error("S03 independent BF16 projection extent differs");
  const auto input=stored_input(actual_input);std::vector<float> result(input.size()/width*outputs);
  for(std::size_t row=0;row<input.size()/width;++row)
    for(std::size_t output=0;output<outputs;++output)
      result[row*outputs+output]=float(bf::detail::dot(std::span(input).subspan(row*width,width),
          original.subspan(output*width*2,width*2)).center);
  return result;
}
inline bool expect_projection(std::span<const float> actual_raw,std::span<const float> actual_input,
    std::span<const std::byte> original,std::size_t outputs,std::size_t width,
    std::string_view label,bool full,std::span<const std::int32_t> selected={}) {
  const auto input=stored_input(actual_input);
  return bf::expect_projection(actual_raw,input,original,outputs,width,label,full,selected);
}
inline std::vector<float> decoded_grouped_reference(std::span<const std::byte> original,
    std::size_t groups,std::size_t outputs,std::size_t width,std::span<const float> actual_input) {
  if(!groups||!outputs||!width||groups>std::numeric_limits<std::size_t>::max()/outputs/width/2||
      original.size()!=groups*outputs*width*2||actual_input.empty()||actual_input.size()%(groups*width)||
      actual_input.size()/(groups*width)>std::numeric_limits<std::size_t>::max()/groups/outputs)
    throw std::runtime_error("S03 independent grouped BF16 projection extent differs");
  const auto input=stored_input(actual_input);const auto rows=input.size()/(groups*width);
  std::vector<float> result(rows*groups*outputs);
  for(std::size_t row=0;row<rows;++row)for(std::size_t group=0;group<groups;++group)
    for(std::size_t output=0;output<outputs;++output) {
      const auto dot=bf::detail::dot(std::span(input).subspan((row*groups+group)*width,width),
          original.subspan((group*outputs+output)*width*2,width*2));
      // This callback models the actual BF16 BMM result, not an observable
      // raw F32 port. The independent interval check below is authoritative.
      result[(row*groups+group)*outputs+output]=bf::widen(bf::round(float(dot.center)));
    }
  return result;
}
inline bool expect_output_a(std::span<const std::uint16_t> actual_stored,
    std::span<const float> actual_input,std::span<const std::byte> original,
    std::size_t groups,std::size_t outputs,std::size_t width,std::string_view label,bool full) {
  if(!groups||!outputs||!width||groups>std::numeric_limits<std::size_t>::max()/outputs/width/2||
      original.size()!=groups*outputs*width*2||actual_input.empty()||actual_input.size()%(groups*width)||
      actual_input.size()/(groups*width)>std::numeric_limits<std::size_t>::max()/groups/outputs||
      actual_stored.size()!=actual_input.size()/(groups*width)*groups*outputs) {
    ADD_FAILURE()<<label<<" BF16 output-a extent";return false;
  }
  const auto input=stored_input(actual_input);const auto rows=input.size()/(groups*width);
  for(std::size_t row=0;row<rows;++row)for(std::size_t group=0;group<groups;++group) {
    const auto check=[&](std::size_t output) {
      const auto dot=bf::detail::dot(std::span(input).subspan((row*groups+group)*width,width),
          original.subspan((group*outputs+output)*width*2,width*2));
      const auto low=bf::round(std::nextafter(float(dot.center-dot.bound),-std::numeric_limits<float>::infinity()));
      const auto high=bf::round(std::nextafter(float(dot.center+dot.bound),std::numeric_limits<float>::infinity()));
      const auto actual=bf::widen(actual_stored[(row*groups+group)*outputs+output]);
      if(!std::isfinite(actual)||actual<bf::widen(low)||actual>bf::widen(high)) {
        ADD_FAILURE()<<label<<" actual BF16["<<row<<','<<group<<','<<output<<"]="<<actual
            <<" center="<<dot.center<<" derived F32 bound="<<dot.bound;return false;
      }
      return true;
    };
    if(full) {for(std::size_t output=0;output<outputs;++output)if(!check(output))return false;}
    else {for(std::size_t sample=0;sample<7;++sample)if(!check(sample*(outputs-1)/6))return false;}
  }
  return true;
}
inline bool expect_store(std::span<const std::uint16_t> actual_stored,
    std::span<const float> actual_raw,std::string_view label) {
  return bf::expect_store(actual_stored,actual_raw,label);
}
// The original certificate accepts one complete bank. Applying it to one
// authenticated selected-expert span with local ID zero preserves every
// arithmetic bound while avoiding a nonexistent full-domain host buffer.
template<class Source,class Leg>
inline void expect_selected_projection(std::span<const float> actual,
    std::span<const float> actual_bfloat_input,std::span<const std::int32_t> global_ids,
    const Source &source,std::size_t stage,Leg leg,std::size_t outputs,std::size_t width,
    dspark_original_expert_reference::ActivationCodec codec,std::string_view label,
    bool full,std::size_t slots=6) {
  if(!outputs||!width||!slots||global_ids.empty()||global_ids.size()%slots||
      global_ids.size()>std::numeric_limits<std::size_t>::max()/outputs||
      actual.size()!=global_ids.size()*outputs||actual_bfloat_input.empty()||actual_bfloat_input.size()%width||
      (actual_bfloat_input.size()/width!=global_ids.size()&&
       actual_bfloat_input.size()/width!=global_ids.size()/slots))
    throw std::runtime_error("S03 selected original MXFP4 observation extent differs");
  constexpr std::array<std::int32_t,1> local_id{0};
  const bool per_slot=actual_bfloat_input.size()/width==global_ids.size();
  for(std::size_t slot=0;slot<global_ids.size();++slot) {
    const auto original=source.expert_bytes(stage,leg,global_ids[slot]);
    const auto input=actual_bfloat_input.subspan((per_slot?slot:slot/slots)*width,width);
    const auto values=actual.subspan(slot*outputs,outputs);
    const auto name=std::string(label)+" global="+std::to_string(global_ids[slot])+" slot="+std::to_string(slot);
    if(full)dspark_original_expert_reference::expect_selected_projection(
        values,original,1,outputs,width,input,local_id,codec,name,1);
    else {
      const auto row_bytes=width/32*17;
      if(width%32||original.size()!=outputs*row_bytes)
        throw std::runtime_error("S03 selected original MXFP4 byte row differs");
      for(std::size_t sample=0;sample<7;++sample) {
        const auto output=sample*(outputs-1)/6;
        dspark_original_expert_reference::expect_selected_projection(values.subspan(output,1),
            original.subspan(output*row_bytes,row_bytes),1,1,width,input,local_id,codec,name,1);
      }
    }
  }
}
} // namespace dspark_bfloat_sidecar_reference
