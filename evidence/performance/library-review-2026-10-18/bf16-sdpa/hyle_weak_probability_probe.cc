#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/mem/buffer.h"

namespace c = hyle::compute;
namespace mem = hyle::mem;

template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
float decode(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits)<<16);
}
std::uint16_t encode(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  return std::uint16_t((bits + 0x7fffU + ((bits>>16)&1U))>>16);
}
// Exact nearest BF16 neighbour selection from this positive FP64 oracle.
// The FP32 conversion only locates the neighbours; distances use FP64, so a
// possible double-rounding midpoint cannot decide the final rounded result.
std::uint16_t round_positive_oracle(double value) {
  auto lo = std::uint16_t(std::bit_cast<std::uint32_t>(float(value))>>16);
  while (double(decode(lo)) > value) --lo;
  while (double(decode(std::uint16_t(lo+1))) <= value) ++lo;
  const double down = value-double(decode(lo));
  const double up = double(decode(std::uint16_t(lo+1)))-value;
  return down < up || (down == up && (lo&1U)==0) ? lo : std::uint16_t(lo+1);
}

void exercise() {
  constexpr std::int64_t batch=1, nq=64, nk=1024, hq=8, hkv=2, depth=256;
  constexpr float scale=.0625f;
  const auto one=encode(1.f), weak_key=encode(-256.f), weak_value=encode(60000.f);
  const auto qdesc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array{batch,nq,hq,depth}));
  const auto kvdesc=take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array{batch,nk,hkv,depth}));
  const std::array descriptions{qdesc,kvdesc,kvdesc};
  const std::array output_desc{qdesc};
  auto context=take(c::Context::create(take(c::cuda::device(0)),c::torch_cuda::provider()));
  auto memory=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  auto queue=take(context.create_queue());
  std::array<mem::Buffer,3> buffers;
  std::array<c::ConstTensorView,3> inputs;
  for (std::size_t port=0;port<3;++port) {
    const auto bytes=take(descriptions[port].required_bytes());
    std::vector<std::uint16_t> values(bytes/2,0);
    const auto rows=port==0?nq:nk, heads=port==0?hq:hkv;
    for (std::int64_t s=0;s<rows;++s)
      for (std::int64_t h=0;h<heads;++h) {
        const auto offset=std::size_t((s*heads+h)*depth);
        if (port==0) values[offset]=one;
        if (port==1 && s!=0) values[offset]=weak_key;
        if (port==2 && s!=0)
          std::fill_n(values.begin()+std::ptrdiff_t(offset),std::size_t(depth),weak_value);
      }
    buffers[port]=take(mem::Buffer::create(memory.ref(),bytes,256));
    require(buffers[port].view().write(std::as_bytes(std::span(values))));
    inputs[port]=take(c::ConstTensorView::create(descriptions[port],buffers[port].view()));
  }
  auto output=take(mem::Buffer::create(memory.ref(),take(qdesc.required_bytes()),256));
  std::vector<std::uint16_t> actual(output.size()/2,0);
  require(output.view().write(std::as_bytes(std::span(actual))));
  const std::array outputs{take(c::TensorView::create(qdesc,output.view()))};
  auto program=take(c::prepare(context,c::Operation(c::ScaledDotProductAttention{
      .scale=scale,.causal=true,.accumulation_type=c::DataType::kFloat32}),
      descriptions,output_desc,memory.ref()));
  if (program.provider().id!=c::torch_cuda::kProviderId ||
      program.runtime_properties().mode!=c::ProviderRuntimeMode::kSharedQueue)
    throw std::runtime_error("Expected actual Torch CUDA shared-Queue program");
  auto invocation=take(program.create_invocation());
  auto completion=take(invocation.submit(queue,inputs,outputs));
  require(completion.wait());
  require(output.view().read(std::as_writable_bytes(std::span(actual))));

  const double weak_score=double(decode(one))*double(decode(weak_key))*double(scale);
  const double weak_exp=std::exp(weak_score);
  std::array<double,nq> oracle{};
  std::array<std::uint16_t,nq> rounded{};
  double max_abs=0,max_relative=0,max_rounding_abs=0,max_rounding_relative=0;
  double max_vs_rounded_abs=0,max_vs_rounded_relative=0;
  int max_vs_rounded_ulps=0;
  std::size_t finite_elements=0,equal_rounded_elements=0;
  bool row_channels_identical=true;
  for (std::int64_t r=0;r<nq;++r) {
    const double weak_count=double(nk-nq+r);
    oracle[r]=weak_count*weak_exp*double(decode(weak_value))/(1.0+weak_count*weak_exp);
    rounded[r]=round_positive_oracle(oracle[r]);
    const double rounded_value=decode(rounded[r]);
    const double rounding_error=std::abs(rounded_value-oracle[r]);
    max_rounding_abs=(std::max)(max_rounding_abs,rounding_error);
    max_rounding_relative=(std::max)(max_rounding_relative,rounding_error/std::abs(oracle[r]));
    const auto first=actual[std::size_t(r*hq*depth)];
    for (std::int64_t j=0;j<hq*depth;++j) {
      const auto bits=actual[std::size_t(r*hq*depth+j)];
      const double candidate=decode(bits);
      if (std::isfinite(candidate)) ++finite_elements;
      else throw std::runtime_error("Nonfinite Hyle BF16 weak-probability output");
      row_channels_identical &= bits==first;
      if (bits==rounded[r]) ++equal_rounded_elements;
      const double error=std::abs(candidate-oracle[r]);
      max_abs=(std::max)(max_abs,error);
      max_relative=(std::max)(max_relative,error/std::abs(oracle[r]));
      const double rounded_error=std::abs(candidate-rounded_value);
      max_vs_rounded_abs=(std::max)(max_vs_rounded_abs,rounded_error);
      max_vs_rounded_relative=(std::max)(max_vs_rounded_relative,rounded_error/std::abs(rounded_value));
      max_vs_rounded_ulps=(std::max)(max_vs_rounded_ulps,std::abs(int(bits)-int(rounded[r])));
    }
  }
  std::cout<<std::setprecision(17)
      <<"{\"case\":\"bf16-weak-probability\",\"batch\":1,\"q\":64,\"kv\":1024,\"query_heads\":8,\"kv_heads\":2,\"depth\":256"
      <<",\"causal\":\"bottom-right\",\"scale\":"<<scale
      <<",\"query_d0_actual\":"<<decode(one)<<",\"weak_key_d0_actual\":"<<decode(weak_key)
      <<",\"weak_value_requested\":60000,\"weak_value_bf16_actual\":"<<decode(weak_value)
      <<",\"strong_key_d0\":0,\"strong_value\":0,\"weak_logit\":"<<weak_score<<",\"weak_exp_fp64\":"<<weak_exp
      <<",\"provider_match\":true,\"shared_queue_wait_ok\":true,\"provider_version\":\""<<program.provider().version<<"\""
      <<",\"elements\":"<<actual.size()<<",\"finite_elements\":"<<finite_elements
      <<",\"row_channels_identical\":"<<(row_channels_identical?"true":"false")
      <<",\"candidate_max_abs_vs_fp64\":"<<max_abs<<",\"candidate_max_relative_vs_fp64\":"<<max_relative
      <<",\"rounded_oracle_max_abs_vs_fp64\":"<<max_rounding_abs
      <<",\"rounded_oracle_max_relative_vs_fp64\":"<<max_rounding_relative
      <<",\"candidate_max_abs_vs_rounded_oracle\":"<<max_vs_rounded_abs
      <<",\"candidate_max_relative_vs_rounded_oracle\":"<<max_vs_rounded_relative
      <<",\"candidate_max_bf16_ulps_vs_rounded_oracle\":"<<max_vs_rounded_ulps
      <<",\"candidate_equal_rounded_oracle_elements\":"<<equal_rounded_elements
      <<",\"timing\":false,\"numerical_tolerance_applied\":false,\"rows\":[";
  for (std::int64_t r=0;r<nq;++r) {
    if (r) std::cout<<',';
    std::cout<<"{\"row\":"<<r<<",\"weak_count\":"<<nk-nq+r
        <<",\"fp64\":"<<oracle[r]<<",\"bf16_rounded_oracle\":"<<decode(rounded[r])
        <<",\"candidate_first_channel\":"<<decode(actual[std::size_t(r*hq*depth)])<<'}';
  }
  std::cout<<"]}\n";
}
int main() {
  try {exercise();return 0;}
  catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
