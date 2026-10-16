#include "topk_bench.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#include <hyle/compute/apple.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#if HYLE_G08_REFERENCE
namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>> g08_original_lower_tensor(
    const Operation &,std::span<const mlx::core::array>,mlx::core::Stream) noexcept;
}
#endif

namespace g08 {
namespace c=hyle::compute;
namespace a=c::apple;
namespace mx=mlx::core;
namespace {
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
class AppleArm final : public Arm {
public:
  explicit AppleArm(const Data &data):data_(data),context_(take(a::Context::create())),
      values_(data.rows*data.count),indices_(values_.size()) {
    const c::Operation operation(c::TopK{.count=std::size_t(data.count)});
    const std::array specs{a::InputSpec{{data.rows,kVocabulary},mx::float32}};
#if HYLE_G08_REFERENCE
    program_=take(context_.compile([operation](const auto &inputs,mx::Stream stream){
      return a::detail::g08_original_lower_tensor(operation,inputs,stream);
    },specs));
#else
    program_=take(context_.prepare(operation,specs));
#endif
    invocation_=take(program_.create_invocation());
    inputs_.emplace_back(data.input.data(),mx::Shape{data.rows,kVocabulary},mx::float32);
    mx::async_eval(inputs_);mx::synchronize(mx::default_stream(mx::Device::gpu));
    for (auto &input:inputs_) input.wait();
    check(invocation_.bind(inputs_));
    std::fprintf(stderr,"# G08_FACT original=%d R=%d K=%d carrier=apple-mlx readback_bytes=%zu per_call_native_arrays=charged final_retained_array=deferred\n",
      HYLE_G08_REFERENCE,data.rows,data.count,values_.size()*8);
  }
  void execute() override {
    auto work=take(invocation_.submit());auto result=take(work.wait());
    const auto arrays=result->arrays();
    if (arrays.size()!=2 || arrays[0].dtype()!=mx::float32 || arrays[1].dtype()!=mx::int32 ||
        arrays[0].size()!=values_.size() || arrays[1].size()!=indices_.size())
      throw std::runtime_error("Apple TopK public output contract differs");
    std::memcpy(values_.data(),arrays[0].data<float>(),values_.size()*sizeof(float));
    std::memcpy(indices_.data(),arrays[1].data<std::int32_t>(),indices_.size()*sizeof(std::int32_t));
  }
  void verify() override {
    for (int cycle=0;cycle<3;++cycle) {execute();verify_outputs(data_,values_,indices_);}
  }
  void finish() override {invocation_={};program_={};check(context_.close());}
private:
  const Data &data_;
  a::Context context_;
  std::vector<mx::array> inputs_;
  a::Program program_;
  a::Invocation invocation_;
  std::vector<float> values_;
  std::vector<std::int32_t> indices_;
};
} // namespace
#if HYLE_G08_REFERENCE
std::unique_ptr<Arm> make_reference(const Data &data) {return std::make_unique<AppleArm>(data);}
#else
std::unique_ptr<Arm> make_candidate(const Data &data) {return std::make_unique<AppleArm>(data);}
#endif
} // namespace g08
