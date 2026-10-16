#include "topk_bench.h"
#include <array>
#include <cstdio>
#include <utility>
#include <hyle/compute/vulkan.h>

namespace g08 {
namespace c=hyle::compute;
#if HYLE_G08_REFERENCE
namespace v=c::g08_original_vulkan;
#else
namespace v=c::vulkan;
#endif
namespace {
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
class VulkanArm final : public Arm {
public:
  explicit VulkanArm(const Data &data):data_(data),context_(take(v::Context::create({.validation=false}))),
      values_(data.rows*data.count),indices_(values_.size()) {
    const auto description=take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t,2>{data.rows,kVocabulary}));
    inputs_[0]=take(context_.allocate(description));
    check(context_.upload(inputs_[0],std::as_bytes(std::span(data.input))));
    const std::array arguments{v::Argument(description)};
    program_=take(v::prepare(context_,c::Operation(c::TopK{.count=std::size_t(data.count)}),arguments));
    invocation_=take(program_.create_invocation());check(invocation_.bind(inputs_));
    std::fprintf(stderr,"# G08_FACT original=%d R=%d K=%d carrier=vulkan ordinary_prepare=1 readback_bytes=%zu download_staging_and_fences=charged\n",
      HYLE_G08_REFERENCE,data.rows,data.count,values_.size()*8);
  }
  void execute() override {
    auto work=take(invocation_.submit());auto result=take(work.wait());
    const auto tensors=result->tensors();
    if (tensors.size()!=2 || tensors[0].description().data_type()!=c::DataType::kFloat32 ||
        tensors[1].description().data_type()!=c::DataType::kInt32 ||
        take(tensors[0].description().required_bytes())!=values_.size()*4 ||
        take(tensors[1].description().required_bytes())!=indices_.size()*4)
      throw std::runtime_error("Vulkan TopK public output contract differs");
    check(context_.download(tensors[0],std::as_writable_bytes(std::span(values_))));
    check(context_.download(tensors[1],std::as_writable_bytes(std::span(indices_))));
  }
  void verify() override {
    for (int cycle=0;cycle<3;++cycle) {execute();verify_outputs(data_,values_,indices_);}
  }
  void finish() override {invocation_={};program_={};check(context_.close());}
private:
  const Data &data_;
  v::Context context_;
  std::array<v::Tensor,1> inputs_;
  v::Program program_;
  v::Invocation invocation_;
  std::vector<float> values_;
  std::vector<std::int32_t> indices_;
};
} // namespace
#if HYLE_G08_REFERENCE
std::unique_ptr<Arm> make_reference(const Data &data) {return std::make_unique<VulkanArm>(data);}
#else
std::unique_ptr<Arm> make_candidate(const Data &data) {return std::make_unique<VulkanArm>(data);}
#endif
} // namespace g08
