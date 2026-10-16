#include "topk_bench.h"

#include <array>
#include <cstdio>
#include <utility>
#include <hyle/compute/prepared_operation.h>
#include <hyle/mem/buffer.h>

#if HYLE_G08_CUDA
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/cuda.h>
#include <hyle/compute/providers/cuda.h>
#else
#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/ggml_cpu.h>
#endif

namespace g08 {
namespace c = hyle::compute;
namespace m = hyle::mem;
#if HYLE_G08_REFERENCE
c::Provider reference_provider() noexcept;
#endif
namespace {
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error(std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

class GenericArm final : public Arm {
public:
  explicit GenericArm(const Data &data) : data_(data), values_(data.rows * data.count),
                                         indices_(values_.size()) {
#if HYLE_G08_REFERENCE
    auto provider = reference_provider();
#elif HYLE_G08_CUDA
    auto provider = c::cuda_native::cccl_provider();
#else
    auto provider = c::ggml_cpu::provider();
#endif
#if HYLE_G08_CUDA
    context_ = take(c::Context::create(take(c::cuda::device(0)), provider));
    device_memory_ = take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice));
    memory_ = device_memory_.ref();
#else
    context_ = take(c::Context::create(c::cpu::device(), provider));
    memory_ = m::system_resource();
#endif
    queue_ = take(context_.create_queue());
    check(queue_.reserve_completions(1));
    const auto description = take(c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 2>{data.rows, kVocabulary}));
    const c::Operation operation(c::TopK{.count=std::size_t(data.count)});
    std::array<c::TensorDesc,2> outputs;
    check(c::infer_output_descriptions(operation, std::span(&description,1), outputs));
    program_ = take(c::prepare(context_, operation, std::span(&description,1), outputs, memory_));
    input_ = take(m::Buffer::create(memory_, take(description.required_bytes()), 256));
    check(input_.view().write(std::as_bytes(std::span(data.input))));
    input_view_[0] = take(c::ConstTensorView::create(description, input_.view()));
    for (std::size_t i=0;i<2;++i) {
      output_[i] = take(m::Buffer::create(memory_, take(outputs[i].required_bytes()),256));
      output_view_[i] = take(c::TensorView::create(outputs[i],output_[i].view()));
    }
    const auto requirement = program_.workspace_requirement();
    if (requirement.size) {
      workspace_ = take(m::Buffer::create(memory_,requirement.size,requirement.alignment));
      invocation_ = take(program_.create_invocation(workspace_.view()));
    } else invocation_ = take(program_.create_invocation());

    // Check the actual ordinary route's cold choice. All measured invocations
    // still use the complete original/current public Provider factory.
#if HYLE_G08_CUDA
    auto ordinary = take(c::cuda::create_context());
    auto ordinary_memory = take(c::cuda::MemoryResource::create(ordinary,c::cuda::MemoryKind::kDevice));
    auto ordinary_program = take(c::prepare(ordinary,operation,std::span(&description,1),outputs,ordinary_memory.ref()));
#else
    auto ordinary = take(c::cpu::create_context());
    auto ordinary_program = take(c::prepare(ordinary,operation,std::span(&description,1),outputs,memory_));
#endif
    if (ordinary_program.provider().id != program_.provider().id ||
        ordinary_program.implementation() != program_.implementation())
      throw std::runtime_error("ordinary and explicit TopK provider/implementation differ");
    const auto identity = program_.provider();
    const auto runtime = program_.runtime_properties();
    const auto persistent = program_.persistent_bytes();
    const auto private_bytes = program_.invocation_private_bytes();
    std::fprintf(stderr,"# G08_FACT original=%d R=%d K=%d provider=%.*s version=%.*s ordinary_match=1 workspace=%zu alignment=%zu persistent_known=%d persistent=%zu invocation_private_known=%d invocation_private=%zu replay=%d blocks_host=%d private_threads=%d capture=%d readback_bytes=%zu\n",
      HYLE_G08_REFERENCE,data.rows,data.count,int(identity.name.size()),identity.name.data(),
      int(identity.version.size()),identity.version.data(),requirement.size,requirement.alignment,
      persistent.has_value(),persistent.value_or(0),private_bytes.has_value(),private_bytes.value_or(0),int(runtime.replay_class),
      runtime.blocks_host,runtime.uses_private_threads,runtime.capture_compatible,values_.size()*8);
    ordinary_program = {}; check(ordinary.drain()); check(ordinary.close());
  }

  void execute() override {
    auto completion = take(invocation_.submit(queue_,input_view_,output_view_));
    check(completion.wait());
    check(output_[0].view().read(std::as_writable_bytes(std::span(values_))));
    check(output_[1].view().read(std::as_writable_bytes(std::span(indices_))));
  }
  void verify() override {
    for (int cycle=0;cycle<3;++cycle) {execute();verify_outputs(data_,values_,indices_);}
  }
  void finish() override {
    invocation_ = {}; program_ = {};
    check(queue_.drain());check(queue_.close());check(context_.drain());check(context_.close());
  }
private:
  const Data &data_;
  c::Context context_;
#if HYLE_G08_CUDA
  c::cuda::MemoryResource device_memory_;
#endif
  m::MemoryResourceRef memory_;
  c::Queue queue_;
  m::Buffer input_;
  std::array<m::Buffer,2> output_;
  m::Buffer workspace_;
  c::PreparedOperation program_;
  c::OperationInvocation invocation_;
  std::array<c::ConstTensorView,1> input_view_;
  std::array<c::TensorView,2> output_view_;
  std::vector<float> values_;
  std::vector<std::int32_t> indices_;
};
} // namespace

#if HYLE_G08_REFERENCE
std::unique_ptr<Arm> make_reference(const Data &data) {return std::make_unique<GenericArm>(data);}
#else
std::unique_ptr<Arm> make_candidate(const Data &data) {return std::make_unique<GenericArm>(data);}
#endif
} // namespace g08
