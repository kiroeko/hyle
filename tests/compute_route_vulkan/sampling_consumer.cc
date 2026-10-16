#include "../compute_models/sampling_fixture.h"

#include <cstdlib>
#if defined(HYLE_G08_GRAPH)
#include <hyle/graph/vulkan.h>
#else
#include <hyle/compute/vulkan.h>
#endif

namespace sampling_test {
namespace v = c::vulkan;
class VulkanSession final {
public:
  VulkanSession(c::Operation operation, std::vector<c::TensorDesc> inputs,
                [[maybe_unused]] std::uint32_t published_mask = UINT32_MAX,
                std::size_t byte_origin = 0)
      : context_(take(v::Context::create(options()))), descriptions_(std::move(inputs)), byte_origin_(byte_origin) {
    for (const auto &description : descriptions_) {
      if (byte_origin_ == 0) {
        inputs_.push_back(take(context_.allocate(description)));
        uploads_.push_back(inputs_.back());
      } else {
        const auto bytes = take(description.required_bytes());
        const auto storage = take(c::TensorDesc::strided_bytes(description.data_type(),
            std::array<std::int64_t,1>{std::int64_t(bytes+byte_origin_)},std::array<std::int64_t,1>{1}));
        uploads_.push_back(take(context_.allocate(storage)));
        auto view = take(uploads_.back().view(take(c::View::slice(0,std::int64_t(byte_origin_),
            std::int64_t(take(description.element_count())),std::int64_t(c::data_type_size(description.data_type())))))));
        view = take(view.view(take(c::View::reshape(take(c::TensorShape::create(description.dimensions()))))));
        check(view.description()==description,"Vulkan origin fixture did not retain the dense logical shape");
        inputs_.push_back(std::move(view));
      }
    }
#if defined(HYLE_G08_GRAPH)
    namespace g = hyle::graph;
    auto builder = take(g::GraphBuilder::create());
    std::vector<g::ValueId> produced;
    for (const auto &description : descriptions_) {
      const auto input = take(builder.input(description));
      produced.push_back(byte_origin_ == 0 ? take(builder.add(c::Copy{},input)) :
          take(builder.add(take(c::View::reshape(take(c::TensorShape::create(description.dimensions())))),input)));
    }
    const auto node = take(builder.add_node(operation, produced));
    for (std::size_t port = 0; port < c::operation_traits(operation).outputs; ++port) {
      if (published_mask & (1U << port))
        require(builder.publish(take(builder.add(c::Copy{}, take(builder.result(node, port))))));
    }
    auto logical = take(std::move(builder).finish());
    program_ = take(g::vulkan::compile(context_, logical));
#else
    std::vector<v::Argument> arguments;
    for (const auto &description : descriptions_) arguments.emplace_back(description);
    program_ = take(v::prepare(context_, operation, std::span<const v::Argument>(arguments)));
#endif
    for (auto &slot : slots_) slot = take(program_.create_invocation());
  }
  hyle::Result<Output> run(const Payloads &payloads, int bank) {
    for (std::size_t i = 0; i < payloads.size(); ++i) {
      if (byte_origin_ == 0) HYLE_RETURN_IF_ERROR(context_.upload(inputs_[i],payloads[i]));
      else {
        Bytes storage(take(uploads_[i].description().required_bytes()),std::byte{0xa5});
        std::copy(payloads[i].begin(),payloads[i].end(),storage.begin()+std::ptrdiff_t(byte_origin_));
        HYLE_RETURN_IF_ERROR(context_.upload(uploads_[i],storage));
      }
    }
    auto &slot = slots_[std::size_t(bank)];
    HYLE_RETURN_IF_ERROR(slot.bind(inputs_));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    const auto view = result.get()->tensors();
    std::vector<v::Tensor> retained(view.begin(), view.end());
    auto reader = [this, retained = std::move(retained)] {
      Payloads bytes;
      for (const auto &tensor : retained) {
        Bytes value(take(tensor.description().required_bytes()));
        require(context_.download(tensor, value)); bytes.push_back(std::move(value));
      }
      return bytes;
    };
    return Output{reader(), std::move(reader)};
  }
  void renew_failed_slot(int bank) {
    slots_[std::size_t(bank)] = {};
    slots_[std::size_t(bank)] = take(program_.create_invocation());
  }
  void close() {
    for (auto &slot : slots_) slot = {};
    program_ = {}; require(context_.close());
  }
private:
  static v::ContextOptions options() {
    bool validation = false;
#if defined(_WIN32)
    char *value = nullptr; std::size_t size = 0;
    if (_dupenv_s(&value, &size, "HYLE_VULKAN_VALIDATION") == 0 && value)
      validation = value[0] == '1';
    std::free(value);
#else
    if (const auto *value = std::getenv("HYLE_VULKAN_VALIDATION")) validation = value[0] == '1';
#endif
    return {.validation=validation};
  }
  v::Context context_;
  std::vector<c::TensorDesc> descriptions_;
  std::vector<v::Tensor> inputs_, uploads_;
  std::size_t byte_origin_ = 0;
  v::Program program_;
  std::array<v::Invocation, 3> slots_;
};

void input_origins() {
  constexpr int rows=3,width=17;
  const auto counter=dense(c::DataType::kUInt32,{rows,4});
  for (const std::size_t origin : {1,3,17}) {
    VulkanSession bits(c::sampling_ops::random_bits(5),{counter},UINT32_MAX,origin);
    const auto keys=identities(rows,0xfedcba9876543210ULL);
    check_words(take(bits.run({pack(keys)},0)),keys,5);bits.close();
    for (const auto type : {c::DataType::kFloat32,c::DataType::kFloat16,c::DataType::kBFloat16}) {
      const std::vector<c::TensorDesc> desc{dense(type,{rows,width}),counter};
      VulkanSession aligned(c::sampling_ops::categorical(true),desc);
      VulkanSession shifted(c::sampling_ops::categorical(true),desc,UINT32_MAX,origin);
      for(int step=0;step<5;++step) {
        std::vector<float> logits(rows*width);
        for(std::size_t i=0;i<logits.size();++i)logits[i]=i%5==0?-INFINITY:float(int((i*7+std::size_t(step))%19)-9)*.25F;
        const auto stored=stored_logits(logits,type);
        const auto draws=identities(rows,0xffffffffULL+std::uint64_t(step));
        const auto expected=take(aligned.run({stored,pack(draws)},step%2));
        const auto actual=take(shifted.run({stored,pack(draws)},step%2));
        check(actual.bytes==expected.bytes,"Vulkan unaligned logits/counters changed the same native draw");
        check_categorical(actual,logits,rows,width,true);
      }
      aligned.close();shifted.close();
    }
    const auto desc=dense(c::DataType::kFloat32,{rows,width});
    VulkanSession top(c::Operation(c::TopK{.count=1}),{desc},UINT32_MAX,origin);
    std::vector<float> values(rows*width,0.F);
    for(int r=0;r<rows;++r)values[std::size_t(r*width+width-1)]=float(r+1);
    const auto actual=take(top.run({pack(values)},0));
    const auto ids=unpack<std::int32_t>(actual.bytes[1]);
    check(ids==std::vector<std::int32_t>(rows,width-1),"Vulkan TopK1 lost unaligned native input origin");
    check(unpack<float>(actual.bytes[0])==std::vector<float>({1,2,3}),"Vulkan TopK1 unaligned value bits differ");
    top.close();
  }
  std::puts("PASS Vulkan origins: dense native View at byte1/3/17, raw/categorical F32/F16/BF16/TopK1");
}
} // namespace sampling_test

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--origins-only") {
      sampling_test::input_origins();return 0;
    }
#if defined(HYLE_G08_GRAPH)
    if (argc == 2 && std::string(argv[1]) == "--ports-only") {
      sampling_test::graph_port_liveness<sampling_test::VulkanSession>(); return 0;
    }
#endif
    sampling_test::run<sampling_test::VulkanSession>(argc == 2 ? std::string_view(argv[1]) : std::string_view{});
    if(argc==1)sampling_test::input_origins();
    return 0;
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Vulkan sampling: %s\n", error.what()); return 1;
  }
}
