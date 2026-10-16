#include "../compute_models/sampling_fixture.h"

#include <mlx/ops.h>
#include <mlx/transforms.h>
#if defined(HYLE_G08_GRAPH)
#include <hyle/graph/apple.h>
#else
#include <hyle/compute/apple.h>
#endif

namespace sampling_test {
namespace a = c::apple;
namespace mx = mlx::core;
class AppleSession final {
public:
  AppleSession(c::Operation operation, std::vector<c::TensorDesc> inputs,
               [[maybe_unused]] std::uint32_t published_mask = UINT32_MAX)
      : context_(take(a::Context::create())), descriptions_(std::move(inputs)) {
#if defined(HYLE_G08_GRAPH)
    namespace g = hyle::graph;
    auto builder = take(g::GraphBuilder::create());
    std::vector<g::ValueId> produced;
    for (const auto &description : descriptions_)
      produced.push_back(take(builder.add(c::Copy{}, take(builder.input(description)))));
    const auto node = take(builder.add_node(operation, produced));
    for (std::size_t port = 0; port < c::operation_traits(operation).outputs; ++port) {
      if (published_mask & (1U << port))
        require(builder.publish(take(builder.add(c::Copy{}, take(builder.result(node, port))))));
    }
    auto logical = take(std::move(builder).finish());
    program_ = take(g::apple::compile(context_, logical));
#else
    std::vector<a::InputSpec> specs;
    for (const auto &description : descriptions_) specs.push_back(take(a::input_spec(description)));
    program_ = take(context_.prepare(operation, specs));
#endif
    for (auto &slot : slots_) slot = take(program_.create_invocation());
  }
  hyle::Result<Output> run(const Payloads &payloads, int bank) {
    std::vector<mx::array> arrays;
    for (std::size_t i = 0; i < payloads.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto spec, a::input_spec(descriptions_[i]));
      const auto &bytes = payloads[i];
      // Use the raw byte owner plus native bitcast, so narrow values are not
      // re-rounded or converted by a C++ template constructor.
      auto raw = mx::array(reinterpret_cast<const std::uint8_t *>(bytes.data()),
                            {int(bytes.size())}, mx::uint8);
      arrays.push_back(mx::reshape(mx::view(raw, spec.dtype), spec.shape));
    }
    mx::async_eval(arrays);
    mx::synchronize(mx::default_stream(mx::Device::gpu));
    for (auto &array : arrays) array.wait();
    auto &slot = slots_[std::size_t(bank)];
    HYLE_RETURN_IF_ERROR(slot.bind(arrays));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    const auto view = result.get()->arrays();
    std::vector<mx::array> retained(view.begin(), view.end());
    auto reader = [retained = std::move(retained)] {
      Payloads bytes;
      for (const auto &array : retained) {
        const auto *data = array.data<std::uint8_t>();
        Bytes value(array.nbytes());
        if (!value.empty()) std::memcpy(value.data(), data, value.size());
        bytes.push_back(std::move(value));
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
  a::Context context_;
  std::vector<c::TensorDesc> descriptions_;
  a::Program program_;
  std::array<a::Invocation, 3> slots_;
};
} // namespace sampling_test

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  try {
#if defined(HYLE_G08_GRAPH)
    if (argc == 2 && std::string(argv[1]) == "--ports-only") {
      sampling_test::graph_port_liveness<sampling_test::AppleSession>(); return 0;
    }
#endif
    sampling_test::run<sampling_test::AppleSession>(argc == 2 ? std::string_view(argv[1]) : std::string_view{}); return 0;
  }
  catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Apple sampling: %s\n", error.what()); return 1;
  }
}
