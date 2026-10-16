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
  VulkanSession(c::Operation operation, std::vector<c::TensorDesc> inputs)
      : context_(take(v::Context::create(options()))), descriptions_(std::move(inputs)) {
    for (const auto &description : descriptions_) inputs_.push_back(take(context_.allocate(description)));
#if defined(HYLE_G08_GRAPH)
    namespace g = hyle::graph;
    auto builder = take(g::GraphBuilder::create());
    std::vector<g::ValueId> produced;
    for (const auto &description : descriptions_)
      produced.push_back(take(builder.add(c::Copy{}, take(builder.input(description)))));
    const auto node = take(builder.add_node(operation, produced));
    for (std::size_t port = 0; port < c::operation_traits(operation).outputs; ++port)
      require(builder.publish(take(builder.add(c::Copy{}, take(builder.result(node, port))))));
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
    for (std::size_t i = 0; i < payloads.size(); ++i)
      HYLE_RETURN_IF_ERROR(context_.upload(inputs_[i], payloads[i]));
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
  std::vector<v::Tensor> inputs_;
  v::Program program_;
  std::array<v::Invocation, 3> slots_;
};
} // namespace sampling_test

int main() {
  try { sampling_test::run<sampling_test::VulkanSession>(); return 0; }
  catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL Vulkan sampling: %s\n", error.what()); return 1;
  }
}
