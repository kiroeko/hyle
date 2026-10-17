// Actual Hyle Direct/Queue/device-resource space probe; no performance timing.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/mem/buffer.h"

namespace c = hyle::compute;
namespace mem = hyle::mem;
namespace alloc = c10::cuda::CUDACachingAllocator;

template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

struct Snapshot {
  bool initialized = false;
  std::int64_t allocated = 0, allocated_peak = 0, reserved = 0, reserved_peak = 0;
};
Snapshot snapshot() {
  // Do not initialize or allocate through Torch before Hyle cold prepare.
  // An uninitialized caching allocator has no tracked device allocations.
  if (!alloc::get()->initialized()) return {};
  const auto stats = alloc::getDeviceStats(0);
  return {true, stats.allocated_bytes[0].current, stats.allocated_bytes[0].peak,
                stats.reserved_bytes[0].current, stats.reserved_bytes[0].peak};
}
void print_snapshot(const char* key, const Snapshot& s) {
  std::cout << ",\"" << key << "\":{\"initialized\":" << (s.initialized?"true":"false")
      << ",\"allocated_current_bytes\":" << s.allocated
      << ",\"allocated_peak_bytes\":" << s.allocated_peak
      << ",\"reserved_current_bytes\":" << s.reserved
      << ",\"reserved_peak_bytes\":" << s.reserved_peak << '}';
}

void exercise(std::int64_t length) {
  constexpr std::int64_t batches = 1, query_heads = 8, kv_heads = 2, depth = 256;
  constexpr float value = 1.25f;
  constexpr std::uint16_t value_bf16 = 0x3fa0;
  auto context = take(c::Context::create(take(c::cuda::device(0)), c::torch_cuda::provider()));
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  auto queue = take(context.create_queue());
  const auto q = take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{batches,length,query_heads,depth}));
  const auto kv = take(c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array{batches,length,kv_heads,depth}));
  const std::array descriptions{q,kv,kv};
  const std::array output_descriptions{q};
  std::array<mem::Buffer,3> input_buffers;
  std::array<c::ConstTensorView,3> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto bytes = take(descriptions[i].required_bytes());
    input_buffers[i] = take(mem::Buffer::create(memory.ref(),bytes,256));
    inputs[i] = take(c::ConstTensorView::create(descriptions[i],input_buffers[i].view()));
    const std::vector<std::uint16_t> data(bytes/2, i==2 ? value_bf16 : 0);
    require(input_buffers[i].view().write(std::as_bytes(std::span(data))));
  }
  const auto output_bytes = take(q.required_bytes());
  auto output = take(mem::Buffer::create(memory.ref(),output_bytes,256));
  {
    const std::vector<std::uint16_t> zero_output(output_bytes/2,0);
    require(output.view().write(std::as_bytes(std::span(zero_output))));
  }
  const std::array outputs{take(c::TensorView::create(q,output.view()))};
  const auto after_ports = snapshot();
  const c::Operation operation{c::ScaledDotProductAttention{
      .scale=.0625f, .causal=true, .accumulation_type=c::DataType::kFloat32}};
  auto program = take(c::prepare(context,operation,descriptions,output_descriptions,memory.ref()));
  const bool provider_matches = program.provider().id == c::torch_cuda::kProviderId;
  const auto runtime = program.runtime_properties();
  if (!provider_matches || runtime.mode != c::ProviderRuntimeMode::kSharedQueue ||
      runtime.workspace_owner != c::ProviderWorkspaceOwner::kProvider ||
      runtime.replay_class != c::ProviderReplayClass::kManaged || program.capture_compatible())
    throw std::runtime_error("Hyle did not select the expected Managed Torch shared-Queue contract");
  const auto after_prepare = snapshot();
  auto invocation = take(program.create_invocation());
  const auto before_submit = snapshot();
  alloc::resetPeakStats(0);
  auto completion = take(invocation.submit(queue,inputs,outputs));
  require(completion.wait());
  const auto after_wait = snapshot();
  // Q/K are zero, so each active key has equal weight. Constant nonzero V
  // yields the same constant at every output, including causal edge rows.
  std::vector<std::uint16_t> actual(output_bytes/2);
  require(output.view().read(std::as_writable_bytes(std::span(actual))));
  double max_error = 0;
  bool all_finite = true;
  for (const auto bits : actual) {
    const auto decoded = std::bit_cast<float>(std::uint32_t(bits)<<16);
    all_finite &= std::isfinite(decoded);
    max_error = (std::max)(max_error,std::abs(double(decoded)-value));
  }
  const bool correct = all_finite && max_error == 0;
  const auto q_bytes = take(q.required_bytes()), kv_bytes = take(kv.required_bytes());
  std::cout << std::setprecision(12)
      << "{\"q\":" << length << ",\"kv\":" << length
      << ",\"batch\":1,\"query_heads\":8,\"kv_heads\":2,\"depth\":256"
      << ",\"port_dtype\":\"BF16\",\"accumulator\":\"F32\",\"causal\":\"bottom-right\""
      << ",\"provider_match\":" << (provider_matches?"true":"false")
      << ",\"provider\":\"" << program.provider().name << "\""
      << ",\"provider_version\":\"" << program.provider().version << "\""
      << ",\"torch_allocator\":\"" << alloc::name() << "\""
      << ",\"shared_queue_wait_ok\":true,\"capture_compatible\":false"
      << ",\"provider_private_bytes_known\":" << (program.invocation_private_bytes().has_value()?"true":"false")
      << ",\"ports_requested_bytes\":{\"query\":" << q_bytes << ",\"key\":" << kv_bytes
      << ",\"value\":" << kv_bytes << ",\"output\":" << output_bytes
      << ",\"total\":" << q_bytes+2*kv_bytes+output_bytes << '}'
      << ",\"old_single_f32_scores_bytes\":" << batches*query_heads*length*length*4
      << ",\"torch_first_submit_peak_extra_allocated_bytes\":" << after_wait.allocated_peak-before_submit.allocated
      << ",\"finite\":" << (all_finite?"true":"false")
      << ",\"max_output_error\":" << max_error << ",\"correct\":" << (correct?"true":"false")
      << ",\"timing\":false,\"fresh_process\":true";
  print_snapshot("torch_after_ports",after_ports);
  print_snapshot("torch_after_cold_prepare",after_prepare);
  print_snapshot("torch_before_submit",before_submit);
  print_snapshot("torch_after_wait",after_wait);
  std::cout << "}\n";
  if (!correct) throw std::runtime_error("Hyle BF16 output differs from constant analytical reference");
}

int main(int argc,char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: hyle_space_probe.exe 512|1024|2048|8192");
    const auto length = std::stoll(argv[1]);
    if (length!=512 && length!=1024 && length!=2048 && length!=8192)
      throw std::runtime_error("unsupported space probe shape");
    exercise(length);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
