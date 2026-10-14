#include <hyle/compute/apple_memory.h>
#include <hyle/mem/allocator.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace n = c::apple;
namespace mx = mlx::core;
namespace mem = hyle::mem;
using hyle::Status;
Status require(bool ok, const char *message) {
  return ok ? Status::Ok() : Status::Internal(message);
}
struct Metadata final {
  std::atomic<std::size_t> calls{0};
  mem::AllocatorRef allocator() noexcept {
    return {
      [](void *p, std::size_t bytes, std::size_t alignment) noexcept {
        static_cast<Metadata *>(p)->calls.fetch_add(1, std::memory_order_relaxed);
        return mem::system_allocator().allocate(bytes, alignment);
      },
      [](void *, void *p, std::size_t bytes, std::size_t alignment) noexcept {
        mem::system_allocator().deallocate(p, bytes, alignment);
      },
      [](void *s, void *p, std::size_t old_bytes, std::size_t new_bytes,
         std::size_t alignment) noexcept {
        static_cast<Metadata *>(s)->calls.fetch_add(1, std::memory_order_relaxed);
        return mem::system_allocator().reallocate(p, old_bytes, new_bytes, alignment);
      }, this};
  }
};
class SpoofResource final : public mem::MemoryResource {
public:
  explicit SpoofResource(mem::MemoryDomainId domain) : domain_(domain) {}
  std::string_view name() const noexcept override { return "spoofed Apple domain"; }
  mem::MemoryProperties properties() const noexcept override {
    auto result = mem::system_resource().properties();
    result.domain = domain_;
    return result;
  }
  hyle::Result<mem::ResourceAllocation> allocate(std::size_t bytes,
                                               std::size_t alignment) noexcept override {
    return mem::system_resource().resource().allocate(bytes, alignment);
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    mem::system_resource().resource().deallocate(allocation);
  }
private:
  mem::MemoryDomainId domain_;
};
Status run() {
  Metadata metadata;
  HYLE_ASSIGN_OR_RETURN(auto context, n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto other_device, n::MemoryResource::create(context));
  struct Close final {
    n::Context &context;
    ~Close() { context.close().IgnoreError(); }
  } close{context};
  const std::array specs{n::InputSpec{{1, 2, 256}, mx::float16}};
  HYLE_ASSIGN_OR_RETURN(auto producer, context.prepare(c::Unary{c::UnaryKind::kAbs}, specs));
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(
      c::DataType::kFloat16, std::array<std::int64_t, 3>{1, 2, 256}));
  HYLE_ASSIGN_OR_RETURN(auto producer_slot, producer.create_invocation());
  const auto bytes = desc.required_bytes().value();
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(device.ref(), 3 * bytes + 512));
  const std::array copy_specs{c::CopyResourceSpec{device.ref(), bytes},
                              c::CopyResourceSpec{device.ref(), bytes}};
  HYLE_ASSIGN_OR_RETURN(auto program, n::CopyBatchProgram::prepare(
      context, copy_specs, {.allocator = metadata.allocator()}));
  std::array<n::CopyBatchInvocation, 2> slots;
  for (auto &slot : slots) {
    HYLE_ASSIGN_OR_RETURN(slot, n::CopyBatchInvocation::create(program));
  }
  const std::array small_specs{c::CopyResourceSpec{device.ref(), bytes - 4},
                               c::CopyResourceSpec{device.ref(), bytes}};
  HYLE_ASSIGN_OR_RETURN(auto small_program, n::CopyBatchProgram::prepare(context, small_specs));
  HYLE_ASSIGN_OR_RETURN(auto small, n::CopyBatchInvocation::create(small_program));
  HYLE_ASSIGN_OR_RETURN(auto other_pool, c::LeasePool::create(other_device.ref(), bytes + 512));
  HYLE_ASSIGN_OR_RETURN(auto other_writer, other_pool.allocate(bytes));
  HYLE_RETURN_IF_ERROR(require(slots[0].bind_mutable({1}, std::move(other_writer)).IsInvalidArgument()
      && other_writer.valid(), "foreign writer accepted or consumed"));
  HYLE_ASSIGN_OR_RETURN(auto other_read, other_writer.publish());
  HYLE_RETURN_IF_ERROR(require(slots[0].bind_source({0}, other_read).IsInvalidArgument(),
                              "ordinary foreign native source bypassed exact identity"));
  SpoofResource spoof(device.ref().domain());
  HYLE_ASSIGN_OR_RETURN(auto spoof_buffer, mem::Buffer::create(mem::MemoryResourceRef(spoof), bytes));
  HYLE_ASSIGN_OR_RETURN(auto spoof_read, c::ReadLease::adopt(std::move(spoof_buffer)));
  HYLE_RETURN_IF_ERROR(require(slots[0].bind_source({0}, spoof_read).IsInvalidArgument(),
                              "public domain spoof accepted as native export"));
  HYLE_ASSIGN_OR_RETURN(auto host_pool, c::LeasePool::create(mem::system_resource(), bytes + 512));
  HYLE_ASSIGN_OR_RETURN(auto host_writer, host_pool.allocate(bytes));
  HYLE_ASSIGN_OR_RETURN(auto host_read, host_writer.publish());
  HYLE_RETURN_IF_ERROR(require(slots[0].bind_source({0}, host_read).IsInvalidArgument(),
                              "foreign host source accepted"));
  const std::array<std::uint16_t, 4> expected{0x3000, 0x3400, 0x3600, 0x3800};
  c::ReadLease previous;
  std::uint16_t previous_bits = 0;
  for (unsigned step = 0; step < 64; ++step) {
    auto source = mx::full(mx::Shape{1, 2, 256}, -float(step % 4 + 1) / 8.F, mx::float16);
    mx::async_eval({source});
    mx::synchronize(mx::default_stream(mx::Device::gpu));
    source.wait();
    const std::array inputs{source};
    HYLE_RETURN_IF_ERROR(producer_slot.bind(inputs));
    c::ReadLease ready;
    {
      HYLE_ASSIGN_OR_RETURN(auto submitted, producer_slot.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, submitted.wait());
      HYLE_ASSIGN_OR_RETURN(ready, device.import_readonly(result->arrays()[0], desc));
    }
    HYLE_RETURN_IF_ERROR(require(ready.view()->resource() != device.ref() &&
      (!previous.valid() || ready.view()->resource() != previous.view()->resource()),
      "readonly exports lost distinct original owners"));
    HYLE_RETURN_IF_ERROR(require(small.bind_source({0}, ready).IsInvalidArgument() && ready.valid(),
                                "over-capacity import accepted or consumed"));
    if (previous.valid()) {
      std::array<std::uint16_t, 512> old{};
      HYLE_RETURN_IF_ERROR(previous.view()->read(std::as_writable_bytes(std::span(old))));
      for (auto value : old)
        HYLE_RETURN_IF_ERROR(require(value == previous_bits, "producer reuse changed old export"));
    }
    auto &copy = slots[step % 2];
    HYLE_RETURN_IF_ERROR(copy.reset());
    HYLE_ASSIGN_OR_RETURN(auto destination, pool.allocate(bytes));
    const auto offset = step % 3 == 0 ? std::size_t{4} : 0;
    const auto copied = bytes - 2 * offset;
    HYLE_ASSIGN_OR_RETURN(auto bounded, ready.subview(offset, copied));
    const auto before = metadata.calls.load(std::memory_order_relaxed);
    HYLE_RETURN_IF_ERROR(copy.bind_source({0}, bounded));
    HYLE_RETURN_IF_ERROR(copy.bind_mutable({1}, std::move(destination)));
    const std::array<c::CopyPhase, 1> phases{{{}}};
    const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, offset, copied}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases, ranges));
    HYLE_RETURN_IF_ERROR(require(metadata.calls.load(std::memory_order_relaxed) == before,
                                "readonly bind/prepare allocated HYLE metadata"));
    {
      HYLE_ASSIGN_OR_RETURN(auto submitted, copy.submit());
      HYLE_RETURN_IF_ERROR(require(!copy.unbind_mutable({1}), "retained submission released writer"));
      HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_ASSIGN_OR_RETURN(auto output, submitted.output({0}, {1}, offset, copied));
      std::array<std::uint16_t, 512> actual{};
      HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual)).first(copied)));
      for (std::size_t i = 0; i < copied / 2; ++i)
        HYLE_RETURN_IF_ERROR(require(actual[i] == expected[step % 4], "native copy bits differ"));
      HYLE_RETURN_IF_ERROR(require(submitted.report().logical_bytes == copied, "copy exceeded dirty extent"));
    }
    previous = std::move(ready);
    previous_bits = expected[step % 4];
  }
  for (auto &copy : slots) HYLE_RETURN_IF_ERROR(copy.reset());
  HYLE_RETURN_IF_ERROR(require(pool.stats().active_uses == 0 && pool.stats().live_blocks == 0,
                              "slots retained retired destination uses"));
  std::fprintf(stderr, "Apple ABS -> readonly export -> fixed F2 program/two slots PASS: "
      "64 episodes, F16 bits, old owners, partial views, capacity, foreign/spoof/writer rejection, "
      "no bind/prepare HYLE allocation; ready waits explicit\n");
  return Status::Ok();
}
} // namespace
int main() {
  const auto status = run();
  if (!status.ok()) std::fprintf(stderr, "native producer copy: %.*s\n",
                                int(status.message().size()), status.message().data());
  return status.ok() ? 0 : 1;
}
