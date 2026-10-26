#include "../../engine/inference/host_state_storage.h"
#include "../../engine/inference/state_transfer.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
namespace i = hyle::inference;
namespace c = hyle::compute;
using hyle::Status;

void check(Status status) {
  if (!status.ok()) {
    char text[1024];
    status.ToString(text, sizeof(text));
    throw std::runtime_error(text);
  }
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) check(result.error());
  return std::move(*result);
}
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

// All borrowed resources and pools outlive the original native CopyBatch.
// The escaped host lease deliberately retains no Backend or transfer scope.
struct TransferResources final {
  std::shared_ptr<i::Backend> backend;
  std::shared_ptr<i::HostStateStorage> storage;
  c::LeasePool source, host, restored;
};

void run(i::BackendKind kind) {
  constexpr std::size_t prefix = 64, extent = 128, limit = 1u << 20;
  std::array<std::byte, extent> expected{};
  expected.fill(std::byte{0xcd});
  for (std::size_t n = 0; n < prefix; ++n) expected[n] = std::byte((n * 37 + 11) & 255);
  std::shared_ptr<i::HostStateStorage> escaped_storage;
  c::ReadLease escaped;
  std::weak_ptr<i::Backend> old_backend;
  bool export_pending = false, import_pending = false;
  {
    auto resources = std::make_shared<TransferResources>();
    resources->backend = std::make_shared<i::Backend>(take(i::Backend::create({.primary = kind})));
    auto &backend = *resources->backend;
    old_backend = resources->backend;
    resources->storage = take(i::HostStateStorage::create(backend, 2, limit));
    check(resources->storage->validate_backend(backend));
    {
      auto foreign = take(i::Backend::create({.primary = kind}));
      require(resources->storage->validate_backend(foreign).IsInvalidArgument(),
          "independent Backends shared a host-storage identity");
    }
    const auto budget = resources->storage->budget();
    const auto allocator = take(budget.metadata_allocator());
    resources->source = take(c::LeasePool::create(backend.device(), 256, {.max_blocks = 1}));
    resources->restored = take(c::LeasePool::create(backend.device(), 256, {.max_blocks = 1}));
    resources->host = take(c::LeasePool::create(resources->storage->resource(), 256,
        {.max_blocks = 1, .allocator = allocator, .resource_budget = budget}));
    auto source_writer = take(resources->source.allocate(prefix));
    check(take(source_writer.view()).write(std::span(expected).first(prefix)));
    auto source = take(source_writer.publish());
    auto destination = take(resources->host.allocate(extent));
    check(take(destination.view()).fill(std::byte{0xcd}));
    const auto generation = destination.generation();
    std::array<i::StateTransferCopy, 1> copies{{{source, std::move(destination), prefix + 1}}};
    auto invalid = i::StateTransfer::prepare(backend, copies, resources, budget);
    require(!invalid && invalid.error().IsInvalidArgument() && copies[0].destination.valid() &&
        copies[0].destination.size() == extent && copies[0].destination.generation() == generation,
        "invalid initialized prefix consumed the original destination writer");
    copies[0].bytes = prefix;
    auto transfer = take(i::StateTransfer::prepare(backend, copies, resources, budget));
    require(!copies[0].destination.valid(), "successful preparation did not consume its destination");
    require(!transfer.take_writer(0), "destination escaped before successful native retirement");
    check(transfer.submit());
    export_pending = transfer.in_flight();
    require(transfer.submit().IsFailedPrecondition(), "single-use transfer accepted a second submit");
    check(transfer.wait());
    require(!transfer.in_flight(), "successful wait left original export work pending");
    auto host_writer = take(transfer.take_writer(0));
    require(host_writer.size() == extent && host_writer.generation() == generation,
        "export did not return its original complete writer");
    require(!transfer.take_writer(0), "export returned the same writer twice");
    auto host = take(host_writer.publish());
    std::array<std::byte, extent> actual{};
    check(take(host.view()).read(actual));
    require(actual == expected, "export changed its initialized suffix or copied incorrect bytes");

    auto restored_writer = take(resources->restored.allocate(extent));
    check(take(restored_writer.view()).fill(std::byte{0xef}));
    std::array<i::StateTransferCopy, 1> restore_copies{{{
        take(host.subview(0, prefix)), std::move(restored_writer), prefix}}};
    auto restore = take(i::StateTransfer::prepare(backend, restore_copies, resources, budget));
    check(restore.submit());
    import_pending = restore.in_flight();
    check(restore.wait());
    require(!restore.in_flight(), "successful wait left original import work pending");
    auto restored = take(restore.take_writer(0));
    require(restored.size() == extent, "import lost its original untouched writer suffix");
    check(take(restored.view()).read(actual));
    require(std::equal(actual.begin(), actual.begin() + prefix, expected.begin()) &&
        std::all_of(actual.begin() + prefix, actual.end(), [](auto byte) { return byte == std::byte{0xef}; }),
        "import copied incorrect bytes or changed its initialized suffix");
    check(take(restored.view()).fill(std::byte{0x5a}));
    check(take(host.view()).read(actual));
    require(actual == expected, "restored device mutation changed its retained host image");
    escaped_storage = resources->storage;
    escaped = std::move(host);
  }
  require(old_backend.expired(), "escaped host storage retained its execution Backend");
  std::array<std::byte, extent> after{};
  check(take(escaped.view()).read(after));
  require(after == expected, "host image failed after original Backend and pools were destroyed");
  escaped = {};
  require(take(escaped_storage->usage()).live_bytes <= limit, "host budget exceeded its requested-byte bound");
  std::cout << "state_transfer_pass prefix_bytes=" << prefix << " writer_bytes=" << extent
            << " export_pending_observed=" << export_pending << " import_pending_observed=" << import_pending
            << " backend_expired=1\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: inference_state_transfer_test --cpu|--cuda|--apple|--vulkan");
    const std::string_view selected(argv[1]);
    auto kind = i::BackendKind::kDefault;
    if (selected == "--cpu") kind = i::BackendKind::kCpu;
    else if (selected == "--cuda") kind = i::BackendKind::kCuda;
    else if (selected == "--apple") kind = i::BackendKind::kApple;
    else if (selected == "--vulkan") kind = i::BackendKind::kVulkan;
    else throw std::runtime_error("unknown transfer-test backend");
    const auto compiled = i::Backend::compiled_backends();
    if (std::find(compiled.begin(), compiled.end(), kind) == compiled.end()) {
      std::cerr << "SKIP: selected route is not built\n"; return 77;
    }
    run(kind);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
