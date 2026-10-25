#include "native_batch_regions.h"
#include "hyle/compute/detail/batch_provider.h"

#include <algorithm>
#include "hyle/compute/detail/paged_arena_access.h"

namespace hyle::graph::batch::detail {
namespace c = compute;
namespace gd = graph::detail;
namespace {
Status validate_facts(const NativeBatchRegionFacts &facts) noexcept {
  if (!facts.node.valid()) return Status::InvalidArgument("native batch node is empty");
  HYLE_RETURN_IF_ERROR(c::validate_batch_capacity(facts.capacity));
  HYLE_RETURN_IF_ERROR(c::validate_batch_attention_options(facts.causal, facts.window_left));
  if (facts.mla != c::MlaCacheLayout::kNone && facts.mla != c::MlaCacheLayout::kCkv512 &&
      facts.mla != c::MlaCacheLayout::kCkv512Rope64)
    return Status::InvalidArgument("native batch MLA interpretation is invalid");
  if (facts.output_query_rows > facts.capacity.max_query_rows)
    return Status::InvalidArgument("native batch output exceeds its admitted capacity");
  const bool paged = facts.capacity.storage == c::BatchStorage::kPaged;
  const auto count = facts.mla == c::MlaCacheLayout::kNone ? 3u : 2u;
  if (facts.paged_program) {
    const auto &program = *facts.paged_program;
    if (!paged || !program.valid() || !facts.layout.valid() || !program.schema().owns(facts.layout) ||
        program.schema().capacity() != facts.capacity || program.schema().mla_layout() != facts.mla ||
        program.causal() != facts.causal || program.window_left() != facts.window_left ||
        facts.metadata_input.valid() || facts.output_query_rows != facts.layout.facts().query_rows)
      return Status::FailedPrecondition("native page region has no exact owned recipe/layout");
  } else {
    if (!facts.metadata_input.valid() || facts.metadata_input.kind() != ValueKind::kInput)
      return Status::InvalidArgument("native capacity region needs its external metadata input");
    // Capacity descriptors do not own an initial schema/layout. Once prepared,
    // their original slot reports the actual owned version, not a host table
    // supplied by the caller or a replacement recipe identity.
    if (facts.layout.valid() && (!c::detail::BatchLayoutAccess::same_topology(facts.layout.origin_schema().capacity(), facts.capacity) ||
        facts.layout.origin_schema().mla_layout() != facts.mla ||
        facts.layout.facts().query_rows > facts.output_query_rows))
      return Status::InvalidArgument("native capacity prepared layout differs");
  }
  if (facts.inputs.size() != count || facts.input_ports.size() != count)
    return Status::InvalidArgument("native page region physical port map differs");
  for (std::size_t port = 0; port < count; ++port) {
    if (!facts.inputs[port].valid() ||
        (paged && port && facts.inputs[port].kind() != ValueKind::kInput))
      return Status::InvalidArgument("native page region needs exact external cache inputs");
    if (facts.metadata_input.valid() && facts.metadata_input == facts.inputs[port])
      return Status::InvalidArgument("native metadata input cannot alias a payload input");
    if (paged && port && facts.inputs[port] == facts.inputs[0])
      return Status::InvalidArgument("native page cache cannot share its query input");
    for (std::size_t other = 0; other < port; ++other)
      if (facts.input_ports[port] == facts.input_ports[other] &&
          facts.inputs[port] != facts.inputs[other])
        return Status::InvalidArgument("native page region physical input identities differ");
  }
  return Status::Ok();
}
Status validate_layout(const NativeBatchRegionFacts &facts,
    const c::BatchLayoutVersion &layout) noexcept {
  if (!layout.valid() || !c::detail::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(), facts.capacity) ||
      layout.origin_schema().mla_layout() != facts.mla ||
      layout.facts().query_rows > facts.output_query_rows)
    return Status::InvalidArgument("native batch layout capacity/output differs");
  if (facts.paged_program && (layout.facts().query_rows != facts.output_query_rows ||
      !facts.paged_program->schema().owns(layout)))
    return Status::InvalidArgument("native page layout schema/active output differs");
  return c::validate_batch_attention_layout(layout, facts.causal, facts.window_left);
}
} // namespace
Result<NativeBatchRegionFacts> inspect_native_batch_region(const NativeBatchRegionRef &region) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto facts, region.table->inspect(region.prepared, region.invocation));
  HYLE_RETURN_IF_ERROR(validate_facts(facts));
  if (facts.node != region.node || facts.paged_program != region.paged_program ||
      facts.metadata_input != region.metadata_input ||
      facts.capacity != region.capacity || facts.mla != region.mla ||
      facts.causal != region.causal || facts.window_left != region.window_left ||
      facts.output_query_rows != region.output_query_rows ||
      facts.inputs.size() != region.inputs.size() ||
      facts.input_ports.size() != region.input_ports.size() ||
      !std::equal(facts.inputs.begin(), facts.inputs.end(), region.inputs.begin()) ||
      !std::equal(facts.input_ports.begin(), facts.input_ports.end(), region.input_ports.begin()))
    return std::unexpected(Status::InvalidArgument("native page prepared port identity changed"));
  return facts;
}

Status collect_native_batch_regions(Invocation &invocation, NativeBatchRegions &regions) noexcept {
  regions.clear();
  return gd::RegionAccess::visit(invocation,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
        if (!view.prepared_vtable || !view.prepared_vtable->query_extension) return Status::Ok();
        const auto *table = static_cast<const NativeBatchRegionVTable *>(
            view.prepared_vtable->query_extension(view.prepared, kNativeBatchRegionExtensionId));
        if (!table) return Status::Ok();
        if (!table->inspect || !table->prepare || !table->reset || !view.invocation)
          return Status::FailedPrecondition("native batch selected capability is incomplete");
        HYLE_ASSIGN_OR_RETURN(auto facts, table->inspect(view.prepared, view.invocation));
        HYLE_RETURN_IF_ERROR(validate_facts(facts));
        if (facts.capacity.storage == c::BatchStorage::kPaged &&
            (!table->validate_frame || !table->commit_frame || !table->report ||
             !table->validate_written || !table->commit_written))
          return Status::FailedPrecondition("native page selected capability is incomplete");
        if (view.node_ordinals.size() != 1 || view.node_ordinals[0] != facts.node.ordinal())
          return Status::InvalidArgument("native page capability differs from its selected node");
        auto &regions = *static_cast<NativeBatchRegions *>(opaque);
        if (std::any_of(regions.begin(), regions.end(), [&](const auto &region) {
              return region.node == facts.node;
            }))
          return Status::InvalidArgument("native page node has duplicate selected slots");
        try {
          regions.push_back({table, view.prepared_vtable, view.prepared, view.invocation,
              view.step, view.lane, facts.node, facts.paged_program, facts.inputs,
              facts.input_ports, facts.metadata_input, facts.capacity, facts.mla,
              facts.causal, facts.window_left, facts.output_query_rows});
        } catch (...) {
          return Status::ResourceExhausted("native page original slot references");
        }
        return Status::Ok();
      }, &regions);
}

Status split_native_batch_layouts(std::span<const NativeBatchRegionRef> regions,
    std::span<const NodeLayout> layouts, NodeLayouts &shared) noexcept {
  shared.clear();
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    const auto &binding = layouts[i];
    if (!binding.node.valid() || !binding.layout.valid())
      return Status::InvalidArgument("native page node/layout is empty");
    for (std::size_t other = 0; other < i; ++other)
      if (layouts[other].node == binding.node)
        return Status::InvalidArgument("duplicate native Graph preparation node");
    const auto found = std::find_if(regions.begin(), regions.end(), [&](const auto &region) {
      return region.node == binding.node;
    });
    if (found == regions.end()) {
      try { shared.push_back(binding); }
      catch (...) { return Status::ResourceExhausted("native Graph cold layout partition"); }
      continue;
    }
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect_native_batch_region(*found));
    HYLE_RETURN_IF_ERROR(validate_layout(facts, binding.layout));
  }
  for (const auto &region : regions)
    if (std::none_of(layouts.begin(), layouts.end(), [&](const auto &binding) {
          return binding.node == region.node;
        }))
      return Status::InvalidArgument("native Graph requires every selected page node layout");
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto a = std::find_if(layouts.begin(), layouts.end(), [&](const auto &binding) {
      return binding.node == regions[i].node;
    });
    for (std::size_t j = 0; j < i; ++j) {
      const auto b = std::find_if(layouts.begin(), layouts.end(), [&](const auto &binding) {
        return binding.node == regions[j].node;
      });
      if (regions[i].metadata_input.valid() &&
          regions[i].metadata_input == regions[j].metadata_input && a->layout != b->layout)
        return Status::InvalidArgument("shared native metadata input needs one owned layout version");
      if (regions[i].capacity.storage != c::BatchStorage::kPaged ||
          regions[j].capacity.storage != c::BatchStorage::kPaged) continue;
      for (std::size_t p = 1; p < regions[i].inputs.size(); ++p)
        for (std::size_t q = 1; q < regions[j].inputs.size(); ++q)
          if (regions[i].inputs[p] == regions[j].inputs[q] && a->layout != b->layout)
            return Status::InvalidArgument("shared native page input needs one owned layout version");
    }
  }
  return Status::Ok();
}

Status prepare_native_batch_regions(std::span<const NativeBatchRegionRef> regions,
    std::span<const NodeLayout> layouts) noexcept {
  for (const auto &region : regions) {
    const auto found = std::find_if(layouts.begin(), layouts.end(), [&](const auto &binding) {
      return binding.node == region.node;
    });
    if (found == layouts.end()) return Status::InvalidArgument("native page layout is missing");
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect_native_batch_region(region));
    HYLE_RETURN_IF_ERROR(validate_layout(facts, found->layout));
  }
  for (const auto &region : regions) {
    const auto found = std::find_if(layouts.begin(), layouts.end(), [&](const auto &binding) {
      return binding.node == region.node;
    });
    HYLE_RETURN_IF_ERROR(region.table->prepare(region.prepared, region.invocation, found->layout));
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect_native_batch_region(region));
    if (facts.layout != found->layout)
      return Status::FailedPrecondition("native batch preparation did not retain its exact owned layout");
  }
  return Status::Ok();
}

Status reset_native_batch_regions(std::span<const NativeBatchRegionRef> regions) noexcept {
  for (const auto &region : regions)
    HYLE_RETURN_IF_ERROR(region.table->reset(region.prepared, region.invocation));
  return Status::Ok();
}
Result<std::uint64_t> native_paged_read_capacity(
    std::span<const NativeBatchRegionRef> regions) noexcept {
  std::uint64_t result = 0;
  for (const auto &region : regions) {
    if (region.capacity.storage != c::BatchStorage::kPaged) continue;
    const auto count = region.inputs.size() - 1;
    const auto refs = std::uint64_t(region.capacity.max_page_refs);
    if (refs && count > (UINT64_MAX - result) / refs)
      return std::unexpected(Status::ResourceExhausted("native page guard capacity overflow"));
    result += count * refs;
  }
  return result;
}

Status native_paged_frame(std::span<const NativeBatchRegionRef> regions, ValueId value,
    const c::PagedArenaRead &frame, ProjectedReaders &readers) noexcept {
  for (const auto &region : regions) {
    if (region.capacity.storage != c::BatchStorage::kPaged) continue;
    if (region.inputs[0] == value)
      return Status::InvalidArgument("native page frame cannot bind query input");
    bool used = false;
    for (std::size_t port = 1; port < region.inputs.size(); ++port)
      used |= region.inputs[port] == value;
    if (!used) continue;
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect_native_batch_region(region));
    if (!frame.valid() || frame.layout() != facts.layout)
      return Status::InvalidArgument("native page frame differs from its prepared layout");
    const auto capacity = facts.capacity;
    for (std::size_t port = 1; port < region.inputs.size(); ++port) {
      if (region.inputs[port] != value) continue;
      HYLE_RETURN_IF_ERROR(c::detail::PagedArenaAccess::validate_storage(frame,
          port == 1 ? capacity.key_storage : capacity.value_storage));
    }
    HYLE_RETURN_IF_ERROR(region.table->validate_frame(region.prepared, region.invocation, value, frame));
    try {
      for (std::size_t port = 1; port < region.inputs.size(); ++port) {
        if (region.inputs[port] != value) continue;
        const auto physical = region.input_ports[port];
        if (std::any_of(readers.begin(), readers.end(), [&](const auto &reader) {
              return reader.step == region.step && reader.input_port == physical;
            })) continue;
        readers.push_back({region.step, region.implementation, region.prepared, physical});
      }
    } catch (...) { return Status::ResourceExhausted("native page exact reader metadata"); }
  }
  return Status::Ok();
}
void commit_native_paged_frame(std::span<const NativeBatchRegionRef> regions, ValueId value,
    c::PagedArenaRead frame) noexcept {
  for (const auto &region : regions)
    if (region.capacity.storage == c::BatchStorage::kPaged &&
        std::any_of(region.inputs.begin() + 1, region.inputs.end(),
            [&](ValueId input) { return input == value; }))
      region.table->commit_frame(region.prepared, region.invocation, value, frame);
}
Result<c::NativePagedBatchReport> native_paged_report(Invocation &invocation, NodeId node) noexcept {
  struct Call { NodeId node; c::NativePagedBatchReport report{}; bool found = false; } call{node};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,
      [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
        if (!view.prepared_vtable || !view.prepared_vtable->query_extension) return Status::Ok();
        const auto *table = static_cast<const NativeBatchRegionVTable *>(
            view.prepared_vtable->query_extension(view.prepared, kNativeBatchRegionExtensionId));
        if (!table) return Status::Ok();
        if (!table->inspect)
          return Status::FailedPrecondition("native page selected report capability is incomplete");
        HYLE_ASSIGN_OR_RETURN(auto facts, table->inspect(view.prepared, view.invocation));
        auto &call = *static_cast<Call *>(opaque);
        if (facts.node != call.node) return Status::Ok();
        HYLE_RETURN_IF_ERROR(validate_facts(facts));
        if (facts.capacity.storage != c::BatchStorage::kPaged || !table->report)
          return Status::Unimplemented("selected native batch node has no page report");
        if (view.node_ordinals.size() != 1 || view.node_ordinals[0] != facts.node.ordinal())
          return Status::InvalidArgument("native page report differs from its selected node");
        if (call.found) return Status::InvalidArgument("duplicate selected native page report node");
        call.report = table->report(view.prepared, view.invocation);
        call.found = true;
        return Status::Ok();
      }, &call));
  if (!call.found)
    return std::unexpected(Status::InvalidArgument("node is not a selected native page region"));
  return call.report;
}
} // namespace hyle::graph::batch::detail
