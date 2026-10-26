#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "session_state.h"

namespace qwen35::cache {

enum class HostPayloadRole {
  kTargetKey, kTargetValue, kConvHistory, kGdnState,
  kCarryHidden, kPriorHidden, kDraftKey, kDraftValue
};

// Stable logical ordinals only. No physical page, native address, PageRef,
// VersionRef, mutable tail capability or complete model owner is stored.
struct HostPayloadDesc final {
  static constexpr auto no_layer = (std::numeric_limits<std::size_t>::max)();
  HostPayloadRole role = HostPayloadRole::kTargetKey;
  std::size_t layer = no_layer, page = 0;
  std::uint32_t initialized_rows = 0;
  std::size_t bytes = 0;
  friend bool operator==(const HostPayloadDesc &, const HostPayloadDesc &) = default;
};
struct HostAttentionBytes final { std::size_t layer = 0; KvRowBytes row_bytes; };
struct HostRecurrentBytes final { std::size_t layer = 0, history = 0, recurrent = 0; };
struct HostImageSchema final {
  ModelGeometry geometry;
  std::size_t capacity = 0;
  std::vector<HostAttentionBytes> attention;
  std::vector<HostRecurrentBytes> recurrent;
  KvRowBytes draft_row_bytes;
  std::size_t hidden_row_bytes = 0;
  bool draft_cache = false, frontier_hidden = false;
};
struct HostDraftBoundary final {
  std::size_t frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t catchup_token = -1;
  friend bool operator==(const HostDraftBoundary &, const HostDraftBoundary &) = default;
};
struct HostImageBoundary final {
  std::size_t length = 0;
  std::int32_t evaluated_next_id = -1, next_id = -1;
  SamplingCursor evaluated_sampling, sampling;
  bool sampling_wrapper = false, has_carry = false, has_prior = false;
  std::optional<HostDraftBoundary> draft;
};
struct HostImageIdentity final {
  // Non-owning process-local identities. The weak lifetime rejects dead or
  // address-reused Models without retaining their weights/cold inventory.
  const void *model = nullptr, *math = nullptr, *format = nullptr;
  const void *sampling_configuration = nullptr;
  std::size_t sampling_index = 0;
  std::weak_ptr<const void> model_lifetime;
  friend bool operator==(const HostImageIdentity &a, const HostImageIdentity &b) noexcept {
    return a.model == b.model && a.math == b.math && a.format == b.format &&
        a.sampling_configuration == b.sampling_configuration && a.sampling_index == b.sampling_index &&
        !a.model_lifetime.owner_before(b.model_lifetime) && !b.model_lifetime.owner_before(a.model_lifetime);
  }
};

inline hyle::Result<HostImageBoundary> host_image_boundary(const State &state) noexcept;
inline hyle::Result<std::vector<HostPayloadDesc>> make_host_image_layout(
    const HostImageSchema &, const HostImageBoundary &) noexcept;
inline hyle::Result<std::size_t> host_image_total_bytes(std::span<const HostPayloadDesc>) noexcept;
inline hyle::Status validate_host_image_source(const HostImageSchema &, const State &) noexcept;

class HostImage final {
public:
  HostImageIdentity identity;
  HostImageBoundary boundary;
  std::vector<HostPayloadDesc> layout;
  std::size_t total_bytes = 0;
  // These are the independent host endpoint/budget and this image's admission
  // reservation, never ColdResources. Ready leases die before their pool/owner.
  std::shared_ptr<const void> host_owner, admission;
  c::LeasePool pool;

  bool ready() const noexcept { return published_; }
  std::size_t payload_count() const noexcept { return ready_.size(); }
  const c::ReadLease &payload(std::size_t index) const { return ready_.at(index); }
  hyle::Status publish_payload(std::vector<c::ReadLease>) noexcept;
  hyle::Status validate_metadata(const HostImageSchema &, const HostImageIdentity &) const noexcept;
  hyle::Status validate(const HostImageSchema &, const HostImageIdentity &) const noexcept;
private:
  std::vector<c::ReadLease> ready_;
  bool published_ = false;
};

namespace host_image_detail {
inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > (std::numeric_limits<std::size_t>::max)() - a)
    throw std::overflow_error("host image extent overflows");
  return a + b;
}
inline std::size_t checked_multiply(std::size_t a, std::size_t b) {
  if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
    throw std::overflow_error("host image extent overflows");
  return a * b;
}
inline std::size_t page_count(std::size_t rows) noexcept {
  return rows / Plan::page_rows + (rows % Plan::page_rows != 0);
}
inline hyle::Status validate_identity(const HostImageIdentity &identity) noexcept {
  if (!identity.model || !identity.math || !identity.format || !identity.sampling_configuration ||
      identity.model_lifetime.expired())
    return hyle::Status::FailedPrecondition("host image requires its same live model and compiled identities");
  return {};
}
inline hyle::Status validate_schema(const HostImageSchema &schema) noexcept {
  try {
    schema.geometry.validate();
    if (!schema.capacity || schema.capacity > static_cast<std::size_t>(schema.geometry.context_limit) ||
        schema.attention.size() != schema.geometry.attention_layers() ||
        schema.recurrent.size() != schema.geometry.recurrent_layers())
      return hyle::Status::InvalidArgument("host image schema has invalid capacity or incomplete layer inventories");
    std::size_t attention = 0, recurrent = 0;
    for (std::size_t layer = 0; layer < schema.geometry.layers.size(); ++layer) {
      if (schema.geometry.is_gdn(layer)) {
        const auto &bytes = schema.recurrent[recurrent++];
        if (bytes.layer != layer || !bytes.history || !bytes.recurrent)
          return hyle::Status::InvalidArgument("host image recurrent schema lacks an exact real layer extent");
      } else {
        const auto &bytes = schema.attention[attention++];
        if (bytes.layer != layer || !bytes.row_bytes.key || !bytes.row_bytes.value)
          return hyle::Status::InvalidArgument("host image attention schema lacks exact K/V row extents");
      }
    }
    if ((schema.draft_cache && (schema.geometry.mtp_layers != 1 || !schema.draft_row_bytes.key || !schema.draft_row_bytes.value)) ||
        ((schema.draft_cache || schema.frontier_hidden) && !schema.hidden_row_bytes))
      return hyle::Status::InvalidArgument("host image schema lacks its draft or hidden carriers");
    return {};
  } catch (const std::exception &error) { return hyle::Status::InvalidArgument(error.what()); }
  catch (...) { return hyle::Status::InvalidArgument("host image schema validation failed"); }
}
inline hyle::Status validate_boundary(const HostImageSchema &schema, const HostImageBoundary &boundary) noexcept {
  const auto token = [&](std::int32_t id) { return id >= -1 && id < schema.geometry.vocabulary; };
  if (boundary.length > schema.capacity || !token(boundary.next_id) || !token(boundary.evaluated_next_id) ||
      (!boundary.length && (boundary.next_id != -1 || boundary.evaluated_next_id != -1)) ||
      (!boundary.sampling_wrapper && (boundary.next_id != boundary.evaluated_next_id ||
          boundary.sampling != boundary.evaluated_sampling)) ||
      boundary.draft.has_value() != schema.draft_cache)
    return hyle::Status::InvalidArgument("host image scalar boundary is inconsistent with its model schema");
  if ((boundary.has_prior && !boundary.has_carry) ||
      ((boundary.has_carry || boundary.has_prior) && !schema.hidden_row_bytes) ||
      (schema.frontier_hidden && boundary.length && !boundary.has_carry))
    return hyle::Status::InvalidArgument("host image is missing an authoritative hidden carrier");
  if (boundary.draft) {
    const auto &draft = *boundary.draft;
    if (!boundary.has_carry || draft.lag > 1 || boundary.length < draft.lag ||
        draft.frontier != boundary.length - draft.lag ||
        (draft.lag && (!boundary.has_prior || draft.catchup_token < 0 || !token(draft.catchup_token))) ||
        (!draft.lag && draft.catchup_token != -1))
      return hyle::Status::InvalidArgument("host image draft frontier, lag or catchup state is incomplete");
  }
  return {};
}
inline hyle::Status validate_payloads(std::span<const HostPayloadDesc> layout,
                                     std::span<const c::ReadLease> payloads) noexcept {
  if (layout.size() != payloads.size())
    return hyle::Status::InvalidArgument("host image payload count differs from its complete layout");
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    if (!payloads[i].valid() || payloads[i].size() != layout[i].bytes)
      return hyle::Status::InvalidArgument("host image payload does not cover its exact initialized bytes");
    const auto view = payloads[i].view();
    if (!view) return view.error();
    const auto properties = view->resource().properties();
    // Publication already requires original copy success and retirement.
    // Host access is through read()/map_host(), never unsynchronized access
    // inferred from a stronger concurrent-coherence property.
    if (!properties.has(hyle::mem::MemoryProperty::kHostAccessible))
      return hyle::Status::InvalidArgument("host image payload is not host-accessible storage");
  }
  return {};
}
} // namespace host_image_detail

inline hyle::Result<HostImageBoundary> host_image_boundary(const State &state) noexcept {
  const auto *evaluated = state.state_identity();
  if (!state.model || !state.math || !evaluated || evaluated->state_origin ||
      evaluated->model != state.model || evaluated->math != state.math || evaluated->length != state.length ||
      evaluated->keys.size() != state.keys.size() || evaluated->values.size() != state.values.size() ||
      evaluated->state.size() != state.state.size() || evaluated->draft.has_value() != state.draft.has_value() ||
      evaluated->carry_hidden.valid() != state.carry_hidden.valid() || evaluated->prior_hidden.valid() != state.prior_hidden.valid())
    return std::unexpected(hyle::Status::InvalidArgument("host image requires a flat canonical evaluated state origin"));
  if (state.draft && (state.draft->frontier != evaluated->draft->frontier || state.draft->lag != evaluated->draft->lag ||
      state.draft->catchup_token != evaluated->draft->catchup_token))
    return std::unexpected(hyle::Status::InvalidArgument("sampling wrapper changed its evaluated draft boundary"));
  HostImageBoundary result;
  result.length = state.length;
  result.evaluated_next_id = evaluated->next_id; result.evaluated_sampling = evaluated->sampling;
  result.next_id = state.next_id; result.sampling = state.sampling;
  result.sampling_wrapper = bool(state.state_origin);
  result.has_carry = evaluated->carry_hidden.valid(); result.has_prior = evaluated->prior_hidden.valid();
  if (evaluated->draft)
    result.draft = HostDraftBoundary{evaluated->draft->frontier, evaluated->draft->lag, evaluated->draft->catchup_token};
  return result;
}

inline hyle::Result<std::size_t> host_image_total_bytes(std::span<const HostPayloadDesc> layout) noexcept {
  try {
    std::size_t bytes = 0;
    for (const auto &entry : layout) {
      bool page = false, layer = false;
      switch (entry.role) {
      case HostPayloadRole::kTargetKey: case HostPayloadRole::kTargetValue: page = layer = true; break;
      case HostPayloadRole::kDraftKey: case HostPayloadRole::kDraftValue: page = true; break;
      case HostPayloadRole::kConvHistory: case HostPayloadRole::kGdnState: layer = true; break;
      case HostPayloadRole::kCarryHidden: case HostPayloadRole::kPriorHidden: break;
      default: return std::unexpected(hyle::Status::InvalidArgument("host image contains an unknown payload role"));
      }
      if (!entry.bytes || (layer == (entry.layer == HostPayloadDesc::no_layer)) ||
          (page && (!entry.initialized_rows || entry.initialized_rows > Plan::page_rows)) ||
          (!page && (entry.page || entry.initialized_rows)))
        return std::unexpected(hyle::Status::InvalidArgument("host image payload logical ordinal is malformed"));
      bytes = host_image_detail::checked_add(bytes, entry.bytes);
    }
    return bytes;
  } catch (const std::overflow_error &error) { return std::unexpected(hyle::Status::OutOfRange(error.what())); }
  catch (...) { return std::unexpected(hyle::Status::InvalidArgument("host image byte inventory is invalid")); }
}

inline hyle::Result<std::vector<HostPayloadDesc>> make_host_image_layout(
    const HostImageSchema &schema, const HostImageBoundary &boundary) noexcept {
  const auto schema_status = host_image_detail::validate_schema(schema);
  if (!schema_status.ok()) return std::unexpected(schema_status);
  const auto boundary_status = host_image_detail::validate_boundary(schema, boundary);
  if (!boundary_status.ok()) return std::unexpected(boundary_status);
  try {
    using host_image_detail::checked_add; using host_image_detail::checked_multiply;
    const auto target_pages = host_image_detail::page_count(boundary.length);
    const auto draft_pages = boundary.draft ? host_image_detail::page_count(boundary.draft->frontier) : 0;
    auto count = checked_multiply(checked_multiply(schema.attention.size(), target_pages), 2);
    count = checked_add(count, checked_multiply(schema.recurrent.size(), 2));
    count = checked_add(count, std::size_t(boundary.has_carry) + std::size_t(boundary.has_prior));
    count = checked_add(count, checked_multiply(draft_pages, 2));
    // Reject overflow before allocating descriptor metadata, including large
    // row carriers whose descriptor count itself would otherwise look small.
    std::size_t total = 0;
    for (const auto &layer : schema.attention) {
      total = checked_add(total, checked_multiply(boundary.length, layer.row_bytes.key));
      total = checked_add(total, checked_multiply(boundary.length, layer.row_bytes.value));
    }
    for (const auto &layer : schema.recurrent) total = checked_add(total, checked_add(layer.history, layer.recurrent));
    total = checked_add(total, checked_multiply(std::size_t(boundary.has_carry) + std::size_t(boundary.has_prior), schema.hidden_row_bytes));
    if (boundary.draft) {
      total = checked_add(total, checked_multiply(boundary.draft->frontier, schema.draft_row_bytes.key));
      total = checked_add(total, checked_multiply(boundary.draft->frontier, schema.draft_row_bytes.value));
    }
    std::vector<HostPayloadDesc> result;
    if (count > result.max_size()) return std::unexpected(hyle::Status::ResourceExhausted("host image descriptor capacity is exhausted"));
    result.reserve(count);
    const auto pages = [&](HostPayloadRole role, std::size_t layer, std::size_t rows, std::size_t row_bytes) {
      for (std::size_t page = 0; page < host_image_detail::page_count(rows); ++page) {
        const auto initialized = static_cast<std::uint32_t>((std::min)(std::size_t(Plan::page_rows), rows - page * Plan::page_rows));
        result.push_back({role, layer, page, initialized, checked_multiply(initialized, row_bytes)});
      }
    };
    for (const auto &layer : schema.attention) {
      pages(HostPayloadRole::kTargetKey, layer.layer, boundary.length, layer.row_bytes.key);
      pages(HostPayloadRole::kTargetValue, layer.layer, boundary.length, layer.row_bytes.value);
    }
    for (const auto &layer : schema.recurrent) {
      result.push_back({HostPayloadRole::kConvHistory, layer.layer, 0, 0, layer.history});
      result.push_back({HostPayloadRole::kGdnState, layer.layer, 0, 0, layer.recurrent});
    }
    if (boundary.has_carry) result.push_back({HostPayloadRole::kCarryHidden, HostPayloadDesc::no_layer, 0, 0, schema.hidden_row_bytes});
    if (boundary.has_prior) result.push_back({HostPayloadRole::kPriorHidden, HostPayloadDesc::no_layer, 0, 0, schema.hidden_row_bytes});
    if (boundary.draft) {
      pages(HostPayloadRole::kDraftKey, HostPayloadDesc::no_layer, boundary.draft->frontier, schema.draft_row_bytes.key);
      pages(HostPayloadRole::kDraftValue, HostPayloadDesc::no_layer, boundary.draft->frontier, schema.draft_row_bytes.value);
    }
    const auto verified = host_image_total_bytes(result);
    if (!verified || *verified != total || result.size() != count)
      return std::unexpected(hyle::Status::Internal("host image descriptor and byte inventories disagree"));
    return result;
  } catch (const std::overflow_error &error) { return std::unexpected(hyle::Status::OutOfRange(error.what())); }
  catch (const std::bad_alloc &) { return std::unexpected(hyle::Status::ResourceExhausted("host image descriptor allocation failed")); }
  catch (...) { return std::unexpected(hyle::Status::InvalidArgument("host image layout construction failed")); }
}

inline hyle::Status validate_host_image_source(const HostImageSchema &schema, const State &state) noexcept {
  const auto boundary = host_image_boundary(state);
  if (!boundary) return boundary.error();
  const auto layout = make_host_image_layout(schema, *boundary);
  if (!layout) return layout.error();
  const auto &evaluated = *state.state_identity();
  if (evaluated.keys.size() != schema.attention.size() || evaluated.values.size() != schema.attention.size() ||
      evaluated.state.size() != schema.geometry.layers.size() || evaluated.pages.size() != host_image_detail::page_count(evaluated.length))
    return hyle::Status::InvalidArgument("host image source has incomplete target inventories");
  const auto frame = [](const c::PagedArenaRead &read, std::span<const PageRef> pages, std::size_t rows) {
    if (!read.valid()) return false;
    const auto version = read.layout(); const auto facts = version.facts();
    if (facts.requests != 1 || facts.kv_rows != rows || facts.page_refs != pages.size()) return false;
    const auto host = version.host();
    if (host.kv_indptr.size() != 2 || host.kv_indptr[0] != 0 || std::size_t(host.kv_indptr[1]) != pages.size() ||
        host.page_indices.size() != pages.size() || host.last_page_lengths.size() != 1 ||
        host.last_page_lengths[0] != static_cast<std::int32_t>(rows ? (rows - 1) % Plan::page_rows + 1 : 0)) return false;
    for (std::size_t page = 0; page < pages.size(); ++page)
      if (host.page_indices[page] < 0 || static_cast<std::uint32_t>(host.page_indices[page]) != pages[page].page()) return false;
    return true;
  };
  for (std::size_t bank = 0; bank < schema.attention.size(); ++bank)
    if (!frame(evaluated.keys[bank], evaluated.pages, evaluated.length) ||
        !frame(evaluated.values[bank], evaluated.pages, evaluated.length))
      return hyle::Status::InvalidArgument("host image source K/V frames do not describe their exact logical pages");
  std::size_t recurrent = 0;
  for (std::size_t layer = 0; layer < schema.geometry.layers.size(); ++layer) {
    const auto &source = evaluated.state[layer];
    if (schema.geometry.is_gdn(layer)) {
      const auto &bytes = schema.recurrent[recurrent++];
      if (!source.history.valid() || !source.recurrent.valid() || source.history.size() != bytes.history || source.recurrent.size() != bytes.recurrent)
        return hyle::Status::InvalidArgument("host image source lacks complete conv/GDN layer payloads");
    } else if (source.history.valid() || source.recurrent.valid())
      return hyle::Status::InvalidArgument("host image source has unexpected recurrent state in an attention layer");
  }
  for (const auto *hidden : {&evaluated.carry_hidden, &evaluated.prior_hidden})
    if (hidden->valid() && hidden->size() != schema.hidden_row_bytes)
      return hyle::Status::InvalidArgument("host image source hidden carrier has the wrong extent");
  if (evaluated.draft && (evaluated.draft->pages.size() != host_image_detail::page_count(evaluated.draft->frontier) ||
      !frame(evaluated.draft->key, evaluated.draft->pages, evaluated.draft->frontier) ||
      !frame(evaluated.draft->value, evaluated.draft->pages, evaluated.draft->frontier)))
    return hyle::Status::InvalidArgument("host image source lacks its complete draft logical pages");
  return {};
}

inline hyle::Status HostImage::publish_payload(std::vector<c::ReadLease> ready) noexcept {
  if (published_) return hyle::Status::AlreadyExists("host image payload was already published");
  const auto bytes = host_image_total_bytes(layout);
  if (!bytes) return bytes.error();
  if (*bytes != total_bytes || (!layout.empty() && (!host_owner || !pool.valid())))
    return hyle::Status::InvalidArgument("host image host owner, pool or total byte extent is incomplete");
  const auto status = host_image_detail::validate_payloads(layout, ready);
  if (!status.ok()) return status;
  // The caller has waited for every original copy to retire before publishing
  // these ready leases. Merely submitting a copy cannot call this operation.
  ready_ = std::move(ready); published_ = true;
  return {};
}
inline hyle::Status HostImage::validate_metadata(const HostImageSchema &schema, const HostImageIdentity &expected) const noexcept {
  const auto expected_status = host_image_detail::validate_identity(expected);
  if (!expected_status.ok()) return expected_status;
  const auto status = host_image_detail::validate_identity(identity);
  if (!status.ok()) return status;
  if (!(identity == expected)) return hyle::Status::InvalidArgument("host image belongs to another live model, math, format or sampling configuration");
  const auto exact = make_host_image_layout(schema, boundary);
  if (!exact) return exact.error();
  if (*exact != layout) return hyle::Status::InvalidArgument("host image layout omits, duplicates or changes an exact logical payload");
  const auto bytes = host_image_total_bytes(layout);
  if (!bytes) return bytes.error();
  if (*bytes != total_bytes) return hyle::Status::InvalidArgument("host image total bytes differ from its complete inventory");
  return {};
}
inline hyle::Status HostImage::validate(const HostImageSchema &schema, const HostImageIdentity &expected) const noexcept {
  const auto metadata = validate_metadata(schema, expected);
  if (!metadata.ok()) return metadata;
  if (!published_) return hyle::Status::FailedPrecondition("host image payload is not retired and published");
  if (!layout.empty() && (!host_owner || !pool.valid())) return hyle::Status::FailedPrecondition("host image lost its independent host storage owner");
  return host_image_detail::validate_payloads(layout, ready_);
}

} // namespace qwen35::cache
