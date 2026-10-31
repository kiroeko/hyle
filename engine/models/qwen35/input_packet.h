#pragma once

// One original Phase/slot owns the packet and its synchronous host staging.
// Port views remain exact ranges; no native work, Source or writer is moved here.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include "weights.h"

namespace qwen35 {
class InputPacket final {
public:
  struct MetadataDescriptions final {
    std::optional<hyle::compute::TensorDesc> last_ids, history_select, sampling_identities;
  };
  struct MetadataValues final {
    std::span<const std::int32_t> last_ids, history_select;
    std::span<const std::uint32_t> sampling_identities;
  };

  InputPacket() = default;
  InputPacket(const InputPacket &) = delete;
  InputPacket &operator=(const InputPacket &) = delete;
  InputPacket(InputPacket &&) = delete;
  InputPacket &operator=(InputPacket &&) = delete;
  bool prepared() const noexcept { return views_[kTokens].valid(); }
  const hyle::compute::ConstTensorView &tokens() const noexcept { return views_[kTokens]; }
  const hyle::compute::ConstTensorView &positions() const noexcept { return views_[kPositions]; }
  const hyle::compute::ConstTensorView &last_ids() const noexcept { return views_[kLastIds]; }
  const hyle::compute::ConstTensorView &history_select() const noexcept { return views_[kHistorySelect]; }
  const hyle::compute::ConstTensorView &sampling_identities() const noexcept { return views_[kSamplingIdentities]; }
  std::size_t bytes() const noexcept { return packet_bytes_; }
  bool allocation_refused() const noexcept { return allocation_refused_; }

  // Prepare before publishing the Phase or registering its original fixed IO.
  // Failed candidates release all payloads without replacing either owner.
  hyle::Status prepare(hyle::compute::TensorDesc token_description,
      hyle::compute::TensorDesc position_description,
      hyle::mem::MemoryResourceRef device,
      const hyle::compute::ResourceBudget &allocation_budget,
      const hyle::compute::ResourceBudget &execution_budget) {
    return prepare(token_description, position_description, device,
        allocation_budget, execution_budget, MetadataDescriptions{});
  }

  hyle::Status prepare(hyle::compute::TensorDesc token_description,
      hyle::compute::TensorDesc position_description,
      hyle::mem::MemoryResourceRef device,
      const hyle::compute::ResourceBudget &allocation_budget,
      const hyle::compute::ResourceBudget &execution_budget,
      const MetadataDescriptions &metadata) {
    allocation_refused_ = false;
    if (prepared()) return hyle::Status::FailedPrecondition("input packet is already prepared");
    if (!device) return hyle::Status::InvalidArgument("input packet requires an original device resource");
    const std::array<std::optional<hyle::compute::TensorDesc>, kPortCount> descriptions{
        token_description, position_description, metadata.last_ids,
        metadata.history_select, metadata.sampling_identities};
    std::array<std::size_t, kPortCount> port_bytes{}, port_offsets{};
    std::size_t packet_bytes = 0;
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
      if (!descriptions[i]) continue;
      const auto &port = *descriptions[i];
      const auto type = i == kSamplingIdentities ? hyle::compute::DataType::kUInt32
                                               : hyle::compute::DataType::kInt32;
      if (!port.is_plain() || !port.is_dense() || port.data_type() != type)
        return hyle::Status::InvalidArgument("input packet requires original plain dense I32/U32 ports");
      HYLE_ASSIGN_OR_RETURN(const auto bytes, port.required_bytes());
      if (!bytes || packet_bytes > SIZE_MAX - 255)
        return hyle::Status::OutOfRange("input packet byte extent is empty or overflows");
      const auto offset = (packet_bytes + 255) & ~std::size_t{255};
      if (bytes > SIZE_MAX - offset)
        return hyle::Status::OutOfRange("input packet byte extent overflows");
      port_bytes[i] = bytes;
      port_offsets[i] = offset;
      packet_bytes = offset + bytes;
    }
    if (packet_bytes > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()))
      return hyle::Status::OutOfRange("input packet shape exceeds i64");
    const auto description = dense(hyle::compute::DataType::kUInt8,
        {static_cast<std::int64_t>(packet_bytes)});
    Tensor payload, staging;
    try {
      payload = Tensor(device, description, true, allocation_budget);
      // The former host vectors also use system memory. Its actual map/flush
      // implementation adds no new Backend staging capability requirement.
      staging = Tensor(hyle::mem::system_resource(), description, true, execution_budget);
    } catch (const hyle::inference::EvaluationFailure &failure) {
      allocation_refused_ = failure.status().IsResourceExhausted() && !failure.status().has_native_error();
      return failure.status(); // Only cold allocation refusal permits the old owner path.
    }
    HYLE_ASSIGN_OR_RETURN(auto mapped, staging.storage().map_host(hyle::mem::HostAccess::kWrite));
    const auto writable = mapped.writable_bytes();
    auto status = writable ? hyle::Status::Ok() : static_cast<hyle::Status>(writable.error());
    status.Update(mapped.close());
    if (!status.ok()) return status;
    const auto root = payload.storage();
    std::array<hyle::compute::ConstTensorView, kPortCount> views;
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
      if (!descriptions[i]) continue;
      HYLE_ASSIGN_OR_RETURN(auto range, root.subview(port_offsets[i], port_bytes[i]));
      HYLE_ASSIGN_OR_RETURN(const auto view,
          hyle::compute::ConstTensorView::create(*descriptions[i], range));
      views[i] = view;
    }
    payload_ = std::move(payload); staging_ = std::move(staging);
    port_bytes_ = port_bytes; port_offsets_ = port_offsets;
    packet_bytes_ = packet_bytes; views_ = views;
    return hyle::Status::Ok();
  }

  // The caller already borrowed this exact original retired carrier. Source
  // retention and slot rotation remain with that caller, not these port views.
  hyle::Status write(std::span<const std::int32_t> token_values,
                     std::span<const std::int32_t> position_values) {
    return write(token_values, position_values, MetadataValues{});
  }

  hyle::Status write(std::span<const std::int32_t> token_values,
                     std::span<const std::int32_t> position_values,
                     const MetadataValues &metadata) {
    if (!prepared()) return hyle::Status::FailedPrecondition("input packet is not prepared");
    const std::array<std::size_t, kPortCount> sizes{token_values.size(), position_values.size(),
        metadata.last_ids.size(), metadata.history_select.size(), metadata.sampling_identities.size()};
    for (std::size_t i = 0; i < sizes.size(); ++i)
      if (sizes[i] != port_bytes_[i] / sizeof(std::uint32_t))
        return hyle::Status::InvalidArgument("input packet payload differs from its exact cold port extents");
    const std::array<std::span<const std::byte>, kPortCount> values{
        std::as_bytes(token_values), std::as_bytes(position_values), std::as_bytes(metadata.last_ids),
        std::as_bytes(metadata.history_select), std::as_bytes(metadata.sampling_identities)};
    HYLE_ASSIGN_OR_RETURN(auto mapped, staging_.storage().map_host(hyle::mem::HostAccess::kWrite));
    auto writable = mapped.writable_bytes();
    if (!writable) {
      auto status = static_cast<hyle::Status>(writable.error());
      status.Update(mapped.close()); return status;
    }
    std::fill(writable->begin(), writable->end(), std::byte{});
    for (std::size_t i = 0; i < values.size(); ++i)
      if (port_bytes_[i]) std::memcpy(writable->data() + port_offsets_[i], values[i].data(), port_bytes_[i]);
    auto status = mapped.flush();
    if (status.ok()) status = payload_.storage().write(mapped.bytes());
    status.Update(mapped.close());
    return status;
  }

private:
  enum Port : std::size_t { kTokens, kPositions, kLastIds, kHistorySelect, kSamplingIdentities, kPortCount };
  Tensor payload_, staging_;
  std::array<hyle::compute::ConstTensorView, kPortCount> views_;
  std::array<std::size_t, kPortCount> port_bytes_{}, port_offsets_{};
  std::size_t packet_bytes_ = 0;
  bool allocation_refused_ = false;
};
} // namespace qwen35
