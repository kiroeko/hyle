#pragma once

// One original Phase/slot owns the packet and its synchronous host staging.
// Port views remain exact ranges; no native work, Source or writer is moved here.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "weights.h"

namespace qwen35 {
class InputPacket final {
public:
  InputPacket() = default;
  InputPacket(const InputPacket &) = delete;
  InputPacket &operator=(const InputPacket &) = delete;
  InputPacket(InputPacket &&) = delete;
  InputPacket &operator=(InputPacket &&) = delete;
  bool prepared() const noexcept { return tokens_.valid(); }
  const hyle::compute::ConstTensorView &tokens() const noexcept { return tokens_; }
  const hyle::compute::ConstTensorView &positions() const noexcept { return positions_; }
  std::size_t bytes() const noexcept { return packet_bytes_; }
  bool allocation_refused() const noexcept { return allocation_refused_; }

  // Prepare before publishing the Phase or registering its original fixed IO.
  // Failed candidates release all payloads without replacing either owner.
  hyle::Status prepare(hyle::compute::TensorDesc token_description,
      hyle::compute::TensorDesc position_description,
      hyle::mem::MemoryResourceRef device,
      const hyle::compute::ResourceBudget &allocation_budget,
      const hyle::compute::ResourceBudget &execution_budget) {
    allocation_refused_ = false;
    if (prepared()) return hyle::Status::FailedPrecondition("input packet is already prepared");
    if (!device || !token_description.is_plain() || !token_description.is_dense() ||
        !position_description.is_plain() || !position_description.is_dense() ||
        token_description.data_type() != hyle::compute::DataType::kInt32 ||
        position_description.data_type() != hyle::compute::DataType::kInt32)
      return hyle::Status::InvalidArgument("input packet requires the original dense I32 ports");
    HYLE_ASSIGN_OR_RETURN(const auto token_bytes, token_description.required_bytes());
    HYLE_ASSIGN_OR_RETURN(const auto position_bytes, position_description.required_bytes());
    if (!token_bytes || !position_bytes || token_bytes > SIZE_MAX - 255)
      return hyle::Status::OutOfRange("input packet byte extent is empty or overflows");
    const auto position_offset = cache::add(token_bytes, 255) & ~std::size_t{255};
    if (position_bytes > SIZE_MAX - position_offset)
      return hyle::Status::OutOfRange("input packet byte extent overflows");
    const auto packet_bytes = cache::add(position_offset, position_bytes);
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
    HYLE_ASSIGN_OR_RETURN(auto token_range, root.subview(0, token_bytes));
    HYLE_ASSIGN_OR_RETURN(const auto token_view,
        hyle::compute::ConstTensorView::create(token_description, token_range));
    HYLE_ASSIGN_OR_RETURN(auto position_range, root.subview(position_offset, position_bytes));
    HYLE_ASSIGN_OR_RETURN(const auto position_view,
        hyle::compute::ConstTensorView::create(position_description, position_range));
    payload_ = std::move(payload); staging_ = std::move(staging);
    token_bytes_ = token_bytes; position_bytes_ = position_bytes;
    position_offset_ = position_offset; packet_bytes_ = packet_bytes;
    tokens_ = token_view; positions_ = position_view;
    return hyle::Status::Ok();
  }

  // The caller already borrowed this exact original retired carrier. Source
  // retention and slot rotation remain with that caller, not these port views.
  hyle::Status write(std::span<const std::int32_t> token_values,
                     std::span<const std::int32_t> position_values) {
    if (!prepared()) return hyle::Status::FailedPrecondition("input packet is not prepared");
    if (token_values.size_bytes() != token_bytes_ || position_values.size_bytes() != position_bytes_)
      return hyle::Status::InvalidArgument("input packet payload differs from its exact cold port extents");
    HYLE_ASSIGN_OR_RETURN(auto mapped, staging_.storage().map_host(hyle::mem::HostAccess::kWrite));
    auto writable = mapped.writable_bytes();
    if (!writable) {
      auto status = static_cast<hyle::Status>(writable.error());
      status.Update(mapped.close()); return status;
    }
    std::fill(writable->begin(), writable->end(), std::byte{});
    std::memcpy(writable->data(), token_values.data(), token_bytes_);
    std::memcpy(writable->data() + position_offset_, position_values.data(), position_bytes_);
    auto status = mapped.flush();
    if (status.ok()) status = payload_.storage().write(mapped.bytes());
    status.Update(mapped.close());
    return status;
  }

private:
  Tensor payload_, staging_;
  hyle::compute::ConstTensorView tokens_, positions_;
  std::size_t token_bytes_ = 0, position_offset_ = 0, position_bytes_ = 0, packet_bytes_ = 0;
  bool allocation_refused_ = false;
};
} // namespace qwen35
