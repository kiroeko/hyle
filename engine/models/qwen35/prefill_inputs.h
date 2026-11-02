#pragma once

#include "input_packet.h"

namespace qwen35 {
// Only the current admitted turn is visible here. Its immutable packets use
// the original storage domain, budget and native read-lease retirement.
class PrefillInputPackets final {
public:
  hyle::Result<bool> prepare(const InputPacket &packet,
      std::span<const std::int32_t> prompt, std::size_t offset,
      std::size_t position, std::size_t rows, std::size_t history_rows,
      std::size_t turn, hyle::mem::MemoryResourceRef device,
      const hyle::inference::Backend &backend,
      const hyle::compute::ResourceBudget &allocation_budget,
      const hyle::compute::ResourceBudget &execution_budget) {
    if (!packet.prepared() || !rows || offset >= prompt.size() ||
        packet.port_bytes_[InputPacket::kTokens] != rows * sizeof(std::int32_t) ||
        packet.port_bytes_[InputPacket::kPositions] != rows * sizeof(std::int32_t) ||
        packet.port_bytes_[InputPacket::kLastIds] != sizeof(std::int32_t) ||
        packet.port_bytes_[InputPacket::kHistorySelect] != history_rows * sizeof(std::int32_t) ||
        packet.port_bytes_[InputPacket::kSamplingIdentities]) return false;
    if (payload_.ready && prompt.data() == prompt_ && prompt.size() == prompt_size_ && turn == turn_ &&
        rows == rows_ && history_rows == history_rows_ && offset >= offset_ &&
        (offset - offset_) % rows == 0 && position == position_ + offset - offset_ &&
        packet.packet_bytes_ == packet_bytes_ && packet.port_offsets_ == port_offsets_)
      return true;
    // Single-chunk requests retain their original fixed IO and need no upload
    // allocation or dynamic binding. A partial final chunk can reuse a plan.
    if (prompt.size() - offset <= rows) return false;
    const auto count = 1 + (prompt.size() - offset - 1) / rows;
    if (packet.packet_bytes_ > SIZE_MAX - 255)
      return std::unexpected(hyle::Status::OutOfRange("prefill packet stride overflows"));
    const auto stride = (packet.packet_bytes_ + 255) & ~std::size_t{255};
    if (count > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()) / stride ||
        position > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()) ||
        prompt.size() - offset > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()) - position)
      return std::unexpected(hyle::Status::OutOfRange("prefill packet extent exceeds model coordinates"));
    try {
      const auto description = dense(hyle::compute::DataType::kUInt8,
          {static_cast<std::int64_t>(count * stride)});
      Tensor payload(device, description, true, allocation_budget);
      Tensor staging(hyle::mem::system_resource(), description, true, execution_budget);
      HYLE_ASSIGN_OR_RETURN(auto mapped, staging.storage().map_host(hyle::mem::HostAccess::kWrite));
      auto writable = mapped.writable_bytes();
      if (!writable) {
        auto status = static_cast<hyle::Status>(writable.error());
        status.Update(mapped.close()); return std::unexpected(std::move(status));
      }
      std::fill(writable->begin(), writable->end(), std::byte{});
      for (std::size_t part = 0; part < count; ++part) {
        const auto begin = offset + part * rows;
        const auto length = (std::min)(rows, prompt.size() - begin);
        auto *base = writable->data() + part * stride;
        std::memcpy(base + packet.port_offsets_[InputPacket::kTokens],
            prompt.data() + begin, length * sizeof(std::int32_t));
        auto *positions = reinterpret_cast<std::int32_t *>(base + packet.port_offsets_[InputPacket::kPositions]);
        for (std::size_t row = 0; row < length; ++row)
          positions[row] = static_cast<std::int32_t>(position + begin - offset + row);
        const auto last = static_cast<std::int32_t>(length - 1);
        std::memcpy(base + packet.port_offsets_[InputPacket::kLastIds], &last, sizeof(last));
        auto *history = reinterpret_cast<std::int32_t *>(base + packet.port_offsets_[InputPacket::kHistorySelect]);
        for (std::size_t row = 0; row < history_rows; ++row)
          history[row] = static_cast<std::int32_t>(length + row);
      }
      auto status = mapped.flush();
      if (status.ok()) status = payload.storage().write(mapped.bytes());
      status.Update(mapped.close());
      if (!status.ok()) return std::unexpected(std::move(status));
      payload.freeze(backend);
      payload_ = std::move(payload);
      prompt_ = prompt.data(); prompt_size_ = prompt.size(); turn_ = turn;
      offset_ = offset; position_ = position; rows_ = rows; history_rows_ = history_rows;
      stride_ = stride; packet_bytes_ = packet.packet_bytes_; port_offsets_ = packet.port_offsets_;
      return true;
    } catch (const hyle::inference::EvaluationFailure &failure) {
      if (failure.status().IsResourceExhausted() && !failure.status().has_native_error()) return false;
      return std::unexpected(failure.status());
    }
  }

  hyle::Result<hyle::compute::ReadLease> port(const InputPacket &packet,
      std::size_t input_offset, std::size_t port) const noexcept {
    if (!payload_.ready || port >= port_offsets_.size() || input_offset < offset_ ||
        (input_offset - offset_) % rows_ != 0)
      return std::unexpected(hyle::Status::FailedPrecondition("prefill packet is not prepared for this chunk"));
    return payload_.ready.subview((input_offset - offset_) / rows_ * stride_ + port_offsets_[port],
        packet.port_bytes_[port]);
  }

private:
  Tensor payload_;
  const std::int32_t *prompt_ = nullptr;
  std::size_t prompt_size_ = 0, turn_ = 0, offset_ = 0, position_ = 0;
  std::size_t rows_ = 0, history_rows_ = 0, stride_ = 0, packet_bytes_ = 0;
  std::array<std::size_t, InputPacket::kPortCount> port_offsets_{};
};
} // namespace qwen35
