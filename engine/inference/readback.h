#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "backend.h"
#include "prepared_output.h"

namespace hyle::inference {

// Prepare only the caller-owned backing. Native-owned results are obtained
// from the original Invocation after successful completion, never cached.
// The Backend resource owners must outlive this buffer and its native uses.
struct PreparedReadback final {
  PreparedOutputFacts facts;

  PreparedReadback() = default;
  PreparedReadback(const PreparedReadback &) = delete;
  PreparedReadback &operator=(const PreparedReadback &) = delete;
  PreparedReadback(PreparedReadback &&other) noexcept { *this = std::move(other); }
  PreparedReadback &operator=(PreparedReadback &&other) noexcept {
    if (this != &other) {
      writer_.reset(); ready_.reset(); pool_ = {}; caller_ = {};
      facts = std::move(other.facts);
      caller_ = std::move(other.caller_); pool_ = std::move(other.pool_);
      writer_ = std::move(other.writer_); ready_ = std::move(other.ready_);
      storage_ = std::exchange(other.storage_, {});
    }
    return *this;
  }

  static Result<PreparedReadback> prepare(const graph::CompiledGraph &compiled,
                                        graph::ValueId value,
                                        const Backend &backend,
                                        const compute::ResourceBudget &budget = {}) noexcept {
    PreparedReadback result;
    HYLE_ASSIGN_OR_RETURN(result.facts, prepare_output(compiled, value));
    HYLE_ASSIGN_OR_RETURN(auto resource, backend.resource_for(result.facts.storage));
    if (result.facts.caller_owned()) {
      HYLE_ASSIGN_OR_RETURN(auto bytes, result.facts.description.required_bytes());
      if (bytes) {
        HYLE_ASSIGN_OR_RETURN(result.pool_, compute::LeasePool::create(resource, bytes,
            {.max_blocks = 1, .resource_budget = budget}));
        HYLE_ASSIGN_OR_RETURN(result.writer_, result.pool_.allocate(bytes));
        HYLE_ASSIGN_OR_RETURN(result.storage_, result.writer_.view());
      } else {
        // A zero-byte output has no mutable block to transfer. Preserve its
        // original empty-view binding; it cannot become an OutputSource.
        HYLE_ASSIGN_OR_RETURN(result.caller_, mem::Buffer::create(resource, bytes, 256));
      }
    }
    return result;
  }

  // Bind once per Invocation. Later runs reuse the Graph-owned permission;
  // use reclaim before explicitly binding this backing into another run.
  Status bind(graph::Invocation &run) noexcept {
    if (!facts.caller_owned()) return Status::Ok();
    if (caller_) {
      HYLE_ASSIGN_OR_RETURN(auto tensor,
          compute::TensorView::create(facts.description, caller_.view()));
      return run.bind(facts.value, tensor);
    }
    if (!pool_.valid()) return Status::FailedPrecondition("prepared readback has no caller pool");
    HYLE_RETURN_IF_ERROR(run.prepare_resources());
    if (!writer_) {
      if (ready_) {
        HYLE_ASSIGN_OR_RETURN(writer_, ready_.try_write());
      } else {
        HYLE_ASSIGN_OR_RETURN(auto bytes, facts.description.required_bytes());
        HYLE_ASSIGN_OR_RETURN(writer_, pool_.allocate(bytes));
      }
    }
    // The Graph owns this exact writer after success. Its original submission
    // path promotes the published binding for subsequent runs at the same
    // address; this helper must not retain an extra reader between runs.
    return run.bind(facts.value, facts.description, std::move(writer_));
  }

  // The caller has already waited successfully. Read the original Graph's
  // published lease; temporary owners end with this synchronous read. When a
  // pending OutputSource has admitted readers, detach that view first so its
  // checked publication can precede the ordinary Graph result publication.
  Status read(const Backend &backend, graph::Invocation &run,
              std::span<std::byte> bytes) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto output, run.result(facts.value));
    if (facts.caller_owned()) {
      if (caller_) return backend.read(caller_.view(), bytes);
      HYLE_ASSIGN_OR_RETURN(auto owner, output.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto storage, owner.view());
      HYLE_RETURN_IF_ERROR(check_storage(storage));
      return backend.read(storage, bytes);
    }
    if (auto tensor = output.tensor(); tensor)
      return backend.read(tensor->storage(), bytes);
    HYLE_ASSIGN_OR_RETURN(auto owner, output.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto storage, owner.view());
    return backend.read(storage, bytes);
  }

  // Optional retired withdrawal before changing Invocation. Call before other
  // withdrawals revoke output observation. Escaped results/source pins reject
  // unbind through the original Graph gate. Escaped read leases reject writer
  // promotion; the actual ready owner remains here for a later bind/reclaim.
  // Ordinary repeated submit needs neither reclaim nor another bind.
  Status reclaim(graph::Invocation &run) noexcept {
    if (!facts.caller_owned() || writer_) return Status::Ok();
    if (caller_) return run.unbind(facts.value);
    if (!pool_.valid()) return Status::FailedPrecondition("prepared readback has no caller pool");
    if (!ready_) {
      HYLE_ASSIGN_OR_RETURN(auto output, run.result(facts.value));
      HYLE_ASSIGN_OR_RETURN(auto ready, output.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto storage, ready.view());
      HYLE_RETURN_IF_ERROR(check_storage(storage));
      output.reset();
      HYLE_RETURN_IF_ERROR(run.unbind(facts.value));
      ready_ = std::move(ready);
    }
    HYLE_ASSIGN_OR_RETURN(writer_, ready_.try_write());
    return Status::Ok();
  }

  // The full writer's original domain proof for OutputSource::create. The
  // Backend/compiled StorageDomain must retain the actual resource owner.
  const compute::LeasePool &pool() const noexcept { return pool_; }

  // Scratch belongs to the already prepared caller and is used only for F32
  // publication. Read exactly this dense port; never widen a native subrange
  // into neighboring bytes or flatten a padded/encoded description.
  Status predicates(const Backend &backend, graph::Invocation &run,
                    std::span<std::uint8_t> output,
                    std::span<float> scratch) const noexcept {
    HYLE_ASSIGN_OR_RETURN(auto bytes, facts.description.required_bytes());
    const auto type = facts.description.data_type();
    if (!facts.description.is_plain() || !facts.description.is_dense() ||
        (type != compute::DataType::kBool && type != compute::DataType::kFloat32))
      return Status::InvalidArgument("prepared predicate output is not a plain dense Bool or F32 array");
    HYLE_ASSIGN_OR_RETURN(auto elements, facts.description.element_count());
    const auto element_bytes = type == compute::DataType::kFloat32 ? sizeof(float) : sizeof(std::uint8_t);
    if (output.size() > (std::numeric_limits<std::size_t>::max)() / element_bytes)
      return Status::OutOfRange("prepared predicate byte count overflows size_t");
    if (elements != output.size() || bytes != output.size() * element_bytes)
      return Status::InvalidArgument("prepared predicate output count or byte extent differs");
    if (type == compute::DataType::kFloat32 && scratch.size() < output.size())
      return Status::InvalidArgument("prepared F32 predicate scratch is too small");
    if (output.empty()) return Status::Ok();
    if (type == compute::DataType::kBool) {
      HYLE_RETURN_IF_ERROR(read(backend, run, std::as_writable_bytes(output)));
      for (const auto value : output)
        if (value > 1)
          return Status::InvalidArgument("host predicate is not exact Bool zero or one");
    } else {
      const auto values = scratch.first(output.size());
      HYLE_RETURN_IF_ERROR(read(backend, run, std::as_writable_bytes(values)));
      for (std::size_t i = 0; i < output.size(); ++i) {
        const auto value = values[i];
        if (value != 0 && value != 1)
          return Status::InvalidArgument("host predicate is not exact F32 zero or one");
        output[i] = static_cast<std::uint8_t>(value == 1);
      }
    }
    return Status::Ok();
  }

  Result<std::uint8_t> predicate(const Backend &backend,
                                 graph::Invocation &run) const noexcept {
    std::uint8_t value = 0;
    float scratch = 0;
    HYLE_RETURN_IF_ERROR(predicates(backend, run, std::span(&value, 1), std::span(&scratch, 1)));
    return value;
  }

private:
  Status check_storage(mem::ConstBufferView storage) const noexcept {
    if (!storage_.valid() || storage.resource() != storage_.resource() ||
        storage.native_handle() != storage_.native_handle() ||
        storage.offset() != storage_.offset() || storage.size() != storage_.size())
      return Status::InvalidArgument("Graph readback differs from its original full caller storage");
    return Status::Ok();
  }
  // Only the legacy zero-byte binding uses a Buffer. Every nonempty caller
  // output transfers a full WriteLease, even without a ResourceBudget.
  mem::Buffer caller_;
  compute::LeasePool pool_;
  compute::WriteLease writer_;
  compute::ReadLease ready_;
  mem::ConstBufferView storage_;
};

} // namespace hyle::inference
