#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

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
  PreparedReadback(PreparedReadback &&) noexcept = default;
  PreparedReadback &operator=(PreparedReadback &&other) noexcept {
    if (this != &other) {
      writer_.reset(); pool_ = {}; caller_ = {};
      facts = std::move(other.facts);
      caller_ = std::move(other.caller_); pool_ = std::move(other.pool_);
      writer_ = std::move(other.writer_);
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
      if (budget && bytes) {
        HYLE_ASSIGN_OR_RETURN(result.pool_, compute::LeasePool::create(resource, bytes,
            {.max_blocks = 1, .resource_budget = budget}));
        HYLE_ASSIGN_OR_RETURN(result.writer_, result.pool_.allocate(bytes));
      } else {
        HYLE_ASSIGN_OR_RETURN(result.caller_, mem::Buffer::create(resource, bytes, 256));
      }
    }
    return result;
  }

  Status bind(graph::Invocation &run) noexcept {
    if (!facts.caller_owned()) return Status::Ok();
    HYLE_ASSIGN_OR_RETURN(auto storage, caller_storage());
    HYLE_ASSIGN_OR_RETURN(auto tensor,
                         compute::TensorView::create(facts.description, storage));
    return run.bind(facts.value, tensor);
  }

  // The caller has already waited successfully and has exclusive access to
  // these results. A temporary native output/lease remains alive for the
  // complete synchronous read; it does not change the output's ownership.
  Status read(const Backend &backend, graph::Invocation &run,
              std::span<std::byte> bytes) const noexcept {
    if (facts.caller_owned()) {
      HYLE_ASSIGN_OR_RETURN(auto storage, caller_storage());
      return backend.read(storage, bytes);
    }
    HYLE_ASSIGN_OR_RETURN(auto output, run.result(facts.value));
    if (auto tensor = output.tensor(); tensor)
      return backend.read(tensor->storage(), bytes);
    HYLE_ASSIGN_OR_RETURN(auto owner, output.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto storage, owner.view());
    return backend.read(storage, bytes);
  }

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
  Result<mem::BufferView> caller_storage() noexcept {
    if (writer_) return writer_.view();
    return caller_.view();
  }
  Result<mem::ConstBufferView> caller_storage() const noexcept {
    if (writer_) {
      HYLE_ASSIGN_OR_RETURN(auto storage, writer_.view());
      return mem::ConstBufferView(storage);
    }
    return caller_.view();
  }
  mem::Buffer caller_;
  compute::LeasePool pool_;
  compute::WriteLease writer_;
};

} // namespace hyle::inference
