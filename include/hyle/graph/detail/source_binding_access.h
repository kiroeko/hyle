#pragma once

#include <utility>

#include "hyle/graph/source.h"

namespace hyle::graph::detail {
// Same-build typed-module bridge only. This copies borrowed descriptors;
// admission, retention and execution remain on the original Invocation.
struct SourceBindingAccess final {
  static SourceBindingData &get(SourceBinding &binding) noexcept {
    return binding.data_;
  }
  static const SourceBindingData &get(const SourceBinding &binding) noexcept {
    return binding.data_;
  }
  static SourceBinding make(SourceBindingData data) noexcept {
    return SourceBinding(std::move(data));
  }
};
} // namespace hyle::graph::detail
