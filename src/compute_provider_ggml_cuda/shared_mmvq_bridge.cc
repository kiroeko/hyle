#include "shared_mmvq_state.h"
extern "C" void *hyle_shared_mmvq_select(const void *context, const void *node, const void *input,
                                         unsigned stream, std::size_t bytes,
                                         bool *quantize) noexcept {
  if (!quantize) {
    if (hyle_shared_mmvq::active)
      hyle_shared_mmvq::active->poisoned = true;
    return nullptr;
  }
  try {
    const auto selection = hyle_shared_mmvq::select(context, node, input, stream, bytes);
    *quantize = selection.quantize;
    return selection.data;
  } catch (...) {
    if (hyle_shared_mmvq::active)
      hyle_shared_mmvq::active->poisoned = true;
    if (quantize)
      *quantize = false;
    return nullptr;
  }
}
extern "C" bool hyle_shared_mmvq_begin(void *value) noexcept {
  try {
    hyle_shared_mmvq::begin(static_cast<hyle_shared_mmvq::Plan *>(value));
    return true;
  } catch (...) {
    return false;
  }
}
extern "C" bool hyle_shared_mmvq_check(void *value) noexcept {
  auto *plan = static_cast<hyle_shared_mmvq::Plan *>(value);
  return !plan || (hyle_shared_mmvq::active == plan && !plan->poisoned);
}
extern "C" void hyle_shared_mmvq_end(void *value) noexcept {
  hyle_shared_mmvq::end(static_cast<hyle_shared_mmvq::Plan *>(value));
}
