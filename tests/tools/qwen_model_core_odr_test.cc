// A second translation unit verifies that the internal model core can be
// included by multiple consumers without duplicate definitions.
#include "../../engine/models/qwen35/model_core.h"

const void *qwen_model_core_probe_names() noexcept {
  return qwen35::layer0_probe_names.data();
}
