#pragma once

#include <cstddef>

#include "hyle/compute/completion.h"
#include "hyle/compute/device.h"
#include "hyle/compute/visibility.h"
#include "hyle/error/result.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute {

enum class TransferMode : unsigned char {
  kImmediate = 0,
  kQueuedDirect,
};

// Exact route facts for the supplied Queue and resource pair. Unsupported
// routes return Unimplemented rather than pretending that hidden staging or a
// host wait is asynchronous.
struct TransferProperties final {
  TransferMode mode = TransferMode::kImmediate;
  bool peer = false;
};

HYLE_COMPUTE_API Result<TransferProperties>
transfer_properties(const Queue &queue, mem::BufferView destination,
                    mem::ConstBufferView source) noexcept;

// Record one byte-range operation and return the Completion for the complete
// ordered Queue prefix through it. Source and destination sizes must match;
// overlapping non-identical ranges are rejected before backend admission.
HYLE_COMPUTE_API Result<Completion>
copy_async(Queue &queue, mem::BufferView destination,
           mem::ConstBufferView source) noexcept;

HYLE_COMPUTE_API Result<Completion>
fill_async(Queue &queue, mem::BufferView destination,
           std::byte value) noexcept;

} // namespace hyle::compute
