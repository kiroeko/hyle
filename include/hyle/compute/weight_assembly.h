#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/tensor.h"

namespace hyle::compute {

// One host fragment placed into a registered bank. id identifies the
// destination bank, not a global expert. Offset and source size are bytes.
// A dense plain bank requires its scalar storage type; an encoded bank takes
// dense plain U8 fragments of its already formatted physical bytes. No codec
// conversion, scale conversion or bit packing is implicit. Sources expose canonical host
// addresses (HostAccessible and HostAddressIdentity), so native adapters can
// reject staging/backing aliases before writing. No conversion or bit packing
// is implicit. Source readiness is the caller's responsibility.
//
// upload_slices validates every fragment and the complete coverage of each
// selected bank before staging or native submission. Fragments may arrive in
// any order, but must cover each bank exactly once, with no holes/overlaps.
// Bank alignment padding is private transfer storage, not published payload.
// All banks returned in one WeightBatch share its immutable upload owner.
struct WeightSlice final {
  std::uint64_t id = 0;
  ConstTensorView source{};
  std::size_t destination_offset = 0;
};

// Per-upload accounting, not measured bus/DRAM traffic. logical_bytes is the
// validated payload. staged_bytes counts only fully successful host reads;
// submitted_bytes is the accepted native copy extent, including bank padding.
// completed_bytes is that extent only after successful native retirement.
// Pending/failed work does not prove how many bytes the device actually moved.
// copied_bytes_unknown also records a failed host read with unknown partial
// effects. A zero completed_bytes with this flag never means zero traffic.
// poll/wait retain the original terminal status independently of this report.
struct WeightUploadReport final {
  std::size_t logical_bytes = 0;
  std::size_t staged_bytes = 0;
  std::size_t submitted_bytes = 0;
  std::size_t completed_bytes = 0;
  bool copied_bytes_unknown = false;
};

} // namespace hyle::compute
