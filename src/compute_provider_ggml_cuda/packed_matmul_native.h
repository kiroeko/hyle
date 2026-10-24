#pragma once

#include <cstddef>
#include <cstdint>
#include <ggml.h>

struct ggml_backend;

// Private C++17 boundary implemented by the owning native MMQ TU. No
// allocation, data read, current-device mutation or public SDK ABI change.
namespace hyle::compute::ggml_cuda::internal {
enum class PackedMatmulRoute : std::uint8_t { kDeclined, kMmvq, kMmq };
struct PackedScratchPlan {
    PackedMatmulRoute route = PackedMatmulRoute::kDeclined;
    int device = -1;
    ggml_type type = GGML_TYPE_COUNT;
    std::int64_t width = 0, columns = 0, rows = 0;
    std::size_t activation_bytes = 0, scale_bytes = 0, fixup_bytes = 0;
    std::size_t bytes = 0;
};
inline bool operator==(const PackedScratchPlan & a, const PackedScratchPlan & b) {
    return a.route == b.route && a.device == b.device && a.type == b.type &&
        a.width == b.width && a.columns == b.columns && a.rows == b.rows &&
        a.activation_bytes == b.activation_bytes && a.scale_bytes == b.scale_bytes &&
        a.fixup_bytes == b.fixup_bytes && a.bytes == b.bytes;
}
inline bool operator!=(const PackedScratchPlan & a, const PackedScratchPlan & b) { return !(a == b); }
PackedScratchPlan packed_matmul_plan(int device, ggml_type type,
    std::int64_t width, std::int64_t columns, std::int64_t rows);
PackedMatmulRoute packed_matmul_route(int device, ggml_type type,
    std::int64_t width, std::int64_t columns, std::int64_t rows);
// Binds existing ordinary MMQ as well as the new large-bank route. M may
// exceed four here; only packed_matmul_route (the admission query) caps M4.
// Source, output and rank/stride signatures are checked, K must be padded-free.
PackedScratchPlan packed_mmq_node_plan(int device, const ggml_tensor * node);
// Exact cold mapping in this backend, plus stream0 fallback if concurrency
// metadata is later invalidated. No assumed device/stream-count table.
std::uint64_t packed_mmq_stream_mask(ggml_backend *, const ggml_tensor *);
struct PackedScratchBinding {
    const ggml_tensor * node = nullptr;
    PackedScratchPlan plan;
    std::uint64_t streams = 0;
};
struct PackedScratchArena {
    int stream = -1;
    void * data = nullptr;
    std::size_t bytes = 0;
};
// Metadata only. Each actually used stream borrows one maximum-op arena from
// caller workspace; same-stream FIFO permits reuse across its native nodes.
void * create_packed_scratch(ggml_backend *, const PackedScratchBinding *, std::size_t,
                            const PackedScratchArena *, std::size_t);
void destroy_packed_scratch(void *) noexcept;
void begin_packed_scratch(void *);
void check_packed_scratch(void *);
void end_packed_scratch(void *) noexcept;
class PackedScratchEncodeScope {
public:
    explicit PackedScratchEncodeScope(void * state) : state_(state) { begin_packed_scratch(state_); }
    ~PackedScratchEncodeScope() noexcept { end_packed_scratch(state_); }
    PackedScratchEncodeScope(const PackedScratchEncodeScope &) = delete;
    PackedScratchEncodeScope & operator=(const PackedScratchEncodeScope &) = delete;
    void check() { check_packed_scratch(state_); }
private:
    void * state_;
};
} // namespace hyle::compute::ggml_cuda::internal
