#include "packed_test_support.h"

namespace {
auto prepare(const hyle::compute::apple::Context &context,
             hyle::compute::Operation operation,
             const std::vector<mlx::core::array> &inputs) {
  return context.prepare(operation, inputs);
}
TEST(AppleMlxQuantized, NativeFormatsUseOrdinaryEntryAndCheckedExperts) {
  apple_packed_test::formats(prepare);
}
TEST(AppleMlxQuantized, EmbeddingSelectsPackedRowsAndChecksChangedIds) {
  apple_packed_test::embedding(prepare);
}
} // namespace
