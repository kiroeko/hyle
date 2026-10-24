#include "../compute_batch/materialized_asymmetric_fixture.h"
#include "hyle/graph/cpu.h"

TEST(GraphBatchCpu, IndependentKeyAndValueDepthsSurviveReplay) {
  auto context = hyle::graph::cpu::create_context().value();
  materialized_asymmetric_test::exercise(context);
}
