#pragma once

namespace qwen35 {
// Model-specific entry; Compute/Graph remain independent of CLI/model policy.
int run(int argc, char **argv);
} // namespace qwen35
