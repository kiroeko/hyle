#pragma once
#include <array>
#include <functional>
#include <memory>
#include <unordered_map>
#include "dspark_original_role_source.h"
#include "glm_shared_vocabulary.h"
#include "d5_qwen38_24t_semantic_fragments.h"

// Original Q32 static bundle and immutable actual-ID expert attachments.
// This adapter reuses HYDSPT01 authentication/mappings. No network/execution,
// original-affine decode, BF16 narrowing or full expert domain allocation.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_mlx_original_roles.h"
