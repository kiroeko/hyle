#pragma once

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>

#include "dspark_original_role_source.h"
#include "dspark_original_target_roles.h"
#include "dspark_target_geometry.h"
#include "dspark_pro_dense_specs.h"
#include "dspark_flash_draft_specs.h"

// Offline W04 draft. One named view of authenticated formal parameters. A
// composed Flash source borrows the existing target and vocabulary mappings;
// it never builds a concatenated bundle or copies their payloads on the host.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_formal_roles.h"
