/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

// Implementation support for public macros that need a unique local name.
// This header is installed because those macros expand in consumer TUs; its
// HYLE_DETAIL_* names are not product API.
#define HYLE_DETAIL_CONCAT_INNER(left, right) left##right
#define HYLE_DETAIL_CONCAT(left, right) \
  HYLE_DETAIL_CONCAT_INNER(left, right)
