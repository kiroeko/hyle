#pragma once

#include <string>

#include "hyle/error/status.h"

namespace hyle {

// C++ convenience formatting lives outside status.h so the hot carrier header
// does not pull in <string>. The result is created in the caller, not
// transferred across HYLE's shared-library ABI.
inline std::string to_string(const Status &status) {
  // Driver/OS diagnostics are normally short. Format those through one DLL
  // call into a stack buffer, then construct the returned string once. Only a
  // genuinely long message needs the size-query + second formatting call.
  char local[256];
  const std::size_t size = status.ToString(local, sizeof(local));
  if (size < sizeof(local)) return std::string(local, size);

  std::string text;
  text.resize_and_overwrite(size + 1, [&](char *data, std::size_t capacity) {
    status.ToString(data, capacity);
    return size;
  });
  return text;
}

}  // namespace hyle
