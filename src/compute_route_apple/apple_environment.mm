#include "hyle/compute/apple.h"

#include <Availability.h>
#import <Foundation/Foundation.h>

#include <mlx/device.h>

#include "apple_build.h"

namespace hyle::compute::apple::detail {
Environment read_environment() {
  Environment result;
  const auto &info = mlx::core::device_info(mlx::core::Device::gpu);
  result.device_name = std::get<std::string>(info.at("device_name"));
  result.architecture = std::get<std::string>(info.at("architecture"));
  // Native Metal's driver is part of this OS runtime. Foundation provides the
  // actual version and build; do not substitute deployment or SDK macros.
  @autoreleasepool {
    result.os_runtime =
        [[NSProcessInfo processInfo].operatingSystemVersionString UTF8String];
  }
  result.mlx_version = HYLE_APPLE_MLX_VERSION;
  result.mlx_revision = HYLE_APPLE_MLX_REVISION;
  result.metal_compiler = HYLE_APPLE_METAL_COMPILER;
  result.build_sdk = __MAC_OS_X_VERSION_MAX_ALLOWED;
  result.deployment_target = __MAC_OS_X_VERSION_MIN_REQUIRED;
  return result;
}
} // namespace hyle::compute::apple::detail
