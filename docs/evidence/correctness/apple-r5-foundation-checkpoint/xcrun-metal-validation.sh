#!/bin/sh
# Validation environment only: Apple-signed standalone Metal component plus
# the installed Apple Command Line Tools SDK. All other xcrun calls are native.
if [ "$#" -ge 3 ] && [ "$1" = "-sdk" ] && [ "$2" = "macosx" ]; then
  case "$3" in
    metal|metallib)
      hyle_metal_tool="$3"
      shift 3
      SDKROOT=$(/usr/bin/xcrun --sdk macosx --show-sdk-path) || exit $?
      export SDKROOT
      exec /Users/kiro/hyle-verify-tools/metal-17F109/Metal.xctoolchain/usr/bin/"$hyle_metal_tool" "$@"
      ;;
  esac
fi
exec /usr/bin/xcrun "$@"
