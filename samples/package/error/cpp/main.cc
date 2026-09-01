#include <cerrno>
#include <cstdint>
#include <string_view>
#include <system_error>
#include <utility>

#include <hyle/error/format.h>
#include <hyle/error/result.h>
#include <hyle/error/system_error.h>

int main() {
  hyle::Status ok;
  hyle::Status status = hyle::Status::NotFound("model.bin");
  hyle::Status shared_copy = status;
  status = std::move(status).WithContext("loading model");
  hyle::Result<int> result = hyle::make_unexpected(std::move(status));
  constexpr auto kAccessDenied = static_cast<std::int32_t>(0x80070005u);
  hyle::Status com =
      hyle::status_from_hresult(kAccessDenied, "naming thread");
  hyle::Status com_copy = com;
  int registry_value = 42;
  hyle::Result<int&> borrowed = registry_value;
  hyle::Status socket =
      hyle::status_from_winsock(10061, "connecting backend");
  hyle::Status generic = hyle::status_from_error_code(
      std::error_code(EACCES, std::generic_category()), "opening model");
  hyle::Status formatted = hyle::format_status(
      hyle::StatusCode::DataLoss, "tensor shard {} is corrupt", 7);

  if (!ok.ok()) return 1;
  if (!shared_copy.IsNotFound() || shared_copy.message() != "model.bin")
    return 2;
  if (result || !result.error().IsNotFound()) return 3;
  if (hyle::status_code_from_win32(5) !=
      hyle::StatusCode::PermissionDenied)
    return 4;
  if (!com.IsPermissionDenied() || com_copy != com) return 5;
  if (!borrowed || &*borrowed != &registry_value) return 6;
  if (!socket.IsUnavailable() ||
      socket.native_error() !=
          hyle::NativeError{hyle::NativeErrorDomain::WinSock, 10061})
    return 7;
  if (!generic.IsPermissionDenied()) return 8;
  if (formatted.message() != "tensor shard 7 is corrupt") return 9;
  if (com.native_error() !=
      hyle::NativeError{hyle::NativeErrorDomain::HResult, kAccessDenied})
    return 10;
  if (result.error().message() != "loading model: model.bin") return 11;
  return 0;
}
