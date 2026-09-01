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

  return ok.ok() && shared_copy.IsNotFound() &&
                 shared_copy.message() == std::string_view("model.bin") &&
                 !result && result.error().IsNotFound() &&
                 hyle::status_code_from_win32(5) ==
                     hyle::StatusCode::PermissionDenied &&
                 com.IsPermissionDenied() && com_copy == com &&
                 borrowed && &*borrowed == &registry_value &&
                 socket.IsUnavailable() &&
                 socket.native_error() ==
                     hyle::NativeError{hyle::NativeErrorDomain::WinSock,
                                       10061} &&
                 generic.IsPermissionDenied() &&
                 formatted.message() ==
                     std::string_view("tensor shard 7 is corrupt") &&
                 com.native_error() ==
                     hyle::NativeError{hyle::NativeErrorDomain::HResult,
                                       kAccessDenied} &&
                 result.error().message() ==
                     std::string_view("loading model: model.bin")
             ? 0
             : 1;
}
