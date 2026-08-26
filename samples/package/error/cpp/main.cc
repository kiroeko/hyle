#include <cstdint>
#include <string_view>
#include <utility>

#include <hyle/error/result.h>

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

  return ok.ok() && shared_copy.IsNotFound() &&
                 shared_copy.message() == std::string_view("model.bin") &&
                 !result && result.error().IsNotFound() &&
                 hyle::status_code_from_win32(5) ==
                     hyle::StatusCode::PermissionDenied &&
                 com.IsPermissionDenied() && com_copy == com &&
                 com.native_error() ==
                     hyle::NativeError{hyle::NativeErrorDomain::HResult,
                                       kAccessDenied} &&
                 result.error().message() ==
                     std::string_view("loading model: model.bin")
             ? 0
             : 1;
}
