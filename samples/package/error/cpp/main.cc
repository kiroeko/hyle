#include <string_view>
#include <utility>

#include <hyle/error/result.h>

int main() {
  hyle::Status status = hyle::Status::NotFound("model.bin");
  status = std::move(status).WithContext("loading model");
  hyle::Result<int> result = hyle::make_unexpected(std::move(status));

  return !result && result.error().IsNotFound() &&
                 hyle::status_code_from_win32(5) ==
                     hyle::StatusCode::PermissionDenied &&
                 result.error().message() ==
                     std::string_view("loading model: model.bin")
             ? 0
             : 1;
}
