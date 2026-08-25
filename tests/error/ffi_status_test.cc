#include <atomic>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/c/error.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "../../src/error/status_access.h"

namespace {

TEST(ErrorFfi, NullIsCompleteOkRepresentation) {
  EXPECT_EQ(hyle_status_get_code(nullptr), HYLE_STATUS_OK);
  EXPECT_EQ(hyle_status_message(nullptr, nullptr, 0), 0u);
  hyle_status_destroy(nullptr);
}

TEST(ErrorFfi, CodeOnlyHandleIsAllocationFreeProcessToken) {
  hyle_status first = hyle::detail::status_to_ffi(
      hyle::Status::ResourceExhausted());
  hyle_status second = hyle::detail::status_to_ffi(
      hyle::Status::ResourceExhausted());
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(hyle_status_get_code(first), HYLE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(hyle_status_message(first, nullptr, 0), 0u);
  hyle_status_destroy(first);
  hyle_status_destroy(second);
}

TEST(ErrorFfi, OwnershipTransferLeavesNativeSourcesNonOk) {
  hyle::Status source = hyle::Status::Internal("transferred");
  hyle_status status =
      hyle::detail::status_to_ffi(std::move(source));
  EXPECT_TRUE(source.IsUnknown());
  EXPECT_EQ(hyle_status_get_code(status), HYLE_STATUS_INTERNAL);
  hyle_status_destroy(status);

  hyle::Result<int> result =
      hyle::make_unexpected(hyle::Status::NotFound("model"));
  hyle_status result_status = hyle::detail::status_to_ffi(
      std::move(result).error());
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsUnknown());
  EXPECT_EQ(hyle_status_get_code(result_status), HYLE_STATUS_NOT_FOUND);
  hyle_status_destroy(result_status);
}

TEST(ErrorFfi, CopiesEmbeddedNullMessageAndNativeDetail) {
  constexpr char kMessage[] = {'d', 'e', 'v', '\0', 'l', 'o', 's', 't'};
  hyle_status status = hyle::detail::status_to_ffi(hyle::Status::FromNative(
      hyle::StatusCode::Unavailable,
      {hyle::NativeErrorDomain::Vulkan, -4},
      std::string_view(kMessage, sizeof(kMessage))));
  ASSERT_NE(status, nullptr);

  EXPECT_EQ(hyle_status_get_code(status), HYLE_STATUS_UNAVAILABLE);
  hyle_native_error_domain domain = HYLE_NATIVE_ERROR_NONE;
  std::int64_t native_code = 0;
  EXPECT_EQ(hyle_status_get_native_error(status, &domain, &native_code), 1);
  EXPECT_EQ(domain, HYLE_NATIVE_ERROR_VULKAN);
  EXPECT_EQ(native_code, -4);

  EXPECT_EQ(hyle_status_message(status, nullptr, 0), sizeof(kMessage));
  char copied[sizeof(kMessage) + 1] = {};
  EXPECT_EQ(hyle_status_message(status, copied, sizeof(copied)),
            sizeof(kMessage));
  EXPECT_EQ(std::memcmp(copied, kMessage, sizeof(kMessage)), 0);
  EXPECT_EQ(copied[sizeof(kMessage)], '\0');

  char tiny[5] = {};
  const std::uint64_t required =
      hyle_status_to_string(status, tiny, sizeof(tiny));
  EXPECT_GT(required, sizeof(tiny) - 1);
  EXPECT_EQ(tiny[sizeof(tiny) - 1], '\0');
  hyle_status_destroy(status);
}

TEST(ErrorFfi, ImmutableQueriesMayRunConcurrentlyWhileOwnerIsPinned) {
  hyle_status status = hyle::detail::status_to_ffi(
      hyle::Status::FromNative(hyle::StatusCode::Unavailable,
                               {hyle::NativeErrorDomain::Vulkan, -4},
                               "device lost"));
  ASSERT_NE(status, nullptr);

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int worker = 0; worker < 8; ++worker) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < 2000; ++iteration) {
        hyle_native_error_domain domain = HYLE_NATIVE_ERROR_NONE;
        std::int64_t code = 0;
        char message[16] = {};
        if (hyle_status_get_code(status) != HYLE_STATUS_UNAVAILABLE ||
            hyle_status_get_native_error(status, &domain, &code) != 1 ||
            domain != HYLE_NATIVE_ERROR_VULKAN || code != -4 ||
            hyle_status_message(status, message, sizeof(message)) != 11 ||
            std::memcmp(message, "device lost", 11) != 0) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
  hyle_status_destroy(status);
}

} // namespace
