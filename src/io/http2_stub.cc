#include "hyle/c/io.h"

#include <cstddef>

#include "c_io_internal.h"

extern "C" {

int hyle_io_http2_available(void) {
  hyle::io::c_detail::clear_error();
  return 0;
}

hyle_io_http2_session* hyle_io_http2_session_create(
    hyle_allocator, hyle_io_http2_role) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return nullptr;
}

void hyle_io_http2_session_destroy(hyle_io_http2_session*) {}

int hyle_io_http2_session_valid(const hyle_io_http2_session*) { return 0; }

int hyle_io_http2_feed(hyle_io_http2_session*, const void*, size_t) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return 0;
}

size_t hyle_io_http2_next_output(hyle_io_http2_session*, const void** data) {
  if (data != nullptr) *data = nullptr;
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return SIZE_MAX;
}

int32_t hyle_io_http2_submit_request(
    hyle_io_http2_session*, const char*, size_t, const char*, size_t,
    const char*, size_t, const char*, size_t, const hyle_io_http2_header*,
    size_t, const void*, size_t) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return -1;
}

int hyle_io_http2_submit_response(
    hyle_io_http2_session*, int32_t, int, const hyle_io_http2_header*, size_t,
    const void*, size_t) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return 0;
}

int hyle_io_http2_reset(hyle_io_http2_session*, int32_t, uint32_t) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return 0;
}

int hyle_io_http2_goaway(hyle_io_http2_session*, uint32_t) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return 0;
}

int hyle_io_http2_next_message(hyle_io_http2_session*,
                               hyle_io_http2_message*) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return -1;
}

int hyle_io_http2_message_header(const hyle_io_http2_session*, size_t,
                                 hyle_io_http2_header*) {
  hyle::io::c_detail::set_error("HTTP/2 is not available in this HYLE build");
  return 0;
}

}  // extern "C"
