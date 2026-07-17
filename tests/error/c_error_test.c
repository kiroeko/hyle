/* Pure C consumer of include/hyle/c/error.h (close-01: no C++). */
#include "hyle/c/error.h"

#include <stdio.h>
#include <string.h>

static int g_fails = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__);\
      ++g_fails;                                                               \
    }                                                                          \
  } while (0)

static int g_handler_hits = 0;
static char g_last_msg[128];

static void on_report(void* user, hyle_severity sev, const char* file,
                      uint32_t line, const char* message) {
  (void)user;
  (void)sev;
  (void)file;
  (void)line;
  ++g_handler_hits;
  if (message != NULL) {
    size_t n = strlen(message);
    if (n >= sizeof(g_last_msg)) n = sizeof(g_last_msg) - 1;
    memcpy(g_last_msg, message, n);
    g_last_msg[n] = '\0';
  } else {
    g_last_msg[0] = '\0';
  }
}

int main(void) {
  CHECK(hyle_error_c_abi_version() == HYLE_ERROR_C_ABI_VERSION);

  /* API table */
  hyle_error_api api;
  CHECK(hyle_error_api_get(&api, HYLE_ERROR_C_ABI_VERSION) == 0);
  CHECK(api.abi_version == HYLE_ERROR_C_ABI_VERSION);
  CHECK(api.status_ok != NULL);
  CHECK(hyle_error_api_get(NULL, HYLE_ERROR_C_ABI_VERSION) == -1);
  CHECK(hyle_error_api_get(&api, 999) == -1);

  /* OK */
  hyle_status ok = hyle_status_ok();
  CHECK(hyle_status_is_ok(ok));
  CHECK(hyle_status_get_code(ok) == HYLE_STATUS_OK);
  CHECK(!hyle_status_has_message(ok));
  {
    char buf[32];
    size_t n = hyle_status_to_string(ok, buf, sizeof(buf));
    CHECK(n == 2);
    CHECK(strcmp(buf, "OK") == 0);
  }
  hyle_status_free(&ok);
  CHECK(hyle_status_is_ok(ok));

  /* convenience factory */
  hyle_status nf = hyle_status_not_found(NULL);
  CHECK(!hyle_status_is_ok(nf));
  CHECK(hyle_status_is_not_found(nf));
  CHECK(hyle_status_code_is(nf, HYLE_STATUS_NOT_FOUND));
  CHECK(strcmp(hyle_status_code_name(HYLE_STATUS_NOT_FOUND), "NotFound") == 0);

  /* clone + free original */
  hyle_status nf2 = hyle_status_clone(nf);
  CHECK(hyle_status_equals(nf, nf2));
  hyle_status_free(&nf);
  CHECK(hyle_status_get_code(nf2) == HYLE_STATUS_NOT_FOUND);

  /* move */
  hyle_status dst = hyle_status_ok();
  hyle_status_move(&dst, &nf2);
  CHECK(hyle_status_is_ok(nf2));
  CHECK(hyle_status_is_not_found(dst));
  hyle_status_free(&dst);

  /* with message + internal factory */
  hyle_status boom = hyle_status_internal("boom");
  CHECK(hyle_status_is_internal(boom));
  CHECK(hyle_status_has_message(boom));
  {
    char msg[32];
    size_t n = hyle_status_message(boom, msg, sizeof(msg));
    CHECK(n == 4);
    CHECK(strcmp(msg, "boom") == 0);
    char full[64];
    hyle_status_to_string(boom, full, sizeof(full));
    CHECK(strcmp(full, "Internal: boom") == 0);
  }
  hyle_status_free(&boom);

  /* equals */
  hyle_status a = hyle_status_aborted("x");
  hyle_status b = hyle_status_aborted("x");
  hyle_status c = hyle_status_aborted("y");
  CHECK(hyle_status_equals(a, b));
  CHECK(!hyle_status_equals(a, c));
  hyle_status_free(&a);
  hyle_status_free(&b);
  hyle_status_free(&c);

  /* diagnostic handler */
  g_handler_hits = 0;
  g_last_msg[0] = '\0';
  hyle_error_set_handler(on_report, NULL);
  hyle_error_report(HYLE_SEVERITY_ERROR, "c_error_test.c", 1, "hello-c");
  CHECK(g_handler_hits == 1);
  CHECK(strcmp(g_last_msg, "hello-c") == 0);
  hyle_error_set_handler(NULL, NULL);

  /* vtable path still works */
  hyle_status via = api.status_new(HYLE_STATUS_UNAVAILABLE, "down");
  CHECK(api.status_get_code(via) == HYLE_STATUS_UNAVAILABLE);
  api.status_free(&via);

  if (g_fails != 0) {
    fprintf(stderr, "c_error_test: %d check(s) failed\n", g_fails);
    return 1;
  }
  printf("c_error_test: all checks passed\n");
  return 0;
}
