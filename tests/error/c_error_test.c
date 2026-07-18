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
static hyle_severity g_last_severity = HYLE_SEVERITY_WARNING;

static void on_report(void* user, hyle_severity sev, const char* file,
                      uint32_t line, const char* message) {
  (void)user;
  g_last_severity = sev;
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

static int g_handler2_hits = 0;
static void on_report2(void* user, hyle_severity sev, const char* file,
                       uint32_t line, const char* message) {
  (void)user;
  (void)sev;
  (void)file;
  (void)line;
  (void)message;
  ++g_handler2_hits;
}

static int g_self_uninstall_hits = 0;
static void on_report_and_uninstall(void* user, hyle_severity sev,
                                    const char* file, uint32_t line,
                                    const char* message) {
  (void)user;
  (void)sev;
  (void)file;
  (void)line;
  (void)message;
  ++g_self_uninstall_hits;
  hyle_error_set_handler(NULL, NULL);
}

static hyle_status helper_fail(void) {
  return hyle_status_not_found("gone");
}

static hyle_status helper_propagate(void) {
  HYLE_C_RETURN_IF_ERROR(helper_fail());
  return hyle_status_ok();
}

int main(void) {
  CHECK(hyle_error_last_error() != NULL);

  /* OK */
  hyle_status ok = hyle_status_ok();
  CHECK(hyle_status_is_ok(ok));
  CHECK(hyle_status_get_code(ok) == HYLE_STATUS_OK);
  CHECK(!hyle_status_has_message(ok));
  CHECK(strcmp(hyle_status_message_cstr(ok), "") == 0);  /* OK -> "" */
  {
    hyle_status generic =
        hyle_status_new(HYLE_STATUS_INVALID_ARGUMENT, "generic");
    CHECK(hyle_status_is_invalid_argument(generic));
    CHECK(strcmp(hyle_status_message_cstr(generic), "generic") == 0);
    hyle_status_free(&generic);
  }
  {
    char buf[32];
    size_t n = hyle_status_to_string(ok, buf, sizeof(buf));
    CHECK(n == 2);
    CHECK(strcmp(buf, "OK") == 0);
  }
  hyle_status_free(&ok);
  CHECK(hyle_status_is_ok(ok));

  /* convenience factory + full predicate set (sample) */
  hyle_status nf = hyle_status_not_found(NULL);
  CHECK(!hyle_status_is_ok(nf));
  CHECK(hyle_status_is_not_found(nf));
  CHECK(hyle_status_code_is(nf, HYLE_STATUS_NOT_FOUND));
  CHECK(strcmp(hyle_status_code_name(HYLE_STATUS_NOT_FOUND), "NotFound") == 0);
  CHECK(hyle_status_message_cstr(nf)[0] == '\0');  /* inline code, no message */

  hyle_status ca = hyle_status_cancelled(NULL);
  CHECK(hyle_status_is_cancelled(ca));
  hyle_status_free(&ca);
  hyle_status un = hyle_status_unknown(NULL);
  CHECK(hyle_status_is_unknown(un));
  hyle_status_free(&un);
  hyle_status ia = hyle_status_invalid_argument(NULL);
  CHECK(hyle_status_is_invalid_argument(ia));
  hyle_status_free(&ia);
  hyle_status de = hyle_status_deadline_exceeded(NULL);
  CHECK(hyle_status_is_deadline_exceeded(de));
  hyle_status_free(&de);
  hyle_status ae = hyle_status_already_exists(NULL);
  CHECK(hyle_status_is_already_exists(ae));
  hyle_status_free(&ae);
  hyle_status pd = hyle_status_permission_denied(NULL);
  CHECK(hyle_status_is_permission_denied(pd));
  hyle_status_free(&pd);
  hyle_status re = hyle_status_resource_exhausted(NULL);
  CHECK(hyle_status_is_resource_exhausted(re));
  hyle_status_free(&re);
  hyle_status fp = hyle_status_failed_precondition(NULL);
  CHECK(hyle_status_is_failed_precondition(fp));
  hyle_status_free(&fp);
  hyle_status oor = hyle_status_out_of_range(NULL);
  CHECK(hyle_status_is_out_of_range(oor));
  hyle_status_free(&oor);
  hyle_status ui = hyle_status_unimplemented(NULL);
  CHECK(hyle_status_is_unimplemented(ui));
  hyle_status_free(&ui);
  hyle_status ua = hyle_status_unavailable(NULL);
  CHECK(hyle_status_is_unavailable(ua));
  hyle_status_free(&ua);
  hyle_status dl = hyle_status_data_loss(NULL);
  CHECK(hyle_status_is_data_loss(dl));
  hyle_status_free(&dl);
  hyle_status au = hyle_status_unauthenticated(NULL);
  CHECK(hyle_status_is_unauthenticated(au));
  CHECK(strcmp(hyle_status_code_name(HYLE_STATUS_UNAUTHENTICATED),
               "Unauthenticated") == 0);
  hyle_status_free(&au);

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

  /* move edge cases: NULL src frees dst; self-move is a no-op short-circuit */
  {
    hyle_status m = hyle_status_internal("to-be-freed");
    CHECK(hyle_status_is_internal(m));
    hyle_status_move(&m, NULL);           /* NULL src -> frees dst, leaves it OK */
    CHECK(hyle_status_is_ok(m));

    hyle_status self = hyle_status_aborted("keep");
    hyle_status_move(&self, &self);       /* dst==src -> no-op, still valid */
    CHECK(hyle_status_is_aborted(self));
    hyle_status_free(&self);

    hyle_status_move(NULL, NULL);         /* NULL dst -> safe no-op */
  }

  /* free is idempotent and NULL-safe */
  hyle_status_free(NULL);                 /* must not crash */
  {
    hyle_status f = hyle_status_data_loss("x");
    hyle_status_free(&f);
    CHECK(hyle_status_is_ok(f));           /* cleared to OK */
    hyle_status_free(&f);                  /* second free is safe (idempotent) */
    CHECK(hyle_status_is_ok(f));
  }

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
    const char* cm = hyle_status_message_cstr(boom);  /* zero-copy borrow */
    CHECK(cm != NULL && strcmp(cm, "boom") == 0);
  }
  hyle_status_free(&boom);

  /* equals */
  hyle_status a = hyle_status_aborted("x");
  hyle_status b = hyle_status_aborted("x");
  hyle_status c = hyle_status_aborted("y");
  CHECK(hyle_status_equals(a, b));
  CHECK(!hyle_status_equals(a, c));
  CHECK(hyle_status_is_aborted(a));
  hyle_status_free(&a);
  hyle_status_free(&b);
  hyle_status_free(&c);

  /* diagnostic handler + reportf + macros */
  g_handler_hits = 0;
  g_last_msg[0] = '\0';
  hyle_error_set_handler(on_report, NULL);
  hyle_error_report(HYLE_SEVERITY_ERROR, "c_error_test.c", 1, "hello-c");
  CHECK(g_handler_hits == 1);
  CHECK(strcmp(g_last_msg, "hello-c") == 0);
  hyle_error_reportf(HYLE_SEVERITY_WARNING, "c_error_test.c", 2, "n=%d", 7);
  CHECK(g_handler_hits == 2);
  CHECK(strcmp(g_last_msg, "n=7") == 0);
  CHECK(g_last_severity == HYLE_SEVERITY_WARNING);
  hyle_error_report((hyle_severity)99, NULL, 0, "invalid-severity");
  CHECK(g_handler_hits == 3);
  CHECK(g_last_severity == HYLE_SEVERITY_ERROR);
  HYLE_C_ERROR("macro-lit");
  CHECK(g_handler_hits == 4);
  CHECK(strcmp(g_last_msg, "macro-lit") == 0);
  HYLE_C_WARNF("w=%s", "ok");
  CHECK(g_handler_hits == 5);
  CHECK(strcmp(g_last_msg, "w=ok") == 0);

  /* v3: nesting-safe push/pop. Outer handler = on_report (counts g_handler_hits);
   * push a second handler, verify reports route to it, pop, verify the OUTER one
   * is restored exactly (LIFO). */
  {
    int outer_before = g_handler_hits;
    hyle_error_saved_handler_o* tok =
        hyle_error_push_handler(on_report2, NULL);
    CHECK(tok != NULL);
    g_handler2_hits = 0;
    hyle_error_report(HYLE_SEVERITY_ERROR, "c_error_test.c", 3, "inner");
    CHECK(g_handler2_hits == 1);            /* inner handler saw it */
    CHECK(g_handler_hits == outer_before);  /* outer did NOT */
    hyle_error_pop_handler(tok);            /* restore outer */
    hyle_error_report(HYLE_SEVERITY_ERROR, "c_error_test.c", 4, "back");
    CHECK(g_handler_hits == outer_before + 1);  /* outer restored */
    CHECK(strcmp(g_last_msg, "back") == 0);
    hyle_error_pop_handler(NULL);           /* NULL is a safe no-op */
  }

  hyle_error_set_handler(NULL, NULL);

  /* Handler replacement is reentrant: a callback may uninstall itself. */
  g_self_uninstall_hits = 0;
  hyle_error_set_handler(on_report_and_uninstall, NULL);
  hyle_error_report(HYLE_SEVERITY_ERROR, "c_error_test.c", 5,
                    "self-uninstall");
  CHECK(g_self_uninstall_hits == 1);
  hyle_error_set_handler(NULL, NULL);  /* already uninstalled: safe no-op */

  /* HYLE_C_RETURN_IF_ERROR */
  {
    hyle_status prop = helper_propagate();
    CHECK(hyle_status_is_not_found(prop));
    hyle_status_free(&prop);
  }

  if (g_fails != 0) {
    fprintf(stderr, "c_error_test: %d check(s) failed\n", g_fails);
    return 1;
  }
  printf("c_error_test: all checks passed\n");
  return 0;
}
