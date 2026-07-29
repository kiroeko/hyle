/* Pure C consumer of include/hyle/c/error.h (close-01: no C++). */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hyle/c/error.h"

static int g_fails = 0;

#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, \
              __LINE__);                                             \
      ++g_fails;                                                     \
    }                                                                \
  } while (0)

static hyle_status helper_fail(void) { return hyle_status_not_found("gone"); }

static hyle_status helper_propagate(void) {
  HYLE_C_RETURN_IF_ERROR(helper_fail());
  return hyle_status_ok();
}

int main(void) {
  (void)"HYLE_C_ERROR_COVERAGE_BODY_MARKER";
  /* OK */
  hyle_status zero = HYLE_STATUS_INIT;
  CHECK(hyle_status_is_ok(zero));
  hyle_status_free(&zero);

  hyle_status ok = hyle_status_ok();
  CHECK(hyle_status_is_ok(ok));
  CHECK(hyle_status_get_code(ok) == HYLE_STATUS_OK);
  CHECK(!hyle_status_has_message(ok));
  CHECK(strcmp(hyle_status_message_cstr(ok), "") == 0); /* OK -> "" */
  {
    hyle_status generic =
        hyle_status_new(HYLE_STATUS_INVALID_ARGUMENT, "generic");
    CHECK(hyle_status_is_invalid_argument(generic));
    CHECK(strcmp(hyle_status_message_cstr(generic), "generic") == 0);
    hyle_status_free(&generic);
  }
  {
    const char bytes[] = {'a', '\0', 'b'};
    char copied[4] = {0};
    hyle_status binary =
        hyle_status_new_n(HYLE_STATUS_DATA_LOSS, bytes, sizeof(bytes));
    CHECK(hyle_status_message(binary, copied, sizeof(copied)) == sizeof(bytes));
    CHECK(memcmp(copied, bytes, sizeof(bytes)) == 0);
    CHECK(copied[3] == '\0');
    hyle_status_free(&binary);
  }
  {
    hyle_status invalid =
        hyle_status_new((hyle_status_code)999, "foreign code");
    CHECK(hyle_status_is_unknown(invalid));
    CHECK(strcmp(hyle_status_message_cstr(invalid), "foreign code") == 0);
    CHECK(strcmp(hyle_status_code_name((hyle_status_code)999), "Unknown") == 0);
    hyle_status_free(&invalid);
  }
  {
    char buf[32];
    size_t n = hyle_status_to_string(ok, buf, sizeof(buf));
    CHECK(n == 2);
    CHECK(strcmp(buf, "OK") == 0);
  }
  hyle_status_free(&ok);
  CHECK(hyle_status_is_ok(ok));

  /* canonical errno adapter */
  {
    CHECK(hyle_status_code_from_errno(0) == HYLE_STATUS_OK);
    CHECK(hyle_status_code_from_errno(EINVAL) ==
          HYLE_STATUS_INVALID_ARGUMENT);
    CHECK(hyle_status_code_from_errno(ENOENT) == HYLE_STATUS_NOT_FOUND);
    hyle_status native = hyle_status_from_errno(ENOENT, "opening model");
    hyle_native_error_domain domain = HYLE_NATIVE_ERROR_NONE;
    int64_t native_code = 0;
    CHECK(hyle_status_is_not_found(native));
    CHECK(strncmp(hyle_status_message_cstr(native), "opening model: ", 15) ==
          0);
    CHECK(hyle_status_get_native_error(native, &domain, &native_code));
    CHECK(domain == HYLE_NATIVE_ERROR_POSIX_ERRNO);
    CHECK(native_code == ENOENT);
    hyle_status_free(&native);
    native = hyle_status_from_errno_n(ENOENT, "open\0model", 10);
    CHECK(hyle_status_is_not_found(native));
    CHECK(hyle_status_message(native, NULL, 0) > 10);
    hyle_status_free(&native);
    native = hyle_status_from_errno(0, "ignored");
    CHECK(hyle_status_is_ok(native));
    hyle_status_free(&native);
  }

  {
    hyle_native_error_domain domain = HYLE_NATIVE_ERROR_NONE;
    int64_t native_code = 0;
    hyle_status native = hyle_status_new_native(
        HYLE_STATUS_UNAVAILABLE, HYLE_NATIVE_ERROR_VULKAN, -4, "device lost");
    CHECK(hyle_status_get_native_error(native, &domain, &native_code));
    CHECK(domain == HYLE_NATIVE_ERROR_VULKAN && native_code == -4);
    CHECK(strcmp(hyle_native_error_domain_name(domain), "Vulkan") == 0);
    hyle_status_add_context_n(&native, "submit\0kernel", 13);
    CHECK(hyle_status_get_native_error(native, &domain, &native_code));
    CHECK(domain == HYLE_NATIVE_ERROR_VULKAN && native_code == -4);
    hyle_status_free(&native);
    native = hyle_status_new_native_n(
        HYLE_STATUS_INTERNAL, (hyle_native_error_domain)1234, 77, NULL, 99);
    CHECK(hyle_status_get_native_error(native, &domain, &native_code));
    CHECK(domain == HYLE_NATIVE_ERROR_UNKNOWN && native_code == 77);
    hyle_status_free(&native);
    native = hyle_status_internal(NULL);
    CHECK(!hyle_status_get_native_error(native, &domain, &native_code));
    CHECK(domain == HYLE_NATIVE_ERROR_NONE && native_code == 0);
    hyle_status_free(&native);
  }

  /* convenience factory + full predicate set (sample) */
  hyle_status nf = hyle_status_not_found(NULL);
  CHECK(!hyle_status_is_ok(nf));
  CHECK(hyle_status_is_not_found(nf));
  CHECK(hyle_status_code_is(nf, HYLE_STATUS_NOT_FOUND));
  CHECK(strcmp(hyle_status_code_name(HYLE_STATUS_NOT_FOUND), "NotFound") == 0);
  CHECK(hyle_status_message_cstr(nf)[0] == '\0'); /* inline code, no message */

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
  hyle_status dst = HYLE_STATUS_INIT;
  hyle_status_move(&dst, &nf2);
  CHECK(hyle_status_is_ok(nf2));
  CHECK(hyle_status_is_not_found(dst));
  hyle_status_free(&dst);

  /* move edge cases: NULL src frees dst; self-move is a no-op short-circuit */
  {
    hyle_status m = hyle_status_internal("to-be-freed");
    CHECK(hyle_status_is_internal(m));
    hyle_status_move(&m, NULL); /* NULL src -> frees dst, leaves it OK */
    CHECK(hyle_status_is_ok(m));

    hyle_status self = hyle_status_aborted("keep");
    hyle_status_move(&self, &self); /* dst==src -> no-op, still valid */
    CHECK(hyle_status_is_aborted(self));
    hyle_status_free(&self);

    hyle_status_move(NULL, NULL); /* NULL dst -> safe no-op */
  }

  /* free is idempotent and NULL-safe */
  hyle_status_free(NULL); /* must not crash */
  {
    hyle_status f = hyle_status_data_loss("x");
    hyle_status_free(&f);
    CHECK(hyle_status_is_ok(f)); /* cleared to OK */
    hyle_status_free(&f);        /* second free is safe (idempotent) */
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
    const char *cm = hyle_status_message_cstr(boom); /* zero-copy borrow */
    CHECK(cm != NULL && strcmp(cm, "boom") == 0);
  }
  hyle_status_free(&boom);

  /* in-place context: one owning handle, outermost-first text */
  {
    hyle_status contextual = hyle_status_unavailable("device lost");
    hyle_status_add_context(&contextual, "vulkan vkQueueSubmit failed");
    CHECK(hyle_status_is_unavailable(contextual));
    CHECK(strcmp(hyle_status_message_cstr(contextual),
                 "vulkan vkQueueSubmit failed: device lost") == 0);
    hyle_status_add_context(&contextual, "launching kernel");
    CHECK(
        strcmp(hyle_status_message_cstr(contextual),
               "launching kernel: vulkan vkQueueSubmit failed: device lost") ==
        0);
    hyle_status_add_context(&contextual, NULL); /* no-op */
    hyle_status_add_context(NULL, "ignored");   /* no-op */
    hyle_status_free(&contextual);

    hyle_status code_only = hyle_status_not_found(NULL);
    hyle_status_add_context(&code_only, "loading shader");
    CHECK(strcmp(hyle_status_message_cstr(code_only), "loading shader") == 0);
    hyle_status_free(&code_only);

    hyle_status still_ok = hyle_status_ok();
    hyle_status_add_context(&still_ok, "ignored on OK");
    CHECK(hyle_status_is_ok(still_ok));
    hyle_status_free(&still_ok);
  }

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

  /* first-error aggregation consumes each candidate and preserves the first */
  {
    hyle_status overall = HYLE_STATUS_INIT;
    hyle_status first = hyle_status_not_found("first");
    hyle_status_update(&overall, &first);
    CHECK(hyle_status_is_not_found(overall));
    CHECK(hyle_status_is_ok(first));

    hyle_status second = hyle_status_internal("second");
    hyle_status_update(&overall, &second);
    CHECK(hyle_status_is_not_found(overall));
    CHECK(strcmp(hyle_status_message_cstr(overall), "first") == 0);
    CHECK(hyle_status_is_ok(second));

    HYLE_C_UPDATE_STATUS(overall, hyle_status_unavailable("third"));
    CHECK(hyle_status_is_not_found(overall));
    CHECK(strcmp(hyle_status_message_cstr(overall), "first") == 0);

    hyle_status_update(&overall, &overall); /* safe self no-op */
    CHECK(hyle_status_is_not_found(overall));
    hyle_status_update(NULL, &second);       /* safe NULL no-ops */
    hyle_status_update(&overall, NULL);
    hyle_status_free(&overall);
  }

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
