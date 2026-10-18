# F1 Graph aggregate retirement initial validation

Product candidate `7907afec8`; graph owner AUDITED under the F1 card. This is the finite aggregate
retirement sub-FIX, not the complete lease-aware Graph API or F1 terminal qualification.

## Source mechanism and verification

An aggregate previously attached children only during destroy, after poll/wait could have handed an
unknown child to native retirement. The fix arms every selected child before public observation,
uses nested CompletionRetention to preserve an inherited attachment, and retains the original
Invocation per child. The outer sentinel survives until outer lifetime handoff; destroy does not
poll and infer retirement from a cached error. First error is merged, not overwritten.

- Windows Graph core: **31 PASS**.
- Two independent, real cuBLAS F32 MatMul Graph branches: **1 PASS**. Both use original CUDA streams,
  events, allocations and retirement. Warm success completes, then real stream gates hold both branches.
- A test-only decoration returns an unknown observation error over each accepted native Completion.
  This is a deterministic **synthetic observation**, not a reproduced vendor/driver error or mock Provider.
  Actual CUDA retirement remains untouched and controls the last-use boundary.
- The outer error leaves Invocation in flight and all five admitted uses retained: four outer lease
  guards plus one pre-existing child read guard. Dropping Completion/Invocation/business readers does
  not reclaim the four blocks while native work is held.
- After native gates release and original drain retires both branches, both source blocks return;
  the two output handles remain backed. Publication is poisoned by the observation error, while an
  explicitly retained borrowed view verifies both real mathematical outputs against the identity reference.
- The extra child read attachment verifies nesting preservation, including when original child handles
  become cached errors. This adds no shipping test hook or product copy/math kernel.

Logs: [core](hyle_graph_test.log), [native](hyle_graph_native_retirement_test.log).

## Review, P0 and local adversarial pass

Each child independently retains the Invocation, including an internal unreturnable prefix with no
outer lifetime. The release callback saves that owner before releasing the aggregate; outer release
cannot invalidate the child while its own reference remains. No callback touches aggregate after the
last outer release. Inline completed children release safely behind the sentinel. Retention storage
clears before a successful run can reuse the preallocated child slots.

P0: O(selected terminal children), one preallocated nested node and one intrusive reference per child;
no new allocation, route search, command, native event, Queue, thread or reaper. Only Graph aggregate
control changes; the original one-terminal native Completion route and plain TensorView layout remain.
The earlier Direct receipt has measured metadata allocation counters. This Graph sub-FIX has source
cost review and real lifecycle counts, not a timed P1 qualification or performance winner.

Local adversarial pass covers cached unknown errors, abandoned handles, inherited attachments, first
error, lifetime of private Invocation state, and output poisoning versus physical completion. Full
package review/red-team, host-dependency protection before aggregate assembly, lease-aware Graph
results, other platforms/routes and installed static/shared are still pending and not marked PASS.

## Environment and accounting

Reused shared shipping producer `C:/h7/foundation-f1-q0`: developer factors
`graph,compute_backend_cuda,compute_provider_cublas`, MSVC 19.50.35728.0, CUDA 13.3.73, Ninja Release.
Build parallel 1, tests serial 1, no-window launches and EXE-local DLLs. Free RAM at build was 44 GiB.
No Torch/oneDNN/full operator suite, new cache checkout, remote build or timed performance run.
Selector selects Graph and its direct consumers; the broader route/three-delivery gates remain in the
concentrated final F1 validation. Closure/docs checks pass; these do not replace those gates.

[Unified LOC](loc-current.json) self-check PASS: Compute+Graph **69,063 code lines / 305 files**,
whole HYLE self-written non-test **164,290 / 656**. LOC began from a61805267 with the tracked product
diff that became 7907afec8; product contents did not change during the scan. Tests/samples/vendor/docs
are excluded. This count describes implementation, not self-written mathematical kernels or completed FIX count.
