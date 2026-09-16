# R5 Windows selected validation

Baseline `3f6fe5d4cc6b0be33cbb6474d3c3534fa7524221`, immutable candidate
`6f3ff7cae99ff141fe5ccc44ff842c0fd60a1e3d`, tree
`56d4ac4db60f1575aef99939f951c5b2d7a565bd`: aggregate **PASS**.

- Fifteen product delivery gates passed actual source, installed static and installed
  shared consumers. Windows runtime DLLs were staged beside their executables.
- Ninety-three selected gates reused completed receipts with matching candidate and
  verified log hashes, including applicable native correctness, headers, configuration,
  component contracts and tool regression. Reuse is not another execution.
- Seven gates were explicitly skipped with the reasons in `identity.json`. In
  particular, Apple native correctness is not claimed on Windows; its real execution
  belongs to the Mac evidence.
- The resumed batch used Ninja, build parallelism two and normal process priority.
  The earlier interrupted MSBuild attempt is not counted as a passing gate. No CPU,
  CUDA or Apple P1 was rerun by this selector.

`windows-selected-raw.zip` contains the final result, bounded continuation helper/log,
all 108 PASS/REUSED receipts and their verified raw logs. Its SHA-256 is
`ed8023c373bdd09a797c1adf586c50fac010adafbc05b81c1034423dda588db9`.
`identity.json` preserves each artifact hash and each gate verdict/reason.

This completes the Windows selector obligation of the
[Apple delivery task](../../../docs/implementation/tasks/build/apple-r5-product-delivery.md),
[CUDA vendor audit](../../../docs/implementation/tasks/build/cuda-vendor-content-audit-v1.md)
and [native platform selector correction](../../../docs/implementation/tasks/build/validation-native-route-platforms-v1.md).
It does not close the Apple route's pending P1 or establish Vulkan product validation.
