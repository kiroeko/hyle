# CUDA C1 bounded entry diagnostic

One same-artifact native/native BF16 RMSNorm T32 H2048 diagnostic, two orders, cold/warm entry. This is not a Provider winner, product P1 or A/A qualification. Original full submit/terminal wait/reference settlement is charged.

The existing CUDA runtime consumer symmetrically warms2048 calls and reserves completion capacity. The old six-row adapter did two verify calls only. Four fixed diagnostic processes show that adding the established symmetric warmup is insufficient: warm reverse ratio0.926, warm forward1.004, and process costs11us versus30us. Neither entry proves stable qualification. The exact cause is not uniquely established; no repeated warmup/sampling search follows.

`entry-diagnostic.json` binds binary/CPU policy and each log hash. `entry-diagnostic.zip` preserves config/build/diagnostic logs. The old P1 FAIL remains in cuda-u4-provider-choices. Current original vendor math and fixed model coverage are unchanged.

A second source defect was concrete: old module validator returned FAIL but the public runner still entered its product collection, while the module executable did not check the ready marker. The adapter now uses the existing D5 ready-marker guard so the first product process refuses unqualified sampling. This does not change public statistics or margins. Guard/build verification is still pending.
