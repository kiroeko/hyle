# F2 Windows 验证副本：MSVC/Ninja 头依赖缺失

- scope：当前 F2 实现期 Windows 临时构建 `C:/h7/foundation-f1-q0`，不修改公共构建/selector/closure工具。
- 具体事实：`1c1ce5f26` 新 `CopyBatchReport` 布局进入公共header后，Ninja只重编新增/改动 `.cc`；
  旧 `copy_ranges.cc.obj` 和其测试object仍为09:49 UTC，而header为10:20 UTC。
  `ninja -t deps src/compute/CMakeFiles/hyle_compute.dir/copy_ranges.cc.obj` 显示 `#deps 0`。
  `rules.ninja` 的 `msvc_deps_prefix` 是英文 `Note: including file:`，runner实际cl输出中文包含行，
  因此该验证副本没有取得头依赖。公共对象首轮range/空批异常是混合新旧ABI，不能当产品通过或产品缺陷证据。
- 当前 F2 首轮 raw logs、两项失败及同binary单例诊断均保留；不复用该次public/common PASS代签新header产品。
- 有限恢复：只在本批scratch runner设置 `VSLANG=1033`，按两个具名test target清理其临时artifact并
  build1重建；不删产品源/依赖源码、不改全局locale、不改CMake工具或旧证据。核查对应object取得
  canonical `batch_transfer.h` 依赖，然后复跑原精准common/public CUDA门。
- 独立fixture纠正：关闭后的LeasePool不能再取得writer；早完成用例改为验证actual metadata归零、
  只读数据仍正确及旧reader最后释放回收。该改动没有放宽关闭或提前释放语义。
- 当前：`4d937b2f8` 同临时树按上述精确清理重建87项完成；原object现在 `#deps 22`，包含canonical
  `batch_transfer.h`。原精准34项common、5项native/public CUDA及3头全部PASS，恢复已验证。
  失败预算没有重置；两次原有限纠正均已使用。后续本批Windows runner沿用匹配locale，不回写公共工具。
  原失败、单例、旧mtime/hash/零deps以及新依赖输出见
  [当前证据](../../../../evidence/correctness/foundation-f2-public-program/receipt.json)。
