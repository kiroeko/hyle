# 许可与第三方代码边界

## 当前许可

HYLE 原创代码、文档和构建文件采用 [Apache License 2.0](../../LICENSE)。
版权方写作 `The HYLE Authors`，指向全体贡献者，归属以 Git 历史为准。
根目录的 [`NOTICE`](../../NOTICE) 是 Apache-2.0 第 4(d) 条所指的归属说明，再分发时须保留。

原创代码的许可已按维护者选择统一为Apache-2.0；完整条款以根目录原文为准。

第三方组件的具体许可证、附带归属和分发材料按下面的清单逐项保留；
不能只根据 HYLE 自身的 Apache-2.0 声明推断整个依赖闭包的许可或任意组合的兼容性。

源文件需要短标识时写：

```text
SPDX-License-Identifier: Apache-2.0
```

原有 `LicenseRef-HYLE-Proprietary` 文件头和SIMD生成模板已统一为 `Apache-2.0`；
不向第三方文件套用此标识，也不改写历史实验、归档或旧提交中的许可记录。

历史提交包含旧许可声明。使用旧版本时同时读取该版本的 `LICENSE`、文件头和第三方清单，
不要用当前工作树的声明代替历史版本的材料。

## 原创与第三方的边界

Apache-2.0 只覆盖 HYLE 原创部分，不改变 vendored、ported、fetched 或系统依赖的许可证。
权威清单是根目录 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)，版本与哈希在
[`cmake/HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json)：

- `third_party/` 内的实现保留原 LICENSE、NOTICE 与源码版权头；
- `src/simd/sort/vqsort_hyle/` 的 Highway 移植和 `src/platform_thread/wait.cc` 的 Rust Parker
  移植位于 `src/`，仍受各自原许可证约束；
- 修改第三方文件时保留原声明，并按 Apache-2.0 第 4(b) 条或原许可证要求标明改动；
- fetched 或外部后端只有在二进制、传递依赖和许可证文件都进入导出图后，才能称安装包可迁移。

## 构建与分发

安装包必须包含 HYLE 的 `LICENSE`、`NOTICE`和第三方总清单，分别安装为
`share/licenses/Hyle/` 下的 `HYLE-LICENSE`、`HYLE-NOTICE`及 `THIRD_PARTY-NOTICES.md`，
同时保留实际选中实现所需的许可证及归属文本。Runtime-only与Development-only安装均须携带相关材料。
构建系统按 target 图选择这些文件，缺少许可证和缺少 DLL 或 archive 一样属于不可发布。
独立引擎安装同样携带项目许可文本；其依赖SDK仍保留自身许可证，不以引擎的Apache声明替代。

完整源码包由 `hyle_source_package` 从干净的 Git tree 生成，包含全部受版本控制的源码、测试、
benchmark、文档、脚本和 in-tree third_party，并用 `SOURCE-MANIFEST.txt` 固定 commit。
它不包含本机 build、cache 或 fetched checkout；离线用户可另行取得依赖源码，通过标准
`FETCHCONTENT_SOURCE_DIR_<NAME>` 输入组装。

OpenSSL、CUDA、GPU驱动等采用供应商各自的条款。由系统提供的部分不因此变为HYLE代码；
实际随包复制的运行库必须保留对应许可，不能笼统声称全部由系统提供。
Windows明确声明为SHARED的OpenSSL导入携带同一SDK的原始许可；共享SDK保留所选DLL和文本，
静态SDK使用消费者自己的OpenSSL。无法确认动态身份的UNKNOWN库仍为外部依赖，不猜测复制。
LibTorch及PyTorch CUDA包含自己的依赖；主BSD许可不覆盖其所有附属库，须保留所分发平台的实际材料。
锁定CPU SDK缺少部分原始声明，由`third_party/torch_sdk/`补充PyTorch、附属组件及平台运行库条款，
`ORIGIN.json`记录真实build commit、原文件哈希及核实边界。GNU／LLVM OpenMP的精确源release尚未确认；
所保留的完整许可文本不能代替精确源码身份或证明全部源码提供等分发义务已经满足。

Apache声明与再分发要求按[官方许可证原文](https://www.apache.org/licenses/LICENSE-2.0)核对；
本项目的许可证选择不改变第三方原有的版权、专利或供应商限制。

本页是仓库的工程政策，不构成法律意见。
