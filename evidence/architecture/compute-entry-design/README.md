# Compute/Graph 入口设计证据

本目录只支撑 [设计轮](../../../docs/implementation/tasks/build/compute-entry-design.md)，不证明新 API、
联合运行、三交付或性能资格。baseline 为 `17ff6555c7af0a8bd44d648209c40f53de758420`。

- [inventory.json](inventory.json)：74 个去重入口族/配置项，源码行/SHA256、owner/consumer、能力影响和设计裁决；
  [inventory.py](inventory.py) 是此快照的一次性只读复现器，不是新增项目治理/验证驱动。
- [llama-reference.json](llama-reference.json)：官方固定 commit
  `e60eff95fda7b6e46bf402a8e9f1e86d88ab29c8` 的四源文件身份及有界原文，
  对账层/专家 buffer 放置、算子 admission、分图、传输和完成；不导入 product dependency。
- [loc.json](loc.json)：统一 LOC 工具/自检输出，Compute+Graph 实现 62,486 code lines、268 files，
  不含 tests/bench/docs/third_party，包含接口/资源/胶水及 src 内维护片段。

复现入口统计：

```text
python evidence/architecture/compute-entry-design/inventory.py --output .scratch/compute-entry-design/inventory.json
```

复现读固定 Git revision 的源码，不受当前文档改动影响。上游研究可按 JSON 固定 URL/commit 再读取
原文并核对 SHA256；本地原 ggml lock pin 对账见设计卡。源码与工程推论分开，不用 peer 的能力或
async 函数名字签发 HYLE 支持。
