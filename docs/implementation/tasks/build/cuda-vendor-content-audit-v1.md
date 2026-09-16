# CUDA vendor 内容身份与消费前校验

## Closure card / SCOPED

- baseline：`bf685ea3b2d3317d67802c4d1436c9bf1f440267`；owner 为
  `compute_provider_marlin`、`compute_provider_cuda`，仅重开依赖锁与构建 seam。
- trigger：R5 完整源码打包的全 vendor 审计拒绝 Marlin 与 SGLang，工作树及两 vendor Git blob
  均未改变。不是新算法/性能失败，不能绕过源码交付检查。
- 主旅程：完整源码 archive 的内容校验；正交旅程：CUDA ON 消费真实 vendor 时必须先验证内容，
  CUDA OFF/core-only 不要求这两份源码。
- 继承：两 owner 的旧 terminal/P1 保存在
  [baseline](../../../../evidence/correctness/cuda-vendor-content-audit-v1/inherited-baseline.json)。
  R4 的运算代码、默认规则、布局、完成语义、三交付执行和限定性能证据不重写。
- 预算：15 分钟具名排序/接线审计，两个有限 FIX；一次完整源码审计与负向校验，重型串行。
- 非目标：不换 pin/算法、不改哈希算法、不重写公共 registry/selector、不重跑 GPU P1。
- 验证：中央 CMake 正规范摘要与独立 Python 复算、Windows/Unix 完整源码审计；修改任一 vendor
  的验证副本必须在它的 enabled consumer 配置前被拒绝；OFF 保持无依赖；适用 selector 与 docs/closure。
- 性能：P0，只有配置期文件校验；原 C++/CUDA blob 完全相同，继承 R4 P1，不进行新计时。
- stop：锁摘要、消费前校验和真实源码交付通过即 CLOSED；不延伸为全库重审。

## 待固定 FIX

| ID | 缺口 | 允许动作 |
|---|---|---|
| V1 | 新增两 vendor 的锁摘要与既有 CMake 规范不同 | 复核逐文件 blob，使用现有函数生成正确摘要；不改 vendor |
| V2 | 两个 CUDA owning factor 没有消费前树校验 | 各自 enabled 分支调用现有 vendor audit；OFF 不检查 |

七问：成熟来源/pin 不变；真实 source consumer 失败；复用已有 audit；冷配置 P0；纯 C++ 无新 FFI；
三交付运行字节继承且补完整 source 检查；普通用户获得早期、明确的内容身份错误。
