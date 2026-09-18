# C1-Y4/Y20：managed Vulkan 三交付与所选回归

本批 source snapshot 为 `351ec13d82057402594a96fcc862bcdc96b2d839`，产品候选为
`3c39428e3fdf1911b284f7632d891f663ad82f31`。两者的产品、构建和依赖文件相同；后者之后仅
迁移普通 Graph FFN 样例的报告断言为4 native dispatch / 1 managed region。

## 已运行的范围

- Windows 和 MacBook Air 使用同一份独立 source archive，分别完成 source、迁移后的
  installed static/shared SDK 的普通 Direct/Graph 消费，各6个 consumer，合计12个。
- FFN 由 ggml managed gate、原生 MatMul 和应用 SPIR-V 组成，使用生成权重、独立 double
  oracle、1/16行及尾宽、三轮变化输入和旧结果寿命检查。关闭其他计算路线，保留许可证、
  exports、离线依赖与 producer 隐藏检查。此处不声称真实 checkpoint 或整模型已通过。
- Y20 依赖补丁身份变化所选112个 gate 全部通过：96个基础/逐头等检查、1个组件合同、15个
  通用组件三交付检查。8个未选项及理由保存在 `result.json` 和原始 selector 中。
- Windows DLL 均放在 executable 旁，运行时取消 PATH；没有运行 P1、改变电源模式或申请
  UAC。Air 的实际 Vulkan consumer 使用 validation layer，日志未出现 VUID/同步错误。

`result.json` 保存身份、计数、哈希和限定 verdict；四个 ZIP 保存完整命令/输出、CMake 配置、
consumer/SDK 身份与执行详情。最初组件合同因新离线 cache 缺锁定 Highway archive 而失败的
日志也保留；补入本机已有且哈希相符的归档后，仅重跑该失败 gate，没有修改产品或公共工具。

## 清理与限制

先验证归档，再删除 Windows/Air 的过期 producer 和45个重复组件构建目录；三个 cleanup
JSON 记录准确路径和逻辑字节。保留独立 consumer、迁移 SDK 与共享依赖缓存。

这里只签上述交付/所选回归为 PASS。C1-Y/C1 仍为 INCONCLUSIVE：TopK、checked indexing、
Scan 等剩余成熟能力映射，真实2B权重片段、完整四路线资格、默认性能优选及替代实现淘汰均
不能由这批收据代签。CPU oneDNN 的性能资格 BLOCKED 不变。

`loc.json` 由统一脚本对不可变 source snapshot 统计：Compute + Graph **46,990行、193个
实现文件**，包含 GLSL，排除测试、bench 和第三方源码，self-check 通过。
