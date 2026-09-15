# Compute/Graph 六项架构复核

## 本轮合同

- baseline `624f81df`；2026-09-15 用户要求按路线隔离、调研/实验、Apple 组合、低成本渐进路由、
  多后端 Provider 融入和文档清理六项 review。旧 Compute/Graph 不构成设计约束。
- 只做源码/API/已有证据复核、有限研究程序和文档更新；不写产品、不更改 registry、SDK/依赖锁、
  公共治理工具，不恢复大 goal，不扩充 shipping 算子。模型 CPU+GPU 联合推理仍不在范围内。
- 预算 75 分钟；最多一个新 Apple 探针、一次 fixture 纠正。已有 CPU/CUDA/Vulkan 证据复用，不跑 P1
  或全库构建。不把“采用成熟实现”或“小代码通过”签为四条路线已经 SOTA。
- 新探针：MLX v0.32.2 的 GPU Q4 affine MatMul → 自定义 Metal SwiGLU → GPU Q4 MatMul，
  M=1/7、K=H=256，每形状三次变化输入。CPU 从实际 packed codes/scales/biases 独立解码并以 double
  累加作 oracle；逐值误差 `<= 1e-4 * max(1, abs(expected))`。保留第一份结果跨后续执行，检查不被
  覆盖；临时输入 owner 在提交后、最终同步前释放；非法 K 在调用验证阶段拒绝。不是故意 device-loss。
- 只签所测正常 lifetime、形状/参数拒绝与量化组合，不签任意外部 Metal buffer/Queue import、M5 NAX、
  并发、完整错误恢复、P1 或三交付。源码来自 Windows 完整 Git archive，Mac 仅验证。
- 文档清理：长期边界分别归 Compute/Graph/路线设计，删去架构中的重复任务表和旧实现细节；历史
  registry records/原始证据不伪造或抹除，失效计划退出当前入口并标明可从 Git 追溯。
- 验证：上述有限探针、change-impact selected docs/closure、`git diff --check`。
- 停止：逐项给出事实、问题、采用方案与尚未证明的资格；不以未完成的产品资格继续无限调研。

## 结果

复核进行中；未完成项不预写 PASS。
