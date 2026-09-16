# R4 CUDA 正确性与 host 插桩尾项

普通/插桩基线 `f1cf11121b5d4ad9529e3079b82b7e19717e2420`；TSan 可见性修正
`a8c982496984c16b33c3b1d6f3a7622058f5b92c`。均由 Windows 完整 Git archive 部署 WSL，
不回写验证副本。原始失败、命令、cache、编译参数及测试日志保存在 [WSL 原始归档](wsl-raw.tar.gz)
和 [Windows/任务脚本](windows-and-recipes.zip)，见 [机器汇总](summary.json)与 [哈希](sha256.json)。

| 范围 | 实际结果 / 身份 |
|---|---|
| WSL GCC14 普通 shipping | f1cf1112，六个 CUDA factor 的 13/13 CTest PASS |
| Windows MSVC 普通 shipping | a8c98249，13/13 PASS，包含新 cuBLAS metadata OOM 回滚/无热分配与 callback 夹具 |
| WSL ASan | f1cf1112，六个头检查 PASS + 七个 GPU 程序 PASS；下面的精确驱动例外明确保留 |
| WSL UBSan | f1cf1112，13/13 PASS，无 suppression |
| WSL TSan | f1cf1112 原 11/13 PASS；两项具名可见性修正后 a8c98249 的 backend/Graph 2/2 PASS，无 suppression；不重跑未变的十一项 |
| CUDA device memcheck | Compute Sanitizer 2026.2.1，f1cf1112 普通 Direct 2/2、实际 captured Graph 3/3，全链路各 0 errors、0 bytes leaked |

W1/WSL 是同一 RTX4070/SM89，重型运行串行，Ninja 并行 2；Windows exe 旁实际复制 CUDA/CRT DLL。
普通 GPU correctness 不限于静态数据：原测试包含双槽变化输入、dense/Q4、多 Provider、资源/完成/
close/abandon/error 寿命及 metadata 失败回滚。故意非法设备访问的隔离 late-failure 程序参加正常
和 host 插桩检查，未放进要求零 device errors 的 memcheck 消费测试。

## ASan 的已定位外部边界

默认 ASan 环境使仅调用 `cudaGetDeviceCount` 的独立程序返回 CUDA out-of-memory，与 HYLE 无关。
采用 [NVIDIA 已登记的 shadow-gap 设置](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/troubleshooting.html#issues-with-cuda-and-address-sanitizer)
后正常初始化。最小程序即使调用 `cudaDeviceReset` 仍留下 WSL `libcuda.so.1` 内的两笔 260/20 bytes。
七个产品程序的未抑制原报告均恰好只有相同两笔、同一 allocation stack；所有测试断言通过。

最终只排除此 driver module 的 leak 报告，**逐程序强制核对 suppression 为 2 笔、280 bytes**；
数量或字节数改变就失败。没有关闭 leak detection、ASan 内存访问检查或全部 CUDA 错误；未声称
闭源 driver/cuBLAS 内部已插桩。该限定绑定 driver BuildId `8587eaad68ff611de8f0a2c728566e155c744a35`。

## TSan 的两个有限模型修正

独立两线程程序复现 GCC TSan 不能识别 release RMW + final acquire fence 的报告；符合 C++
release-sequence 的最终析构始终读到已发布 payload。仅在 TSan 构建中增加现有 counter 的 acquire
观察，不改变实际引用计数算法或普通路径。测试 callback 则在最后一次访问后发布 finished，owner
完成原 native 等待后 acquire 观察，令工具理解真实完成关系；没有用测试信号代替 CUDA 终态。

[定向审计卡](../../../docs/implementation/tasks/compute/cuda-r4-host-validation-visibility.md)与
[shipping 源码等价证明](shipping-source-equality.json)保留边界：去除唯一新增的 TSan 条件块后，
产品源码字节与 baseline 完全相同。ASan/UBSan 及三交付的产品路径因此可继承；fixture 改动通过
新的 Windows 与 WSL TSan 检查。不重建已通过交付，也不因这些改动重跑 CPU P1。

本证据只完成 R4 的正确性/插桩尾项；正式 CUDA P1 与默认优选规则的完整性能资格仍未签署。
