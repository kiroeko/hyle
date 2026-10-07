# CUDA U3：严格 F32 与完整 native 联合验收

产品／测试候选 `7f6c9107235868d63892d9117a293d1007ba3471` 在 Windows 与 WSL Linux 普通
Direct **651 项**、Graph **649 项**通过，无 skip。包含559固定GGUF数值合同／104制品身份、
34 non-GGUF packed、当前official／三sidecar、Kimi／GLM必要完整链路与本地Qwen实际权重。
原Provider资源／编码／精度回归另通过。精确命令、候选及范围见 [receipt](receipt.json)、
[native batch](native-batch.zip)，Linux测试全集见 [名称核对](linux-test-union.json)。

## 具名根因与修复

原ggml float MMF tile在`mma.cuh`执行TF32乘数指令。HYLE已设置完整F32精度要求，但原
MUL_MAT／MMID分发及MMID capture suitability没有先消费该条件。小N／dyadic输入绕开实际
分支，不能为大shape资格。仅在这三个选择点排除要求F32的F32 MMF，原cuBLAS／sorting程序
承载；默认native／量化／F16/BF16选择与数学kernel保持。原source／pin／NOTICE／软件身份
如实更新，未发新版本性能WIN。详见 [施工卡](../../../docs/implementation/tasks/compute/cuda-final-u3-f32-dispatch.md)。

Kimi MLA大shape回归用不可TF32精确表示的乘数和重复ID，独立源方程通过；完整三profile的
九输出、四replay、actual cache反馈和原点值门通过。临时operand copies、reference factory及
actual-state验证包装全部移除，恢复原独立完整轨迹。早先“仅reference/cache”的解释被源码及
精度回归证伪；失败、诊断与旧源大shapeFAIL保留，不以放宽门或替换末端参考签通过。

## Linux 批量与资源边界

第一次单进程合并全部carrier/模型测试时显存接近12 GiB，Direct达到900秒timeout；Graph
在记录保留后由拥有者停止，不能写PASS。live stack停在CUDA event等待，后来仍有慢进展，
不误报为死锁。分carrier/合同族在新进程中验证同一不可变候选，GGUF进程约1 GiB显存且
560项约59秒通过；全部分组的测试名称union与原请求全集逐项相等。Direct为34+560+21+36，
Graph为34+560+21+34；源码、输入shape／精度、误差门和case集未变，未无限增加采样预算。
这证明当前批量验收，未把原超时翻成性能胜负或承诺vendor private cache的严格总上界。

## 交付与尚未完成

本证据是native正确性／资源／P0资格。最终联合CUDA-only／CPU+CUDA的source/static/shared、
offline／relocation／邻exe DLL／loader仍待；U4 defaults/best-of、U5清退及U6汇总仍未完成。
本卡不签整个CUDA路线CLOSED，不声称全部权重整模型常驻或推理engine。

统一脚本 [LOC](loc.json)：全Compute + Graph实现 **60,658行／253文件**，不含测试、bench、
第三方数学或构建脚本，`--self-check`通过。本批没有P1。
