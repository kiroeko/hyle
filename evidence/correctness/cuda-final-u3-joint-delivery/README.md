# CUDA U3 联合交付：执行记录

本批仍未关闭；此前 native 两平台全集资格见
[原资格](../cuda-final-u3-f32-native/README.md)。本页不把 native 或部分 delivery PASS
代签 source/static/shared、CUDA-only／mixed 全部完成。

## 分组、证据复用与停止

任务驱动 [run-groups.py](run-groups.py) 是本轮专用入口，不改变公共验证治理。
它按实际 executable、邻置 runtime、机器／驱动、fixture 哈希和影响执行的环境固定身份，
保存各组原日志／XML／哈希；用例并集必须逐名等于请求全集。身份匹配的 PASS 复用，
失败项停止自动重跑；显式重跑需要具名影响原因并保留旧日志。

先用 `--plan-only` 查看每组 RUN／REUSE／STOP 及理由；实际执行去掉该参数。
文档变化不触发 build 或 P1。新 binary／ABI／依赖身份不得覆盖旧输出目录，先说明失效范围。
各交付的 link、relocation 和 loader 边界分别验，不能拿 native 证明替掉。
Windows SDK 运行使用 `--minimal-path`，测试进程的 PATH 仅为系统目录，依赖DLL邻exe。

## 已有事实与未决失败

- 候选 `056e67a4` 的独立 Windows CUDA-only static source 消费批次650项中649项通过。
- Q3_K expert `g3560f36d25c826bd` 的 replay1／slot1出现大误差。相同未变 executable
  的单例、560项物理编码组、34个前序 N1／N2加该单例35项均通过。
- 尚不能唯一归因；原650项失败保持未解决，不能以这些定向 PASS 注销。
- 后续交付检查可以推进，但有关 aggregate gate 在因果解释／纠正验收前不签 PASS。
- CUDA-only static SDK已实际 install／搬移，并由独立消费工程离线 configure／链接成功；
  其实际运行仍待，不能仅从 build PASS推断 loader／数值资格。

原数据暂在 Windows canonical 的 `.scratch/validation/cuda-final-u3-delivery/`；
阶段完成时按原日志／XML哈希封存。没有放宽误差门、改用例输入或追加自研数学。
