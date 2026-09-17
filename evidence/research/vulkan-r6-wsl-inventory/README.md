# R6 WSL Vulkan 环境盘点

一次只读 loader/device 盘点：[inventory.json](inventory.json)、[原命令程序](inventory.py)和
[stderr](stderr.log)。成功创建 Vulkan 1.2 instance 并枚举实际设备；当前仅有
`llvmpipe (LLVM 20.1.2, 256 bits)`、device type CPU，API 1.4.318。已安装的其它 ICD JSON
不等于对应物理设备可用；`vulkaninfo` 与 `glslangValidator` 均未安装。

这是环境事实，不是 HYLE native 正确性、性能或 Linux 交付资格。R6 当前实机验收域仍为 Windows
RTX 4070 和 Air M4/MoltenVK；不把软件 device 当成原生 GPU、不切换成默认 CPU fallback。
Linux GPU、AMD/Intel 等保持未资格化，后续出现实际目标设备时再按其能力与交付合同验证。
