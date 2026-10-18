# F5 CUDA public-link真实影响与旧证据资格

77df→25e产品仅三文件：中央CUDA直接FI边PUBLIC投影、CUDA owner非准入安装过滤、
cuda.h的实际FI资格guard include。用既有CMake metadata投影逐行比较，唯一变更是
compute_route_cuda的PUBLIC/PRIVATE边归属；该边union及全部其他factor/product行不变。
四个已签Windows CPU/Vulkan及Vulkan-only SDK都未选CUDA factor，实际prefix没有cuda.h；
它们供应的源码/头/API/数学/可见依赖不受本改动影响。保留原77df artifact/日志/hash，
按这个精确cut恢复25e资格，不把历史run改成新run。CPU/common san同样不选CUDA。

Selector对任意central manifest edit保守选择全factor/product。用户活动goal及有限card
要求只实际owner/直接consumer/package seams；这里不回放无影响旧算子全集或重写selector。
公共component configure合同另外实际运行，两个Linux与两个Windows CUDA SDK重新资格。
其他十SDK及native host-control san/P0/LOC/guide/红队仍未由此签。

精确before/after行、产品文件与四原receipt hash见[receipt](receipt.json)。不签完整closure。
