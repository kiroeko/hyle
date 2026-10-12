# Foundation A Apple：首批 Air source 资格

产品来源为 Windows canonical 的实施 commit e5ab48c2a674d176a9739cf9af89c242d9552af6，
完整 git archive / SHA256 / checksum rsync receipt 为 source-sync.json；Air 只作可丢弃验证副本。
这不是 CANDIDATE，活动范围见[四路线卡](../../../docs/implementation/tasks/graph/foundation-a-route-symmetry.md)。

实际身份：MacBook Air Mac16,12 /16GiB / macOS27.0.1 build26A434 / Apple Clang21.0.0 /
CMake4.3.4；沿原锁定 MLX/ggml Metal 与 Metal tool shim，未用 Mac mini。
源码 shipping shared，HYLE_TESTING=OFF；复用 a-cpu-apple source 树，只有一个定向 fixture target。
configure5.944秒 PASS，build20.018秒 PASS，journey2.687秒 PASS。build 时实际可用
7,418,085,376 bytes，按本轮≥4GiB/<8GiB 规则 j1，test1；不伪报 j4。

[真实 fixture](../../../tests/graph/route_symmetry_apple.cc) 通过 native Apple 与 CPU+Apple：
两形状 exact table、表外拒绝、独立 native slots/旧 array 输出不变、纯控制前驱保留和
原 managed completion 排序、CPU/GPU 控制分片、host_wait=kReject 拒绝、hybrid 数学与
旧结果阻止 reset。没有借 CPU 替换 Apple 数学，也没有新数学 kernel/通用 runtime。

本目录没有代签 checked control 前驱失败/cancel/OOM 的全部 native 风险、installed static/shared、
新头/缺路线/迁址、其他机器、effect ports 或红队；对应未完成格仍 UNPROVEN。
原 source 47 个增量步骤含受公共头影响的 Graph/provider TUs；没有全算子回放。
job.json 有身份、实际命令、进程、available bytes、各产品输入 hash 和最终日志 hash。
