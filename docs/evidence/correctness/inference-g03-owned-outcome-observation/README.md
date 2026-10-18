# G03 原结果 completion 来源聚合：实现检查点

2026-10-14；baseline `16d677b38db855482ada49ed260a3611f5c17147`；scope `169783be6`；
audit `e34080781`；当前三 owner 保持 AUDITED，不是 terminal certificate。
[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-owned-outcome-observation.md)。

OO01/OO02 已实现 trusted 原结果 completion access，以及原 accepted phases/实际使用的
producer bindings 的去重借用；不观察、不分配、不改原对象 identity。OO03 是注入 allocator
的 opt-in 冷槽，保原 suffix completion 和唯一来源 owner，用原 actual-retirement callbacks
分别确认 suffix/source。source 只在 suffix 实际退休后观察或 abandon；回调不等待其它回调。
OO04 通过原 OwnedResult completion 接入，同一 completion()/wait()/result() 状态、同一
payload/ResultSlot，不创建另一个 executor、queue、结果槽或默认 native 字段。

## 实际通过

| 入口 | 实际范围 | 原始记录 |
|---|---|---|
| Windows/MSVC | 定向 compute/Vulkan compile/link；29 completion/result tests，含 DSO owner、unknown、abandon、两个冷槽 64 次无 warm metadata 分配 | [receipt](windows-receipt.json)、windows-raw.zip |
| Air source fixture | Apple3/Vulkan5 原 attention ownership、接纳后 source 取消、重复结果与原槽、abandon；Vulkan unknown/真实 fence wait 和 unknown abandon；另 actual CPU 1024-byte F2→suffix oracle | [receipt](source-receipt.json)、apple/vulkan-native logs、native-cpu log、air-raw.zip |
| Air host risks | 两原生路线各 ASan/UBSan/TSan；每配置原 native3/5 + core28 + CPU7（含 OO02 使用/未使用 producer、容量不消费与 identity） | [receipt](host-receipt.json)、air-raw.zip |
| Air SDK | 两路线 static/shared 四配置，搬迁 prefix，普通 hyle::compute；每配置 4 独立头和原结果 64 episodes，一致成功/取消、原槽与原 payload | [receipt](sdk-receipt.json)、air-raw.zip |
| WSL/CUDA | 原 gated memset→F2→同原 Queue suffix，1 KiB；success/原 Unimplemented cancel/unknown query/unknown abandon 四案例；实际原 guard/reaper 与原 owning WriteLease 结果，success 独立 byte oracle | [receipt](cuda-receipt.json)、cuda-raw.zip |

Air 实际 Mac16,12 arm64、macOS27.0.1/26A434、Apple Clang21；WSL 的实际硬件/编译器见
CUDA receipt。heavy 全串行、最多四 jobs。CUDA dry-run 在执行前确认没有 .cu.o/nvcc；
实际只编 common3 与 consumer1，不改旧 cuda_backend.cc 的单 host TU 观察 hook 设置。
Windows 首 build 的测试 C4834 nodiscard 警告 FAIL 保留，修正测试后第二 build/test PASS。
累计 source50.023s、host159.973s、SDK55.653s、Windows241.594s、CUDA69.105s；不重置失败。

## P0 与限制

普通 Completion40B、OwnedResult<int>48B，原字段不变。Air SDK 冷 observer+int result
storage 实测412B；两槽测试只冷分配两次，64 warm episodes 不调用元数据 allocator。
source fixture 每 episode 的 make_shared/64-node bounded storage 是测试成本，不是生产
pending adapter 已实现零 warm allocation 的证书。OO02 的真实成本是两次有界
binding × accepted-producer-entry 扫描，再填 accepted phases；不称为线性扫描。
没有数学/normal native algorithm 改动，不重跑已闭 P1 或旧全模型矩阵。

宿主 TSan 只复用两份已有、逐 SHA 核验的 intrusive final-fence/Metal callback 模型；
没有新模型、suppression、LSan、shader/driver/private-pool 完成声称。原 source/host 输入包
保存实际源码；dependent_completion.h 在这些测试之后仅追加两行 release 合同注释，
封存脚本严格证明去掉这两行即原头；SDK/CUDA 使用含该注释的当前头。

仍需原 partial accepted-prefix/producer failure 的对抗 fan-out、生产 source owner 冷预留
及逐节点 OOM、实际 Graph 集成。主 G03 的 typed pending page、late native output、七路线
64-step 两请求/两 slots 和数学 oracle 未完成。旧 Graph Windows gates、G04–G10 保留。
本检查点不代签这些事项；详见 [verdict/remaining](validation-result.json) 和
[canonical 输入 SHA](product-inputs.json)。
