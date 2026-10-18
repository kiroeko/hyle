# F5 candidate WSL cpu,cuda shared迁址SDK

固定candidate25e4fb398，GCC14.2 Release/shipping/shared，actual RTX4070/CUDA13.3。
Windows canonical单向固定产品overlay到disposable WSL；原F4 source manifest保留，
F5输入manifest逐文件hash核对。复用同身份producer/cache，只构建batch与直接Graph route。

普通Compute/Graph发现，公开CPU/CUDA cpp复制进隔离consumer-source；编译仅从rename后的
prefix发现HYLE头/库。CPU四域Direct/Graph mean oracle PASS；actual FlashInfer CUDA
paged/ragged×F16/BF16、D256、两请求、GQA/右对齐causal/乱页/tail/容量有效区与old result
Direct/Graph共八次非uniform独立oracle PASS。最大误差0.00228以内，阈值0.007。
这是原Linux FI数学，不由Windows unsupported诊断或CPU/reference代签。

15个实际供应头独立TU PASS；Apple/Vulkan缺头/缺route具名拒绝。
实际prefix rename、package metadata无source/build/staging路径；ldd全部解析，shared
HYLE库必须来自迁址prefix，运行删除LD_LIBRARY_PATH。命令/身份/输入hash见
[receipt](receipt.json)/raw.zip。static HYLE不假称第三方CPU runtime全部静态。

首个static编译因新测试错误使用Result::ok而FAIL，改为公开Result布尔检查；shared真实
公开factory link FAIL与owned-header配置FAIL驱动唯一owner修复。PUBLIC边、资格guard内
公开include及非Linux absent BUILD_INTERFACE均原能力投影，不新增API/数学/runtime。
原日志、输入history及elapsed保留，同root重新配置/安装/隔离消费，未重置预算。

本格仅签CANDIDATE SDK，20 owners非CLOSED；其他SDK/san/P0/LOC/guide/红队仍按固定卡。
