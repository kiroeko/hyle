# R6 最终两系统三种 C++ 交付

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`；完整源码交付快照
`20a029aded4bc00c9fc80a182bd3051853e7c99e`，同一个 source ZIP 先在 Windows、后在 Air 消费。
其 SHA-256 为 `22d4cce9a4f9c9f912b62877c8bc2a13a6d17f08a96d40c5499a3dbb5e3c5fa6`。
[逐路径身份](../vulkan-r6-product-delivery/p0-unchanged.json)证明七项 shipping 输入与产品及
两系统测量快照完全相同；文档/状态/benchmark 差异不冒充产品改动。

## 已执行旅程

| 形态 | Windows RTX 4070 | Air Mac16,12 / MoltenVK |
|---|---|---|
| 完整 source ZIP 解包、离线依赖、公开 facade 嵌入 | Direct + Graph PASS | Direct + Graph PASS |
| installed static SDK，迁移安装前缀后独立消费 | Direct + Graph PASS | Direct + Graph PASS |
| installed shared SDK，迁移安装前缀后独立消费 | Direct + Graph PASS | Direct + Graph PASS |

每系统实际生成并运行六个消费者 EXE，合计五次 CTest 调用（source 同时含两入口）。样例执行
完整 F32 门控 FFN、变化输入、重复回放、应用自定义 SPIR-V 与独立数值 oracle；通过普通
`hyle::compute` / `hyle::graph` 产品入口，不以内部 factor target 或设备枚举替代。

两系统均在 installed SDK 消费前隐藏完整 producer source、source build、static build 与
shared build 路径；消费者源码另行复制。迁移后的 SDK 有独立前缀，成功不依赖原 build tree。
核对安装头、CMake exports、第三方许可证及 CPU/CUDA/Apple/oneDNN binary 缺席；仅启用 Vulkan。
Windows 的 HYLE shared DLL、所需 C++ runtime 和 Vulkan loader 实际放在 EXE 旁；运行前设置
非交互错误模式。每个消费者的实际 `dumpbin` / `otool` 链接信息和 executable hash 都有记录。

build parallel 2 / CTest parallel 1，两个主机按串行队列执行；没有重新运行已完成的 P1。
完整源码内含的依赖/许可证与已嵌入 SPIR-V 在 installed 两形态中按同一规则投影。

## 归档与完整性

[identity.json](identity.json)记录两系统交付裁决和 archive 身份；
[Windows consumer identity](windows-consumer-identity.json)、
[Air consumer identity](air-consumer-identity.json)保留各实际命令、返回码、制品、source ZIP 和
producer 隐藏事实。归档工具再次核对全部命令返回码、五次 CTest、六个 EXE 的 hash、
原 producer 路径缺席及隐藏目录存在。
Windows consumer identity 的外置浏览副本只将 CRLF 换为 LF，JSON 数据逐项相同；原始字节
完整保存在 `windows-raw.zip` 的 `logs/identity.json`，其已记录 hash 不变。

- [windows-raw.zip](windows-raw.zip)：79 个具名文件；SHA-256
  `94084a42c92e9b8d46da96306bbe1291b0e1ca4878adf9d75d16fb4d9edad6ce`。
- [air-raw.zip](air-raw.zip)：78 个具名文件；SHA-256
  `bedcc3773ace5027864afcce3ea53ec5c478b15ce41170817dd8ea255474d61b`。

两包另含逐项 archive-manifest，保留完整 configure/build/install/consumer/CTest/linkage 日志、
SOURCE-MANIFEST、exports、许可证和任务脚本。Windows 接收后重算所有条目和 archive hash，通过。

这完成最终两系统 source/static/shared 项，不替中央 broad selector 或路线 terminal 签证。
平台/设备/数值及外部运行库限制仍按 R6 主卡，旧 `60de4212` 交付检查点继续作为历史事实保留。
