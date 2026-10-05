# CPU标准构建：默认取得已验收Provider

<!-- MODULE REVIEW: AUDITED -->

- baseline：`59f68e326`，Windows canonical；CPU官方四画像新能力已独立CLOSED，工作树干净。
- owner：`compute_provider_native`。用户要求普通用户只选CPU模式，HYLE处理成熟库依赖；
  真实新事实是当前Torch AUTO只发现本地SDK，未安装时静默缺失已验收的官方FP8/FP4域。
- scope：CPU-only LibTorch的取得、锁/缓存/离线与默认CPU入口说明；不改变库版本、数学、API、
  Provider顺序或执行/线程合同，不修改公共source-cache/性能统计核心。
- 普通消费：`HYLE_COMPONENTS=compute;graph`默认CPU ON、其它路线OFF；同一默认闭包接好
  oneDNN、ggml及本轮已验收TorchAO/ATen，用户不枚举Provider。
- 高级控制：既有route/Provider ON/AUTO/OFF、显式SDK root、统一cache和disconnected；
  增加可选本地锁定SDK archive输入，适用于代理不稳/离线/复用既有下载，不增加路线枚举。

## 有限审计与FIX

| ID | 允许改动 | 验收 |
|---|---|---|
| CB1 | 在中央dependency lock登记CPU SDK Windows/Linux x86_64原版archive/hash；Native私有CMake取得器读取，默认缺本地SDK时取得已验收版本 | 锁与原来源一致；CPU OFF/不支持目标不探测或下载；显式错误root拒绝 |
| CB2 | 复用统一source-cache根中的独立SDK子区，按平台/hash唯一身份，锁住单写入、原archive SHA校验、完整原SDK发布；离线命中复用、离线miss具名失败 | 本地锁定原archive一次解包，两fresh配置复用；坏archive/不完整SDK不启用；不每build复制SDK |
| CB3 | 普通CPU preset/指南及高级参数、SDK三交付定位与DLL staging；标准模式不静默降级 | 普通CPU配置实际启用全部Provider且无GPU探测；source/安装消费者复用已验SDK，不要求逐库flag |
| CB4 | 原LibTorch导出的target图没有完整描述Windows四个间接DLL（asmjit/fbgemm/libiomp5md/uv）。已有通用runtime staging只看TARGET_RUNTIME_DLLS，有限增加owner关联的runtime文件声明并按真实consumer链接闭包收集；Native在同一SDK接缝声明七个原DLL | 一次CPU独立消费者调用既有hyle_stage_runtime_dependencies即可邻置完整闭包；不相关目标不复制Torch；既有staging幂等与增量合同回归 |

验证预算：一批CMake取得/隔离/离线/缓存与错误注入；复用现有原SDK及构建，不下载另一份相同SDK；
按selector运行适用构建/交付合同和普通C++消费者。重型j2串行，DLL邻置/隐藏子进程。
P0为configure/缓存与target闭包成本；产品C++、SDK内容、执行路径不变，不重跑官方完整P1。
公共锁加入SDK元数据不改变旧source记录与读取/验证核心；若发现需要改公共治理，另记具体问题。

CPU自研算子逐域裁决由[有限裁决卡](../compute/cpu-native-provider-domain-decision.md)承接用户最新要求，
不以本构建任务宣布旧自研域获胜或全部清退。CPU这两项完成前不处理Apple/CUDA/Vulkan剩余项。

## AUDITED

已核对CMake闭包：普通Compute/Graph默认CPU ON，其它三路线OFF；oneDNN AUTO已自动取得源码，
ggml已在标准CPU闭包；只有Torch AUTO缺少自动取得。CPU SDK有明确已验Windows/Linux原archive/hash，
无需新增数学、公共API或性能验证。源码cache核心只管source身份，不能把prebuilt SDK伪装成Git源树。
CB1–3固定：中央锁增加SDK附属元数据；Native私有取得器在统一cache的独立SDK子区存一个平台/版本，
本地archive直接验SHA/解包，默认在线取得，断网只复用本地输入/完整命中。显式坏root/archive失败，
不静默换库。静态安装依赖shim先解析/取得同一SDK再find Torch，免用户重复列Torch prefix；
shim/取得器/原锁随Native static development seam交付。OFF或未选CPU仍不访问SDK。
增加一个普通CPU preset，现有CMake产品/route布尔与高级裁剪继续复用，不新增重复路线selector。
缓存第一次完整解包核对build-version并保存文件SHA，命中核对完整文件清单；锁住单写入，
不把中断目录当命中，不修改外部用户SDK。C++和原SDK字节不变，原P1身份复用。

CB4是源码已证实的交付缺陷：原SDK只导出torch/c10/torch_cpu，fbgemm是非target的dependent-library
字符串，uv/libiomp5md也不在CMake target图；过去验证靠单独脚本邻置七份DLL。用户要求默认CPU
模式完成依赖，因此本构建卡只扩充已有runtime helper的文件描述，不重写package/治理或性能核心。
安装shared与static均交付同一私有SDK shim；通用helper只对consumer真实链接闭包中的owner添加文件，
避免无关目标因SDK已发现就复制整份runtime。尚未验证，不预称PASS。

Linux安装消费者补查发现具名CB4尾项：shared HYLE的`$ORIGIN`无法找到外部CPU SDK，
即使consumer的find_package成功，实际启动仍报`libtorch_cpu.so`缺失。固定有限修复为
同一owner runtime文件登记与同一个`hyle_stage_runtime_dependencies(app)`入口：Linux不复制
一整份SDK，仅为实际使用CPU SDK的consumer添加已解析SDK目录和可覆盖间接依赖的ELF RPATH。
未关联目标不改变loader路径；启动不再要求用户另写LD_LIBRARY_PATH。source/static/shared
均用小型真实FP8 Direct/Graph重复执行验证，不因此重跑原数学P1或修统计工具。
