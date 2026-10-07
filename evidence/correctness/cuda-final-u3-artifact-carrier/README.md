# CUDA U3 N1 Windows 接线批次

当前任务：[原量化准备载体](../../../docs/implementation/tasks/compute/cuda-final-u3-artifact-carrier.md)。
产品候选 `0cd39365339e75a2d0bdce15a8fa065a8f5408ba`；Windows canonical，RTX4070、
MSVC19.50.35728、CUDA Toolkit13.3.73。原 carrier 是 LibTorch2.8.0+cu129 CUDA leaf＋
原 locked LibTorch2.8.0 CPU host dispatch common；CPU common DLL 不替换，GPU 工作没有 CPU fallback。

## 已实现和已验证

- 普通 CUDA 默认 Direct／Graph 已组装成熟 artifact Provider；原 API、profile／实际 dtype／packing 保持。
- 原 NF4/nested、AWQ permutation／zeros／scale、CT signed4／scale 使用 GPU tensor 程序，
  必要 shifts／order 冷准备复用；库私有分配／managed replay 和非外部 capture 如实声明。
- 固定共同目录的 6 nested NF4、7 AWQ、5 CT，共18个原内宽／完整矩阵合同，两入口各18测试，
  包含 GPU 冷 decode 输出保留为独立常量后，送入成熟 projection、改变激活及重复执行。
- 四种格式×F16/BF16/F32 的12项小型精度／生命周期测试：变化原 payload、两个独立 slot，
  已接受工作在 Context close 后仍能完成和读取结果；不许可关闭后提交新工作。
- 默认 FFN／Provider roster 回归1项；合计 **49 GTests**；**5 CTests**（含三个公开头检查）通过，
  无 fixture skip。原 XML 的两个 native test stdout分别声明19和30测试；进程返回0，测试源码无 skip。
- 默认 DLL staging 已将两套依赖的不同叶子放在同一个 exe 目录，两个普通 consumer进程实际加载并运行。
  此处 producer 是 shared／mixed／testsON／installOFF，不是 shipping static/shared SDK 证据。
- 全 Compute+Graph 非测试实现 **60,555行／253文件**；统一脚本自检PASS。测试、bench与第三方不计入。
- 固定输入工具13测试与selector工具59测试通过；只增加三项vendor→owner登记，不改selector核心。
- 原隔离SDK probe的26个GPU DLL复制件经受管原SDK SHA逐项匹配后删除，回收 **4.03GiB**；
  非递归清理，仅删本轮归档探针的拥有者复制件，未动产品SDK／producer／模型或CPU硬链接。

## 原失败和纠正

全部原 configure／build／运行记录保留在 [批次归档](windows-batch.zip)，不删除失败来签绿：

1. CPU已导入三个global Torch targets、其余辅助targets仍局部；GPU owning CMake重复find导致导出集部分定义。
   改为复用CPU已审SDK；CUDA-only首次发现时完整提升原五target，不改CPU原resolver／数学。
2. wheel原许可证位于dist-info的LICENSE／NOTICE，未定义licenses子目录；固定原选择，并在同identity
   owned stage中完整重解包，不递归删除或接受半成品。vendored树排序须portable字符串大小写敏感，
   不能用WindowsPath的大小写折叠比较；锁已按原CMake规则纠正，三个来源正文不变。
3. 原SDK DeviceIndex是窄整数；冷阶段检查真实ordinal容量、显式转换，并复用backend已有快照而非再查询driver。
4. 新Provider feature经过private route边未向普通consumer传播；route公开实际组装的literal capability。
   测试比较Provider ID不用把预处理条件嵌入GTest宏参数，保留宏关闭时独立公开头检查。
5. 原BLAS handle初始化先申请工作区；必须先调用原`lazyInitDevice(CUDA)`完成allocator初始化。
   原失败在冷prepare，不是模型数值输赢；修正不改变数学／精度门。
6. 新测试初版在逻辑Context关闭后继续submit，违反当前close admission合同；现检查已接受工作保活。
   旧bad_expected_access经明确Status断言定位为`context is closed`，产品没有放宽关闭规则。

## 尚未签署

Linux ABI／carrier／CUDA leaf依赖、CUDA-only与mixed的source/static/shared/offline/搬移、
剩余OOM／严格资源／capture失败项、所有其它格式／完整模型region、U4性能／best-of尚未完成。
Linux官方wheel只读取了metadata并固定原NVIDIA发布依赖身份；没有下载运行，不签Linux PASS。
native Python／隔离C++探针只证明接缝，不算新的HYLE覆盖或性能赢家；本记录不能签N1或整个CUDA CLOSED。

[机器摘要](windows-receipt.json)与归档保存精确candidate、native及原失败、测试／XML、环境、
selector计划／skipped理由、原官方archive metadata、宏只读扫描和LOC。
selector因中央图／lock seam选择的广泛交付仍待执行；批次只证明上列实际门，不能把READY写成PASS。
