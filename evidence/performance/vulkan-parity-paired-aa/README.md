# Vulkan parity：同轮A/A测量恢复资格

快照110d59822，当前Windows RTX4070/NVIDIA617.14、Vulkan1.4.351、ggml456172ec。
consumer链接当前canonical producer，避免混用旧十格式SDK和新fixture头。
产品/API/公共统计核无改动，仅bench接入和本模块paired reader。

[receipt](receipt.json)保存唯一8process：Graph/Graph、Direct/Direct各双执行/双分配顺序。
每process两fragment×两arm×24round，每round6调用覆盖原2/3槽完整cycle；输出/数值/
native终态和warm资源稳定均实际验证。actual CPU affinity/priority、native环境、binary/
reader/candidate及raw SHA已记录，不要求绝对空闲GPU。

事前primary固定为相同round两arm时间比的24点中位数，再用独立process/factor裁决。
原5%因素/10%单process门保持，结果PASS。原表marginal op/s比仍是diagnostic，历史
FAIL不改写。reader严格拒绝缺/重复round、不同cycle、错iterations、非法时间/subject；
旧数据只用于reader反证，不导入新gate。

只满足旧BLOCKED的具名测量恢复条件；不签n15/P1、Graph/Direct等价或Provider WIN。
Graph owning result与Direct Recorder caller buffer存在责任差异，正式A/B必须再审
same-contract，不能由此AA PASS免除。没有更改margin、追加样本或后验追认。
入口与预算见[独立资格卡](../../../docs/implementation/tasks/compute/vulkan-parity-paired-aa-qualification.md)。
raw完整归档；DLL从EXE邻置加载，没有缺DLL弹窗/UAC/全局系统修改。
