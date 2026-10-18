# CPU准确页三交付与混合core实际数学

最终产品runtime candidate42f98b94f，provider1e9不变；[收据](receipt.json)与[raw](raw.zip)保原命令、
FAIL/PASS、安装及消费者artifact hashes。Windows源五tests、WSL三san各14tests见
[风险证据](../inference-g01-cpu-page16-risk/README.md)。
Windows relocated static/shared SDK只用安装头与公开Compute/Graph targets；两dtype/
三D/非零double/64复用＋64演化/不同topology两槽保旧output，及原Direct/Graph双region
ragged/resident三consumer均PASS。失败shared1保prefix与原52.407s，不覆盖；SDK2用
普通public facade consumer补真实CPU闭包，四jobs，没有全库build或产品/API修改。
MacBook Air实际M4 Mac16,12/arm64/AppleClang21/macOS27.0.1，CPU+Apple shared源树
原mixed456 CPU数学14tests PASS；实际编译ops含页地址helper，原TU/hash/命令保留。
本次只运行CPU数学，不代签GPU或联合分页。原duplicate GTest配置FAIL也保留。
源、静态、动态及宿主风险成立，原P1 DIAGNOSTIC保留，修正后P1待验，private上界UNKNOWN；仍为候选，
不代签整个G01/G02。继续[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-cpu-paged-resident.md)。
