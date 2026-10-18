# F5 actual Linux CUDA FI宿主UBSAN

固定product25e，actual RTX4070/GCC14.2/CUDA13.3/SM89，独立ubsan binary tree。
实际commands核对public D256与原native Direct/Graph、BatchInvocation/coordinator/
leased Graph与FI C++ adapter均有对应sanitizer flags；数学仍原成熟FI真实GPU运行。
三consumer全部PASS：四域D256 Direct/Graph、原short/long split、独立slot/旧结果/metadata
generation/update、actual prepare/run错误与suffix/unknown/pending/native retirement、精确页/
append/F4/正常放弃/open failed slot等。实际episode以完整日志为准，不虚构未跑风险。

ASan首build因原projected reader拒绝fixture调用CPU context而private CUDA-only parent漏
依赖，保原link FAIL/555.387秒；同树预算补其真实CPU路线，没有改fixture跳过该风险。
UBSan/TSan parent用内部精确factor仍真实原API/native FI/CPU拒绝用例，只少编未消费的
普通默认Provider；不改变用户产品或Provider选择，18 SDK独立证明普通三交付。

所有原失败、命令/机器/flags/hash/累计见[receipt](receipt.json)/raw.zip。同root预算继续，
无旧算子catalog或P1。GPU kernel/driver私有内存未被宿主san instrument；不能称其访问
或总字节边界已由此通过。此门不代签其他平台/facet/完整F5 closure。
