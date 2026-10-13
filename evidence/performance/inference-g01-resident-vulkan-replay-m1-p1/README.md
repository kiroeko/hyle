# G01 Vulkan 原生记录修正后的合格P1

candidate `a40003f7edb54a3da3113d0cd401b46211e0b0f0`；同原合同canonical SHA
`4ae724d684be30814e2f28d7d933f771b7e03dda56b86cb51464cef706031720`，同Air、
原始shader、完整submit/wait/owner返回、精度与输入。独立clean Git checkout和fresh3
tree，不更改[前轮FAIL](../inference-g01-resident-vulkan-m1-p1/README.md)的任何字节。
[receipt](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、
[raw.zip](raw.zip)与[verify收据](verify-receipt.json)保留实际证据。

16组A/A、八strata×15产品process和30 controls完成，资格PASS、verify退出0，实际
1224.37s，performance仍FAIL。decode比0.719–0.727：约27–28%吞吐损失，仍超原10%
预算；prefill比0.962–0.968，5%非劣PASS；D64留出比0.947–0.961，非劣UNRESOLVED。
三个点全部逐strata CI上界<1，方向LOSS，不把band PARITY或UNRESOLVED改写为方向平手。

新源码定位：外层managed adapter对原strided capacity输入依然canonicalize并加入
warm全KV Copy。前一候选的数学/生命周期PASS不证明零拷贝消费。该真实费用缺口
交由compute_route_vulkan独立窄卡修正，不扩数学、公共工具或改变比较阈值。
本证据只签实际Air的三个片段，非全模型token/s、Formal或所有G01域。
