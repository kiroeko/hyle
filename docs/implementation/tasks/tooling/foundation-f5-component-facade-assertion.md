# F5 component合同的graph-only façade断言反证

状态：AUDITED_TOOL_FIX，独立公共配置测试修复；不是产品模块施工授权，也不改F5完成定义。

原component合同同300秒预算续跑，graph_batch tests=ON归属缺口已修复并通过该配置。
随后CPU ON/tests OFF失败于脚本生成parent：只set(HYLE_COMPONENTS graph)，却要求
hyle::compute与hyle::graph两个完整产品façade同时存在。当前中央合同明确：只有全部root
factor已选才建façade；Compute含compute_state，Graph不选它。tests=ON的测试依赖恰好补了
该factor，不能据此推导tests=OFF的graph-only也应有完整Compute产品。

证据：cmake/HyleCiComponentContract.cmake的compute-route-source及CPU/tests循环；
cmake/HyleComponents.cmake的两个product root与“all root factors present”规则。
SDK普通Compute+Graph都显式选择两产品并真实通过，不能为让旧工具绿而让Graph产品顺带
取得State或重写中央façade语义。原FAIL/103.437+60.953秒保留，聚合component工具门尚未PASS。

源码核对已固定唯一修复：graph-only parent只断言完整Graph façade，后面的实际CPU依赖
factor检查不变；不修改产品选择、façade规则、closure checker、selector或仪器核心。
用户活动goal要求真实package/component验收，这个独立且可逆的断言修复属于完成该验收
所必需的工作；没有新增工具能力或无界扩建，不从本记录推导新的产品实现授权。

| FIX | 改动 | 验收与预算 |
|---|---|---|
| GRAPH-ONLY-FACADE-ASSERT | HyleCiComponentContract的CPU graph-only用例把required product清单从compute/graph改成graph；保全部原factor/缺路线/其他断言 | 同public合同一次复跑；原两FAIL/164.390秒仍计入原300总budget，不能改root重取预算 |

当前继续实际模块测试/头及原18个显式Compute+Graph SDK，其他风险/P0/guide/红队保留；
不能把工具失败写PASS或NOT_APPLICABLE。产品candidate25e不受这个验证脚本修复影响。

原日志见[component失败证据](../../../../evidence/correctness/foundation-f5-component-facade-assertion/README.md)。
