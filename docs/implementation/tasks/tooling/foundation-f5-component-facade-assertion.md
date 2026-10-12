# F5 component合同的graph-only façade断言反证

状态：OPEN_TOOL_FOLLOWUP，独立公共工具任务；不是产品模块施工授权，也不改F5完成定义。

原component合同同300秒预算续跑，graph_batch tests=ON归属缺口已修复并通过该配置。
随后CPU ON/tests OFF失败于脚本生成parent：只set(HYLE_COMPONENTS graph)，却要求
hyle::compute与hyle::graph两个完整产品façade同时存在。当前中央合同明确：只有全部root
factor已选才建façade；Compute含compute_state，Graph不选它。tests=ON的测试依赖恰好补了
该factor，不能据此推导tests=OFF的graph-only也应有完整Compute产品。

证据：cmake/HyleCiComponentContract.cmake的compute-route-source及CPU/tests循环；
cmake/HyleComponents.cmake的两个product root与“all root factors present”规则。
SDK普通Compute+Graph都显式选择两产品并真实通过，不能为让旧工具绿而让Graph产品顺带
取得State或重写中央façade语义。原FAIL/103.437+60.953秒保留，聚合component工具门尚未PASS。

工具有限后续应核对graph-only确有Graph产品、其Compute依赖可用但不伪造完整Compute；
若用例本来要两个完整产品，则parent应显式选compute;graph。必须独立审计/授权公共工具
改动，F5模块内不修改该脚本或checker。当前继续实际模块测试/头及原18个显式Compute+Graph
SDK，其他风险/P0/guide/红队保留；不能把工具失败写PASS或NOT_APPLICABLE。

原日志见[component失败证据](../../../../evidence/correctness/foundation-f5-component-facade-assertion/README.md)。
