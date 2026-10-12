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

唯一断言修复已实施，同原contract第三次运行通过原CPU/tests=ON/OFF graph-only用例，
随后在全factor all配置因private runner未传VULKAN_SDK/glslc环境失败。原三次elapsed总
272.140秒，原300剩27.860；聚合gate保持INCONCLUSIVE，未签PASS或NA。
该环境失败不能驱动产品依赖/SDK改动；后续先按实际已准备的Vulkan SDK身份补运行环境，
不得重置预算。原定向F5模块/显式SKU交付证据不由这个全factor配置失败撤销。

第三次日志见[环境失败与断言修复范围](../../../../evidence/correctness/foundation-f5-component-vulkan-environment/README.md)。

private runner环境补充已准备：实际既有C:/public/tools/hyle-vulkan/1.4.357.0、glslc与
vulkan-1.lib均存在，PATH与VULKAN_SDK沿已过SDK身份。还未重跑。全合同一次cold configure
曾用107.750秒，原剩27.860不足；恢复全聚合门需用户追加一次300秒，保全部原FAIL/time，
不改工具核心/产品或换root重取预算。等待期间继续其余F5独立门，不把整个goal标blocked。

2026-10-13入口复核：脚本首行“Configure-only”及原private runner同名scope不完整。
实际脚本还build/run Core消费者、build container_intrusive与io_transfer测试，后半段验证
Time/Container/IO/缓存/许可等共享工具合同。第三次在第626行all配置失败，不能推定后半段
已运行；脚本第32行清理整个component-contract子树，没有resume/skip/start入口。
这些源码事实保留在原public脚本，本复核不改工具、不删除历史失败或把未运行项写PASS。
本轮改过中央组件声明/PUBLIC链接边，原活动卡已预声明selector component门；不能仅因
此前误以为只配置或当前成本较重，自行缩成更小的通过条件。预算问题仍是同一原300秒
合同，剩27.860秒；此前一次追加300秒问题仍待答复，尚未启动新的复跑。

2026-10-13用户明确授权：“追加，不用再问了，类似的情况都增加时间”。同root/工具链
现追加一次300秒，总上限600秒，原272.140秒与三FAIL全部计入，首续跑剩327.860秒。
以后类似验证超时/余量不足直接追加有限时间并记每次理由/增量/累计，不再逐次询问；
这不授权重置失败、重取root预算、跳过未验项或扩产品范围。原component脚本保持原样，
只private runner记录授权/候选/SDK环境/预算后续跑，完整结果尚待。

该次完整复跑TIMEOUT328.110秒，累计600.250秒，进程树已终态，原日志虽因成功子case
被脚本捕获而为空也按真实sha封存，未写PASS。已越过原all配置及IO build，到offline
contract附近；没有产品反证。按用户standing授权再有限追加600秒，总上限1200，原
全部FAIL/TIMEOUT/elapsed继续计入。新run只增加CMake原trace观察选项来标记真实停止
case，不改public脚本/断言/产品、不换root预算；库/测试仍原并行度且本次重型独占。

最终状态：CLOSED_TOOL_FIX。追加600后的原完整脚本实际PASS256.047秒，累计856.297/1200；
已到原最终“HYLE component contracts passed”，全部旧case保留。原三FAIL/一次TIMEOUT、
SDK环境修复、唯一一行断言差分与standing时间授权在
[完整证据](../../../../evidence/correctness/foundation-f5-component-facade-correction/README.md)。
没有改全局仪器/产品图来让脚本绿；本工具任务不授权新的模块产品改动。
