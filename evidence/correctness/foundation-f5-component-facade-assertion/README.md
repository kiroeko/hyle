# F5 component façade工具断言失败

同Win/MSVC19.50/Ninja与原300秒budget：首次103.437秒缺Graph batch test fragment，已修；
第二次60.953秒，CPU/tests=ON已通过，CPU/tests=OFF的graph-only parent要求完整Compute
façade而FAIL。没有把未选择的compute_state补入Graph来躲断言。

命令、原两个terminal/FAIL/elapsed、完整log及hash见[receipt](receipt.json)/raw.zip；
产品candidate25e不变。[独立工具任务](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/tooling/foundation-f5-component-facade-assertion.md)
承接公共checker问题，F5模块不顺手改工具。聚合component gate保持未通过，不冒充PASS。
