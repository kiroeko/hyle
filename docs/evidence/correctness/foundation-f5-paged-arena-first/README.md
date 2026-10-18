# F5 逐页权限与append/F4首批

AUDITED有限实现checkpoint，未签candidate/全F5。common Win MSVC19.50/shipping/static/
WERROR，实际32项和八头独立TU；WSL RTX4070/GCC14.2/NVCC13.3原FlashInfer native
Direct另有实际两dtype三episode共六oracle，原Direct四域三episode12oracle保持通过。
只新batch/实际owner/原CPU F2 TEST_DEP/真实CUDA consumer闭包，无旧算子或全F4回放。

反证：whole arena ReadLease的subview无法独立写free page；原F1不同block则可分别read/
write，却不能由单page lease描述whole数学arena。PagedArena按物理页冷分配原F1 blocks，
read快照只保同layout/schema与exact referenced page guards，full native capacity视图仅
trusted private adapter可见，不制造whole-arena read permission或raw writable alias。
原Typed metadata/Native kernel保持，Direct submit原LeaseUseBatch持所有实际页到Completion
实际退休。新资源容量8+2*max_page_refs仅opt-in batch冷元数据；普通对象字段不变。

完整共同风险包括foreign/schema/range/不完整/多余/重复、byte cap/metadata OOM、old
reader/native use、unsubmitted/consumed abandon、close/late guard，以及实际CPU F2页copy/
append→F4替换dirty page与layout→expected-base publish；clean page继续共享。F4 backing
ledger与arena不能各持readonly keeper后互等unique：arena只记弱block/generation，原固定
LeasePool在真实最后reader/native use释放并由F4 collect后回收range，再取writer；没有
猜测读者数或绕过F1权限。旧backing保独立schema/accounting owner，末次控制释放才冷
detach pool root，不形成反持Pool的owner环。quota在最后native-use guard释放后归零。

actual CUDA F16/BF16各旧布局3有效KV行→新布局4行→旧布局重读，独立uniform SDPA
oracle0.25/0.375/0.25全部通过。保旧F4版本/旧page frame时，新physical page2可写；原F2
两phase显式依赖先clone K/V dirty page、再append原device incoming rows。实际clone1024+
append512bytes，整个K/V arena4096bytes未复制。F4目录完整列K0/K1/V0/V1/layout，
dirty members与同版本layout在成功ready后替换/publish，clean page原block保持共享。
同query/KV indptr，原native update只更新metadata/version并证明plan reuse，无run规划。

原新增fixture以whole view读1byte尺寸FAIL、缺独立backing owner的cold detach FATAL、
MSVC ResultError/Status ternary编译FAIL保留；初版CUDA append fixture未声明clone→append
边，错误退出时resource先于accepted work释放的FATAL覆盖了原拒绝原因；该原log保留，
不猜测被覆盖的具体Status。修正显式phase dependency及
测试原Context drain先于resource销毁，无放宽F1/F2或改公共工具。所有原失败、input历史、
精确命令、hash与累计budget在raw archives；根/tree/cache复用，库j4/j2/j1、test1，机器
串行，未因新增文件/提交重置。原first/native reference证据不改写。

仍待Graph/private projected binding及CPU/Apple/Vulkan reference同页权限、完整batch
组合控制器、ragged完整append旅程、producer/error/unknown/alias/OOM/cancel其余风险，
三交付/san/P0/guide/终审。不能由本六oracle代签四路线paged消费或单机吞吐/capture。
