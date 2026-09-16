# CUDA host 插桩参数预检

baseline：`0ca1cd0da9e476be44419a50b5ddfdc34544e293`。
本记录仅证明 [R4 C1/C5 尾项](../../../docs/implementation/tasks/compute/cuda-v1-four-route-foundation.md)
中的 nvcc 参数问题，**不是实际 sanitizer 或 CUDA 路线 PASS**。

- [原参数](nvcc-sanitizer-dryrun.log)：nvcc 13.3 拒绝 `-fsanitize=address`。
- [加入转发参数](nvcc-sanitizer-forwarded-dryrun.log)：dry-run 中的 host compile 和最终 g++ link
  均收到 sanitizer flags。没有生成/运行插桩产物，也没有向 device code 赋予 host sanitizer 覆盖。
- 改动仅在 Marlin、HYLE CUDA owning target 的非 MSVC sanitizer 构建启用
  `--forward-unknown-to-host-compiler`；普通 shipping 构建的参数保持不变。

实际普通回归、cuBLAS 分配失败检查及 WSL ASan/UBSan/TSan 仍待执行。系统 CUDA/第三方 device
kernel 的内部空间不由这些 host 检查完整覆盖。可读日志仅规范化换行及行尾空白；原始字节保存在 [raw archive](raw-dryrun.zip)，
源文件身份及限制在 [preflight](preflight.json)，文件哈希见 [manifest](sha256.json)。
