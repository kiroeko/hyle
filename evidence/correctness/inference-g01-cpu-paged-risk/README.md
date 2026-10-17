# CPU准确页宿主风险

最终产品candidate1e9cdfd45，原CPU-only ffa4数学与认证页地址patch；[收据](receipt.json)
及[原始记录](raw.zip)保存每次命令、FAIL/PASS、累计与实际TU sanitizer flags。
ASan+LSan、UBSan、TSan各14tests PASS：三D/F16/BF16独立double、64复用＋64演化，
原resident/ragged数学、paged逐分配失败/小预算/已接纳失败不发布output及真实页退休。
旧fixture遗漏sentinel与ASan旧mtime造成的STALE_BUILD分开保留；重新实际编译后通过，
不改变数值容差/错误合同。source风险补验和所有配置失败保原记录。
此记录不代签混合core/三交付/P1或整个G01；继续[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-cpu-paged-resident.md)。
