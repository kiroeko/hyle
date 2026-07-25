# 许可与第三方代码边界

## 当前决策

HYLE 在正式发布前暂时闭源。HYLE 原创材料采用仓库根 [`LICENSE`](../../LICENSE) 中的
proprietary notice：保留全部权利，仅按权利人的单独书面授权使用。这不是自定义开源许可证，
也不因某人能读取仓库就自动授予使用、修改或分发权。

原创源文件需要短标识时使用：

```text
SPDX-License-Identifier: LicenseRef-HYLE-Proprietary
```

`LicenseRef-` 是 SPDX 为不在 SPDX License List 中的自定义条款保留的形式；对应全文就是根
`LICENSE`。不得再写不可机读、含义不确定的 `SPDX: same as HYLE`。

`Copyright (c) 2026 the HYLE copyright holders` 有意不虚构尚未确定的公司主体。准备对外签约或
分发前，应由项目所有者/法律顾问把它替换成准确的个人或法律实体，并确认提交者的权利归属。

## 原创与第三方的边界

proprietary notice 只覆盖 HYLE 原创代码、文档和构建资产，不能改变 vendored、ported、fetched
或系统依赖的许可证。权威清单是根 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)：

- `third_party/` 内实现保留原 LICENSE 与源码版权头；
- 从 Highway/xsimd/JPEG XL 迁入 `src/simd` 的代码继续受原许可证约束；
- HYLE 可对周围原创代码和自己的修改采用 proprietary 条款，但不得删除第三方条件；
- fetched/外部 backend 只有其二进制、传递依赖和 license payload 都进入 export graph 后，
  才能宣称 install package 可迁移。

## 构建与分发合同

安装包必须包含 HYLE proprietary notice、第三方总清单，以及实际选中实现所需的许可证文本。
头文件安装与二进制闭包不能成为遗漏 notice 的借口；BSD/MIT 等许可证对二进制再分发通常仍有
notice 条件。构建系统按 target 图选择 payload，缺少许可证与缺少 DLL/archive 同样属于不可发布。

内部开发构建可以引用本机 checkout，但任何交付给另一主体的源码包、SDK、安装前缀或二进制
制品都必须保留这些文件。外部 OpenSSL/CUDA/驱动等由部署环境提供时，不复制其文件，但仍须遵守
供应商条款。

## 将来开源

将来选择 MIT/Apache-2.0 等开源许可证时，只能重新许可 HYLE 权利人拥有的部分；第三方部分继续
沿用原条款。切换应以明确提交更新根 `LICENSE`、SPDX 标识、package payload 和用户文档，不能用
一句“现在开源了”覆盖过去版本或第三方来源。

本页是仓库工程政策，不替代针对实际权利主体、雇佣/贡献关系和分发地区的法律审查。
