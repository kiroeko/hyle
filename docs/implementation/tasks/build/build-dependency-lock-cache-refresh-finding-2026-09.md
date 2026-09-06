# Build finding：依赖锁升级后旧 configure cache 可保留旧自动 source override

状态：`FINDING`（2026-09-06）。本页只记录可复现事实，不授权在 SIMD v6 内修改公共构建系统。
Build/package 的当前产品状态仍只以
[`module-status.json`](../../module-status.json) 为准。

## 复现事实

在 SIMD v6 把 Highway lock 从 `f503c485...` 升到 `18e449b6...` 后，对一个此前已配置的 build tree
重新执行 CMake。共享 dependency resolver 已成功下载并校验新 identity，但该 build tree 的
`CMakeCache.txt` 仍保存旧的自动
`FETCHCONTENT_SOURCE_DIR_HIGHWAY=<...highway-f503...>`，因此 configure 继续选择旧 source。
执行官方 CMake cache unset：

```text
cmake -S <source> -B <build> -UFETCHCONTENT_SOURCE_DIR_HIGHWAY [...]
```

后，同一 build tree 才解析到新锁定 source。fresh build 不受影响。

## 产品影响与边界

- 用户显式提供 `FETCHCONTENT_SOURCE_DIR_HIGHWAY` 时，它本来就应稳定优先；不能粗暴清除所有该变量。
- 自动 resolver 写入的旧值和用户显式 override 当前共用同一个 cache key，重新 configure 时无法区分
  provenance，可能让依赖锁升级没有进入旧 build tree。
- 这不是 SIMD 算法或性能失败；SIMD v6 的资格化 build 使用 fresh tree，且 artifact/build contract
  会锁定实际 source identity。
- 本记录不要求在模块轮内扩建性能 runner，也不授权顺手重开 Build/package。

后续若 Build/package 按此事实重开，应设计并验证“显式用户 override 保持优先，自动解析值随中央 lock
identity 更新”的单一 authority；至少覆盖同一 build tree 的 lock A -> lock B 回放、strict offline、
source/static/shared 和 shared-cache 并发合同。
