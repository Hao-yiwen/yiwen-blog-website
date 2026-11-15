---
title: Git Rebase 原理与分支合并实践
authors: [yiwen]
tags: [Git, 工作流]
date: 2024-01-09
---

在多人协作开发中，从 `master` 拉出特性分支开发完成后，主分支往往已有其他人的新提交。频繁使用 `git merge` 会产生大量分叉的 Merge Commit，影响提交历史的可读性与问题回溯。本文梳理使用 `git rebase` 保持线性提交历史的实践。

<!-- truncate -->

## 基本用法

```bash
# 1. 拉取远程最新主分支信息
git fetch origin master

# 2. 将当前特性分支变基到最新 origin/master 之上
git rebase origin/master
```

## Git Rebase 核心机制

1. **工作方式**：先找到当前分支与目标分支的共同祖先（Common Ancestor），将当前分支上的独有提交暂存起来，再将基线移动到目标分支最新提交，最后依次重新应用这些提交。
2. **提交历史**：生成一条干净、线性的提交记录。由于每个提交的父节点（Parent Hash）发生了变化，变基后的提交会生成全新的 Commit Hash。
3. **冲突处理**：遇到冲突时需按提交顺序逐个解决，而 `merge` 则是在最终合并节点一次性解决冲突。

## Merge 与 Rebase 对比

| 维度 | `git merge` | `git rebase` |
| :--- | :--- | :--- |
| **历史形态** | 保留完整分叉与合并节点 | 线性整洁，无多余合并提交 |
| **Commit Hash** | 保持原哈希值不变 | 重写变基提交的哈希值 |
| **最佳时机** | 将完成的长期特性分支合入主干 | 开发过程中同步主干最新进展、整理本地提交 |

## 处理 Rebase 冲突

当特性分支修改了与 `master` 最新提交相同的代码区域时，Git 会暂停变基流程：

```bash
# 1. 手动编辑冲突文件并保存后，标记为已解决
git add <conflict-file>

# 2. 继续执行变基
git rebase --continue

# 若想放弃本次变基并恢复原状
git rebase --abort
```

:::warning 黄金法则
**永远不要对多人共享的公共分支（如 `master` / `main`）执行 `rebase`！** 重写已推送到公共分支的历史会导致团队协作者的本地历史产生严重冲突。
:::
