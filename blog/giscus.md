---
title: Docusaurus 集成 Giscus 评论系统
authors: [yiwen]
tags: [Docusaurus, Giscus, React]
date: 2023-09-12
---

记录为 Docusaurus 3 博客与文档站点接入基于 GitHub Discussions 的 Giscus 评论系统的完整实现方案。

<!-- truncate -->

## 前置条件

1. 仓库为 GitHub **公开仓库（Public）**
2. 在仓库 `Settings -> General -> Features` 中开启 **Discussions**
3. 安装 [giscus GitHub App](https://github.com/apps/giscus) 并授权对应仓库
4. 前往 [giscus 官网](https://giscus.app/zh-CN) 生成 `repoId` 与 `categoryId`

## 封装 Comment 组件

1. 安装依赖：

```bash
pnpm add @giscus/react
```

2. 创建 `src/components/Comment.tsx`，自动跟随站点的明暗主题与语言切换：

```tsx title="src/components/Comment.tsx"
import React, { type ReactNode } from 'react';
import Giscus from '@giscus/react';
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Comment(): ReactNode {
    const { colorMode } = useColorMode();
    const {
        i18n: { currentLocale },
    } = useDocusaurusContext();

    return (
        <div style={{ paddingTop: 50 }}>
            <Giscus
                id="comments"
                repo="your-org/your-repo"
                repoId="YOUR_REPO_ID"
                category="Announcements"
                categoryId="YOUR_CATEGORY_ID"
                mapping="pathname"
                strict="0"
                reactionsEnabled="1"
                emitMetadata="0"
                inputPosition="bottom"
                theme={colorMode === 'dark' ? 'dark' : 'light'}
                lang={currentLocale === 'zh' ? 'zh-CN' : 'en'}
                loading="lazy"
            />
        </div>
    );
}
```

## 使用 Wrap Swizzle 挂载评论区

相比 `--eject` 完整复制上游源码，推荐使用 `--wrap` 方式包裹翻页导航组件，这样未来升级 Docusaurus 时无需手动同步内部代码：

### 1. 文档页挂载

```bash
pnpm swizzle @docusaurus/theme-classic DocItem/Paginator --wrap --typescript
```

```tsx title="src/theme/DocItem/Paginator/index.tsx"
import React, { type ReactNode } from 'react';
import Paginator from '@theme-original/DocItem/Paginator';
import type PaginatorType from '@theme/DocItem/Paginator';
import type { WrapperProps } from '@docusaurus/types';
import Comment from '@site/src/components/Comment';

type Props = WrapperProps<typeof PaginatorType>;

export default function PaginatorWrapper(props: Props): ReactNode {
    return (
        <>
            <Paginator {...props} />
            <Comment />
        </>
    );
}
```

### 2. 博客详情页挂载

```bash
pnpm swizzle @docusaurus/theme-classic BlogPostPaginator --wrap --typescript
```

```tsx title="src/theme/BlogPostPaginator/index.tsx"
import React, { type ReactNode } from 'react';
import Paginator from '@theme-original/BlogPostPaginator';
import type PaginatorType from '@theme/BlogPostPaginator';
import type { WrapperProps } from '@docusaurus/types';
import Comment from '@site/src/components/Comment';

type Props = WrapperProps<typeof PaginatorType>;

export default function PaginatorWrapper(props: Props): ReactNode {
    return (
        <>
            <Paginator {...props} />
            <Comment />
        </>
    );
}
```
