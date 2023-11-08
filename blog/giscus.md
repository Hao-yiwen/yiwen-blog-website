---
sidebar_position: 1
---

# Docusaurus 添加评论

## 前置条件

1. `Docusaurus`: `2.4.1`
2. `@giscus/react`: `2.3.0`

## 开通 giscus

[giscus官网](https://giscus.app/)
:::note

1. 公共仓库
2. 开启`discussions`
3. 下载`giscus app`

:::

## 封装 Comment

1. 安装 `@giscus/react`

```bash
npm i @giscus/react
```

2. 封装 Comment

```tsx title="Comment.tsx"
import React from 'react';
import Giscus from '@giscus/react';

export const Comment = () => {
    return (
        <div style={{ paddingTop: 50 }}>
            <Giscus
                id="comments"
                // highlight-warn-start
                // 这部分填写你自己的
                repo="3Alan/site"
                repoId=""
                category="Announcements"
                categoryId=""
                // highlight-warn-end
                mapping="title"
                strict="1"
                term="Welcome to @giscus/react component!"
                reactionsEnabled="1"
                emitMetadata="0"
                inputPosition="bottom"
                theme="dark_dimmed"
                lang="zh-CN"
                loading="lazy"
            />
        </div>
    );
};

export default Comment;
```

## Swizzling Docusaurus

### Swizzling 文档页面

```bash
yarn run swizzle @docusaurus/theme-classic DocItem/Layout -- --eject --typescript
```

```tsx title="src/theme/DocItem/Layout/index.tsx"
import React from 'react';
import clsx from 'clsx';
import { useWindowSize } from '@docusaurus/theme-common';
// @ts-ignore
import { useDoc } from '@docusaurus/theme-common/internal';
import DocItemPaginator from '@theme/DocItem/Paginator';
import DocVersionBanner from '@theme/DocVersionBanner';
import DocVersionBadge from '@theme/DocVersionBadge';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemTOCMobile from '@theme/DocItem/TOC/Mobile';
import DocItemTOCDesktop from '@theme/DocItem/TOC/Desktop';
import DocItemContent from '@theme/DocItem/Content';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import type { Props } from '@theme/DocItem/Layout';

import styles from './styles.module.css';
// highlight-add-line
import Comment from '../../../components/comment';
/**
 * Decide if the toc should be rendered, on mobile or desktop viewports
 */
function useDocTOC() {
    const { frontMatter, toc } = useDoc();
    const windowSize = useWindowSize();

    const hidden = frontMatter.hide_table_of_contents;
    const canRender = !hidden && toc.length > 0;

    const mobile = canRender ? <DocItemTOCMobile /> : undefined;

    const desktop =
        canRender && (windowSize === 'desktop' || windowSize === 'ssr') ? (
            <DocItemTOCDesktop />
        ) : undefined;

    return {
        hidden,
        mobile,
        desktop,
    };
}

export default function DocItemLayout({ children }: Props): JSX.Element {
    const docTOC = useDocTOC();
    return (
        <div className="row">
            <div className={clsx('col', !docTOC.hidden && styles.docItemCol)}>
                <DocVersionBanner />
                <div className={styles.docItemContainer}>
                    <article>
                        <DocBreadcrumbs />
                        <DocVersionBadge />
                        {docTOC.mobile}
                        <DocItemContent>{children}</DocItemContent>
                        <DocItemFooter />
                    </article>
                    <DocItemPaginator />
                </div>
                // highlight-add-line
                <Comment />
            </div>
            {docTOC.desktop && (
                <div className="col col--3">{docTOC.desktop}</div>
            )}
        </div>
    );
}
```

### Swizzling 博客页面

```bash
yarn run swizzle @docusaurus/theme-classic BlogPostPage -- --eject --typescript
```

```tsx title="src/theme/BlogPostPage/index.tsx"
import React, { type ReactNode } from 'react';
import clsx from 'clsx';
import {
    HtmlClassNameProvider,
    ThemeClassNames,
} from '@docusaurus/theme-common';

import {
    BlogPostProvider,
    useBlogPost,
    // @ts-ignore
} from '@docusaurus/theme-common/internal';
import BlogLayout from '@theme/BlogLayout';
import BlogPostItem from '@theme/BlogPostItem';
import BlogPostPaginator from '@theme/BlogPostPaginator';
import BlogPostPageMetadata from '@theme/BlogPostPage/Metadata';
import TOC from '@theme/TOC';
import type { Props } from '@theme/BlogPostPage';
import type { BlogSidebar } from '@docusaurus/plugin-content-blog';
// highlight-add-line
import Comment from '../../components/comment';

function BlogPostPageContent({
    sidebar,
    children,
}: {
    sidebar: BlogSidebar;
    children: ReactNode;
}): JSX.Element {
    const { metadata, toc } = useBlogPost();
    const { nextItem, prevItem, frontMatter } = metadata;
    const {
        hide_table_of_contents: hideTableOfContents,
        toc_min_heading_level: tocMinHeadingLevel,
        toc_max_heading_level: tocMaxHeadingLevel,
    } = frontMatter;
    return (
        <BlogLayout
            sidebar={sidebar}
            toc={
                !hideTableOfContents && toc.length > 0 ? (
                    <TOC
                        toc={toc}
                        minHeadingLevel={tocMinHeadingLevel}
                        maxHeadingLevel={tocMaxHeadingLevel}
                    />
                ) : undefined
            }
        >
            <BlogPostItem>{children}</BlogPostItem>
            {(nextItem || prevItem) && (
                <BlogPostPaginator nextItem={nextItem} prevItem={prevItem} />
            )}
            // highlight-add-line
            <Comment />
        </BlogLayout>
    );
}

export default function BlogPostPage(props: Props): JSX.Element {
    const BlogPostContent = props.content;
    return (
        <BlogPostProvider content={props.content} isBlogPostPage>
            <HtmlClassNameProvider
                className={clsx(
                    ThemeClassNames.wrapper.blogPages,
                    ThemeClassNames.page.blogPostPage
                )}
            >
                <BlogPostPageMetadata />
                <BlogPostPageContent sidebar={props.sidebar}>
                    <BlogPostContent />
                </BlogPostPageContent>
            </HtmlClassNameProvider>
        </BlogPostProvider>
    );
}
```
