import React, { type ReactNode } from 'react';
import Paginator from '@theme-original/BlogPostPaginator';
import type PaginatorType from '@theme/BlogPostPaginator';
import type { WrapperProps } from '@docusaurus/types';
import Comment from '@site/src/components/Comment';

type Props = WrapperProps<typeof PaginatorType>;

/**
 * Wrap swizzle：在博客文章「上一篇 / 下一篇」导航下方追加评论区。
 * BlogPostPaginator 只在博客文章详情页渲染（列表页用的是 BlogListPaginator），
 * 且只要博客多于一篇就一定会渲染，所以评论区只出现在文章页。
 */
export default function PaginatorWrapper(props: Props): ReactNode {
    return (
        <>
            <Paginator {...props} />
            <Comment />
        </>
    );
}
