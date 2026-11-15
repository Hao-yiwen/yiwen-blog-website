import React, { type ReactNode } from 'react';
import Paginator from '@theme-original/DocItem/Paginator';
import type PaginatorType from '@theme/DocItem/Paginator';
import type { WrapperProps } from '@docusaurus/types';
import Comment from '@site/src/components/Comment';

type Props = WrapperProps<typeof PaginatorType>;

/**
 * Wrap swizzle：在文档页「上一篇 / 下一篇」导航下方追加评论区。
 * 只包裹原组件、不复制其实现，Docusaurus 升级时无需同步上游代码。
 */
export default function PaginatorWrapper(props: Props): ReactNode {
    return (
        <>
            <Paginator {...props} />
            <Comment />
        </>
    );
}
