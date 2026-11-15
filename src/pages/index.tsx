import React, { type ReactNode } from 'react';
import { Redirect } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';

// 首页直接跳转到「人工智能」文档；useBaseUrl 会带上 baseUrl 和当前 locale 前缀（如 /en/）
export default function Home(): ReactNode {
    return <Redirect to={useBaseUrl('/docs/ai/')} />;
}
