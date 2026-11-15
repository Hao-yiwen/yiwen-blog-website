import React, { type ReactNode } from 'react';
import Giscus from '@giscus/react';
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * Giscus 评论区（基于 GitHub Discussions）。
 * - mapping="pathname"：按页面路径关联 Discussion，修改文档路径会导致旧评论无法关联
 * - 主题跟随站点的明暗模式切换，语言跟随当前 locale
 */
export default function Comment(): ReactNode {
    const { colorMode } = useColorMode();
    const {
        i18n: { currentLocale },
    } = useDocusaurusContext();

    return (
        <div style={{ paddingTop: 50 }}>
            <Giscus
                id="comments"
                repo="Hao-yiwen/yiwen-blog-website"
                repoId="R_kgDOKCp9Gw"
                category="Announcements"
                categoryId="DIC_kwDOKCp9G84CZP8J"
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
