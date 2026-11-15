// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const { themes } = require('prism-react-renderer');

const EDIT_URL = 'https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/';

/** @returns {Promise<import('@docusaurus/types').Config>} */
async function createConfig() {
    // remark-math / rehype-katex 是纯 ESM 包，只能动态 import
    const math = (await import('remark-math')).default;
    const katex = (await import('rehype-katex')).default;

    return {
        title: 'yiwen',
        favicon: 'img/logo.png',
        url: 'https://hao-yiwen.github.io',
        baseUrl: '/yiwen-blog-website/',
        organizationName: 'Hao-yiwen',
        projectName: 'yiwen-blog-website',
        onBrokenLinks: 'throw',

        future: {
            // Rspack + SWC + Lightning CSS 构建，显著加快构建速度
            faster: true,
            v4: {
                // faster 中的 ssgWorkerThreads 依赖此开关
                removeLegacyPostBuildHeadAttribute: true,
            },
        },

        markdown: {
            format: 'mdx',
            mermaid: true,
        },
        themes: ['@docusaurus/theme-mermaid'],
        i18n: {
            defaultLocale: 'zh',
            locales: ['zh', 'en'],
        },

        presets: [
            [
                'classic',
                /** @type {import('@docusaurus/preset-classic').Options} */
                ({
                    docs: {
                        sidebarPath: require.resolve('./sidebars.js'),
                        editUrl: EDIT_URL,
                        remarkPlugins: [math],
                        rehypePlugins: [katex],
                        sidebarCollapsed: false,
                        showLastUpdateTime: true,
                        showLastUpdateAuthor: false,
                    },
                    blog: {
                        showReadingTime: true,
                        editUrl: EDIT_URL,
                        blogSidebarCount: 'ALL',
                        remarkPlugins: [math],
                        rehypePlugins: [katex],
                    },
                    theme: {
                        // KaTeX 样式与字体随站点一起打包，不依赖外部 CDN
                        customCss: [
                            require.resolve('katex/dist/katex.min.css'),
                            require.resolve('./src/css/custom.css'),
                        ],
                    },
                }),
            ],
        ],

        plugins: [
            [
                require.resolve('@easyops-cn/docusaurus-search-local'),
                {
                    // 索引文件名带 hash，便于长期缓存
                    hashed: true,
                    language: ['en', 'zh'],
                },
            ],
        ],

        themeConfig:
            /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
            ({
                image: 'img/logo.png',
                colorMode: {
                    respectPrefersColorScheme: true,
                },
                navbar: {
                    title: 'Yiwen',
                    logo: {
                        alt: "yiwen's blog",
                        src: 'img/logo.png',
                    },
                    items: [
                        {
                            type: 'docSidebar',
                            sidebarId: 'aiSidebar',
                            position: 'left',
                            label: '人工智能',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'backendSidebar',
                            position: 'left',
                            label: '服务端开发',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'nativeSidebar',
                            position: 'left',
                            label: 'Native开发',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'webSidebar',
                            position: 'left',
                            label: 'Web开发',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'csSidebar',
                            position: 'left',
                            label: '计算机基础',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'mathSidebar',
                            position: 'left',
                            label: '数学',
                        },
                        {
                            type: 'docSidebar',
                            sidebarId: 'whelkSidebar',
                            position: 'left',
                            label: '抗痘',
                        },
                        { to: '/blog', label: '博客', position: 'left' },
                        {
                            type: 'localeDropdown',
                            position: 'right',
                        },
                        {
                            href: 'https://github.com/Hao-yiwen',
                            label: 'GitHub',
                            position: 'right',
                        },
                    ],
                },
                prism: {
                    theme: themes.github,
                    darkTheme: themes.dracula,
                    additionalLanguages: [
                        'swift',
                        'java',
                        'bash',
                        'dart',
                        'go',
                        'sql',
                        'objectivec',
                        'cpp',
                        'ruby',
                        'kotlin',
                    ],
                },
            }),
    };
}

module.exports = createConfig();
