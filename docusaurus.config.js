// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const {themes} = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

/** @type {import('@docusaurus/types').Config} */
async function createConfig() {
const math = (await import('remark-math')).default;
const katex = (await import('rehype-katex')).default;

return {
    title: 'yiwen',
    favicon: 'img/logo.png',
    url: 'https://your-docusaurus-test-site.com',
    baseUrl: '/yiwen-blog-website/',
    organizationName: 'yiwen',
    projectName: 'docusaurus',
    onBrokenLinks: 'throw',
    markdown: {
        format: 'mdx',
        mermaid: false,
    },
    i18n: {
        defaultLocale: 'zh',
        locales: ['en', 'zh'],
    },

    presets: [
        [
            'classic',
            /** @type {import('@docusaurus/preset-classic').Options} */
            ({
                docs: {
                    sidebarPath: require.resolve('./sidebars.js'),
                    editUrl:
                        'https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/',
                    remarkPlugins: [math],
                    rehypePlugins: [katex],
                    sidebarCollapsed: false,
                    showLastUpdateTime: true,
                    showLastUpdateAuthor: false,
                },
                blog: {
                    showReadingTime: true,
                    editUrl:
                        'https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/',
                    blogSidebarCount: 'ALL',
                    remarkPlugins: [math],
                    rehypePlugins: [katex],
                },
                theme: {
                    customCss: [require.resolve('./src/css/custom.css')],
                },
            }),
        ],
    ],

    plugins: [
        [
            require.resolve('@easyops-cn/docusaurus-search-local'),
            {
                // `hashed` is recommended as long-term-cache of index file is possible.
                hashed: true,
                // For Docs using Chinese, The `language` is recommended to set to:
                // ```
                language: ['en', 'zh'],
                // ```
                // When applying `zh` in language, please install `nodejieba` in your project.
            },
        ],
    ],

    stylesheets: [
        {
            href: 'https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css',
            type: 'text/css',
            integrity:
                'sha384-Xi8rHCmBmhbuyyhbI88391ZKP2dmfnOl4rT9ZfRI7mLTdk1wblIUnrIq35nqwEvC',
            crossorigin: 'anonymous',
        },
    ],

    themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            image: 'img/docusaurus-social-card.jpg',
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
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
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
