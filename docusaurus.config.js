// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const { themes } = require('prism-react-renderer');

const EDIT_URL = 'https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/';

/**
 * 历史路径重定向表（整理重命名与合并后的旧 URL -> 新 URL）
 * @type {Array<{from: string | string[], to: string}>}
 */
const REDIRECTS = [
    // 首页与模块入口兜底重定向
    { from: ['/docs/ai', '/docs/ai/training/intro'], to: '/docs/ai/intro' },

    // Blog 重命名与合并重定向
    { from: '/blog/2023_overview', to: '/blog/2023-overview' },
    { from: '/blog/2024_overview', to: '/blog/2024-overview' },
    { from: '/blog/2025_overview', to: '/blog/2025-overview' },
    { from: '/blog/2024_preview', to: '/blog/2024-preview' },
    { from: '/blog/2024_blog', to: '/blog/2024-blog' },
    { from: '/blog/2025_blog', to: '/blog/2025-blog' },
    { from: '/blog/2026_blog', to: '/blog/2026-blog' },
    { from: '/blog/2024_spring_frial', to: '/blog/2024-spring-festival' },
    { from: '/blog/2024_0712', to: '/blog/comment-translate-plugin' },
    { from: '/blog/andorid_ios', to: '/blog/android-ios' },
    { from: '/blog/aws_use', to: '/blog/aws-ec2-ssh' },
    { from: '/blog/common_use', to: '/blog/common-commands' },
    { from: '/blog/docker_migrate', to: '/blog/docker-registry-migrate' },
    { from: '/blog/docker_use', to: '/blog/macos-centos-vm' },
    {
        from: ['/blog/fork', '/blog/git_cherry_pick', '/blog/git_use'],
        to: '/blog/git-commands',
    },
    { from: '/blog/git', to: '/blog/git-rebase' },
    { from: '/blog/githubpackages', to: '/blog/github-packages' },
    { from: '/blog/linux_clash', to: '/blog/centos-clash-proxy' },
    { from: '/blog/mac_use', to: '/blog/oh-my-zsh-setup' },
    { from: '/blog/medium', to: '/blog/rn-new-architecture' },
    { from: '/blog/new_build', to: '/blog' },
    { from: '/blog/nginx_settings', to: '/blog/nginx-config' },
    { from: '/blog/quzhongxinhua', to: '/blog/decentralization' },
    { from: '/blog/rn_app_size', to: '/blog/rn-app-size' },
    { from: '/blog/rn_debugg', to: '/blog/rn-debugger' },
    {
        from: '/blog/self_iterating_software',
        to: '/blog/self-iterating-software',
    },
    { from: '/blog/vs_debugger', to: '/blog/vscode-debugger' },
    { from: '/docs/native/flutter/20240728', to: '/blog/flutter-vs-rn' },

    // Docs 合并与重命名重定向
    {
        from: [
            '/docs/native/ios/IntegrationRn',
            '/docs/native/ios/ios_integative_rn',
        ],
        to: '/docs/native/ios/ios-integrate-rn',
    },
    {
        from: '/docs/native/ReactNative/expo/intro',
        to: '/docs/native/ReactNative/expo',
    },
    {
        from: [
            '/docs/native/ReactNative/metro',
            '/docs/native/ReactNative/metro_cache_delete',
            '/docs/native/ReactNative/clear_cache',
        ],
        to: '/docs/native/ReactNative/metro/intro',
    },
    {
        from: '/docs/backend/redis/redis_cli_command',
        to: '/docs/backend/redis/redis-cli',
    },
    {
        from: '/docs/backend/redis/redis_type_cli',
        to: '/docs/backend/redis/redis-data-types',
    },
    {
        from: ['/docs/backend/java/base/idea_use', '/docs/native/idea_search'],
        to: '/docs/backend/java/base/idea',
    },
    {
        from: ['/docs/web/font_famaily', '/docs/web/recommand_fontFamily'],
        to: '/docs/web/font-family',
    },
    {
        from: '/docs/web/React/react_debugger',
        to: '/docs/web/react_debugger/intro',
    },
    {
        from: [
            '/docs/native/android/work_manager',
            '/docs/native/android/workd',
        ],
        to: '/docs/native/android/work-manager',
    },
    {
        from: '/docs/native/harmony/harmony_rn',
        to: '/docs/native/harmony/intro',
    },
    {
        from: '/docs/ai/training/fundamentals/training/forwardpropagation',
        to: '/docs/ai/training/fundamentals/training/backpropagation',
    },
    { from: '/docs/web/route_ guard', to: '/docs/web/route-guard' },
    {
        from: '/docs/web/brower_open_window',
        to: '/docs/web/browser-open-window',
    },
    { from: '/docs/web/Js/unknow_any', to: '/docs/web/Js/unknown-vs-any' },
    { from: '/docs/web/Js/parseInt', to: '/docs/web/Js/parse-int' },
    {
        from: '/docs/web/React/reactAndReactDom',
        to: '/docs/web/React/react-and-react-dom',
    },
    { from: '/docs/web/React/useCallback', to: '/docs/web/React/use-callback' },
    {
        from: '/docs/web/React/useStateRef',
        to: '/docs/web/React/use-state-ref',
    },
    { from: '/docs/web/link', to: '/docs/cs_base/hard-link-vs-symlink' },
    { from: '/docs/backend/qa', to: '/docs/backend/java/spring/spring-faq' },
    {
        from: '/docs/backend/java/spring/dependciesInject',
        to: '/docs/backend/java/spring/dependency-injection',
    },
    {
        from: '/docs/backend/java/base/builderPattern',
        to: '/docs/backend/java/base/builder-pattern',
    },
    {
        from: '/docs/backend/architecture/05_Advanced_Architecture_Monorepo',
        to: '/docs/backend/architecture/advanced-architecture-monorepo',
    },
    { from: '/docs/ai/python/decorarors', to: '/docs/ai/python/decorators' },
    { from: '/docs/ai/python/pyhonm', to: '/docs/ai/python/python-m-module' },
    {
        from: '/docs/ai/training/fundamentals/math/qrthogonal',
        to: '/docs/ai/training/fundamentals/math/orthogonal',
    },
    {
        from: '/docs/ai/training/fundamentals/math/Variance',
        to: '/docs/ai/training/fundamentals/math/variance-and-std',
    },
    {
        from: '/docs/ai/training/fundamentals/training/K-FoldCrossValidation',
        to: '/docs/ai/training/fundamentals/training/k-fold-cross-validation',
    },
    {
        from: '/docs/ai/training/fundamentals/training/relu_boom',
        to: '/docs/ai/training/fundamentals/training/vanishing-exploding-gradients',
    },
    {
        from: '/docs/ai/training/optimization/weight_decy',
        to: '/docs/ai/training/optimization/weight-decay',
    },
    {
        from: '/docs/ai/training/frameworks/pytorch/act_pytorch',
        to: '/docs/ai/training/frameworks/pytorch/activation-functions',
    },
    {
        from: '/docs/ai/training/frameworks/tools/upload_proj',
        to: '/docs/ai/training/frameworks/tools/hf-upload-git',
    },
    {
        from: '/docs/native/first_pority',
        to: '/docs/native/first-class-citizen',
    },
    { from: '/docs/native/so_a', to: '/docs/native/shared-vs-static-library' },
    {
        from: '/docs/native/ReactNative/deugger',
        to: '/docs/native/ReactNative/rn-debugger',
    },
    {
        from: '/docs/native/ReactNative/preformence',
        to: '/docs/native/ReactNative/performance',
    },
    {
        from: '/docs/native/ReactNative/often_qs',
        to: '/docs/native/ReactNative/rn-common-issues',
    },
    {
        from: '/docs/native/ReactNative/Text',
        to: '/docs/native/ReactNative/text-component',
    },
    {
        from: '/docs/native/ReactNative/alignContent',
        to: '/docs/native/ReactNative/align-content',
    },
    {
        from: '/docs/native/ReactNative/animatedDriverNativeOrJs',
        to: '/docs/native/ReactNative/animated-driver-native-vs-js',
    },
    {
        from: '/docs/native/ReactNative/classAndFunction',
        to: '/docs/native/ReactNative/class-vs-function-component',
    },
    {
        from: '/docs/native/ReactNative/touchableWithoutFeedback',
        to: '/docs/native/ReactNative/touchable-without-feedback',
    },
    {
        from: '/docs/native/android/android_integation_rn',
        to: '/docs/native/android/android-integrate-rn',
    },
    {
        from: '/docs/native/android/android_integation_rn_third_dependiencies',
        to: '/docs/native/android/android-integrate-rn-third-party',
    },
    {
        from: '/docs/native/android/rn_qs',
        to: '/docs/native/android/android-rn-build-issues',
    },
    {
        from: '/docs/native/android/AnonymousInnerClass',
        to: '/docs/native/android/anonymous-inner-class',
    },
    {
        from: '/docs/native/android/androidStudioUse',
        to: '/docs/native/android/android-studio-tips',
    },
    {
        from: '/docs/native/android/executorService',
        to: '/docs/native/android/executor-service',
    },
    {
        from: '/docs/native/android/navigateUp',
        to: '/docs/native/android/navigate-up',
    },
    {
        from: '/docs/native/android/basic/navigation_grph',
        to: '/docs/native/android/basic/navigation-graph',
    },
    {
        from: '/docs/native/android/basic/viewBinding',
        to: '/docs/native/android/basic/view-binding',
    },
    {
        from: '/docs/native/android/Compose/room_qs',
        to: '/docs/native/android/Compose/room-destructive-migration',
    },
    {
        from: '/docs/native/android/gradle/ksp_impletation',
        to: '/docs/native/android/gradle/ksp-implementation',
    },
    {
        from: '/docs/native/android/javaView/measureHeight_height',
        to: '/docs/native/android/javaView/measured-height-vs-height',
    },
    {
        from: '/docs/native/android/kotlin/contructor',
        to: '/docs/native/android/kotlin/constructor',
    },
    {
        from: '/docs/native/android/kotlin/delagate',
        to: '/docs/native/android/kotlin/delegate',
    },
    {
        from: '/docs/native/android/kotlin/kotlin_permisson',
        to: '/docs/native/android/kotlin/kotlin-visibility-modifiers',
    },
    {
        from: '/docs/native/android/kotlin/mutableStateOf',
        to: '/docs/native/android/kotlin/mutable-state-of',
    },
    {
        from: '/docs/native/android/kotlin/transpat',
        to: '/docs/native/android/kotlin/compose-transparent-status-bar',
    },
    {
        from: '/docs/native/flutter/animated/imlicilt_animated',
        to: '/docs/native/flutter/animated/implicit-animated',
    },
    {
        from: '/docs/native/flutter/nested_scrollivew',
        to: '/docs/native/flutter/nested-scrollview',
    },
    {
        from: '/docs/native/flutter/useMaterialDesign',
        to: '/docs/native/flutter/uses-material-design',
    },
    {
        from: '/docs/native/ios/xcode_copolit',
        to: '/docs/native/ios/xcode-copilot',
    },
    {
        from: '/docs/native/ios/objectiveC/cgflot',
        to: '/docs/native/ios/objectiveC/cgfloat',
    },
    {
        from: '/docs/native/ios/objectiveC/enum_NSEnum',
        to: '/docs/native/ios/objectiveC/enum-vs-ns-enum',
    },
    {
        from: '/docs/native/ios/swift/getAndSet',
        to: '/docs/native/ios/swift/protocol-get-set',
    },
    {
        from: '/docs/native/ios/swiftui/focusState',
        to: '/docs/native/ios/swiftui/focus-state',
    },
    {
        from: '/docs/native/ios/swiftui/staetOrStateObject',
        to: '/docs/native/ios/swiftui/state-vs-state-object',
    },
    {
        from: '/docs/native/ios/uikit/keyborad_dismiss',
        to: '/docs/native/ios/uikit/keyboard-dismiss',
    },
    {
        from: '/docs/native/ios/uikit/storyboard_refrence',
        to: '/docs/native/ios/uikit/storyboard-reference',
    },
];

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
            [
                '@docusaurus/plugin-client-redirects',
                {
                    redirects: REDIRECTS,
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
                        'nginx',
                        'ini',
                    ],
                },
            }),
    };
}

module.exports = createConfig();
