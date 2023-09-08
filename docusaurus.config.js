// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'yiwen',
  favicon: 'img/logo.svg',
  url: 'https://your-docusaurus-test-site.com',
  baseUrl: '/yiwen-blog-website/',
  organizationName: 'yiwen',
  projectName: 'docusaurus',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'zh',
    locales: ['en','zh'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'yiwen',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'webSidebar',
            position: 'left',
            label: 'Web开发',
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
          {to: '/blog', label: '博客', position: 'left'},
          {
            type: 'localeDropdown',
            position: 'right'
          },
          {
            href: 'https://github.com/Hao-yiwen',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} yiwen web, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
