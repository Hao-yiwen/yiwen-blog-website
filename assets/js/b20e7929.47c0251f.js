"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[6071],{3905:(e,t,r)=>{r.d(t,{Zo:()=>c,kt:()=>b});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),p=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},c=function(e){var t=p(e.components);return n.createElement(s.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},g=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),u=p(r),g=o,b=u["".concat(s,".").concat(g)]||u[g]||m[g]||a;return r?n.createElement(b,i(i({ref:t},c),{},{components:r})):n.createElement(b,i({ref:t},c))}));function b(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=g;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[u]="string"==typeof e?e:o,i[1]=l;for(var p=2;p<a;p++)i[p]=r[p];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}g.displayName="MDXCreateElement"},7706:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>m,frontMatter:()=>a,metadata:()=>l,toc:()=>p});var n=r(7462),o=(r(7294),r(3905));const a={title:"\u5e38\u7528\u547d\u4ee4"},i="\u5e38\u7528\u547d\u4ee4",l={permalink:"/yiwen-blog-website/blog/common_use",editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/blog/common_use.md",source:"@site/blog/common_use.md",title:"\u5e38\u7528\u547d\u4ee4",description:"\u7ec8\u7aef\u4ee3\u7406",date:"2024-01-17T07:35:09.000Z",formattedDate:"2024\u5e741\u670817\u65e5",tags:[],readingTime:.63,hasTruncateMarker:!1,authors:[],frontMatter:{title:"\u5e38\u7528\u547d\u4ee4"},prevItem:{title:"aws\u4e2dec2\u767b\u5f55",permalink:"/yiwen-blog-website/blog/aws_use"},nextItem:{title:"Copilot",permalink:"/yiwen-blog-website/blog/copilot"}},s={authorsImageUrls:[]},p=[{value:"\u7ec8\u7aef\u4ee3\u7406",id:"\u7ec8\u7aef\u4ee3\u7406",level:2},{value:"\u5e38\u7528\u547d\u4ee4",id:"\u5e38\u7528\u547d\u4ee4",level:2}],c={toc:p},u="wrapper";function m(e){let{components:t,...r}=e;return(0,o.kt)(u,(0,n.Z)({},c,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"\u7ec8\u7aef\u4ee3\u7406"},"\u7ec8\u7aef\u4ee3\u7406"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash",metastring:'title="ss"',title:'"ss"'},"export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;\n")),(0,o.kt)("h2",{id:"\u5e38\u7528\u547d\u4ee4"},"\u5e38\u7528\u547d\u4ee4"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"\u8bbe\u7f6e\u955c\u50cf\u6e90")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash"},'\u6dd8\u5b9d: npm config set registry https://registry.npm.taobao.org\n\nnpm\u5b98\u65b9: npm config set registry https://registry.npmjs.org/\n\nyarn: yarn config set registry https://registry.npm.taobao.org\n\npip:\n# \u4fee\u6539 ~/.pip/pip.conf\n[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n\ndocker:\n# \u4fee\u6539 /etc/docker/daemon.json\n{\n  "registry-mirrors": ["https://your-mirror.com"]\n}\n')),(0,o.kt)("ol",{start:2},(0,o.kt)("li",{parentName:"ol"},"\u5e38\u7528brew\u547d\u4ee4")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash"},"// \u5305\u4fe1\u606f\nbrew info nginx\n// \u91cd\u542f\u670d\u52a1\nbrew services restart nginx\n// \u670d\u52a1\u4fe1\u606f\nbrew services info nginx\n")),(0,o.kt)("ol",{start:3},(0,o.kt)("li",{parentName:"ol"},"mac\u4fee\u6539hosts")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-bash"},"// \u7f16\u8f91hosts\nsudo vim /etc/hosts\n// \u6dfb\u52a0\u7f51\u5740\n127.0.0.1    example.com\n// \u5237\u65b0 DNS \u7f13\u5b58\nsudo dscacheutil -flushcache\n")))}m.isMDXComponent=!0}}]);