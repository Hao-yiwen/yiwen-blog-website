"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[4117],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>g});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var l=r.createContext({}),u=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=u(e.components);return r.createElement(l.Provider,{value:t},e.children)},s="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),s=u(n),d=o,g=s["".concat(l,".").concat(d)]||s[d]||m[d]||a;return n?r.createElement(g,i(i({ref:t},p),{},{components:n})):r.createElement(g,i({ref:t},p))}));function g(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=d;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c[s]="string"==typeof e?e:o,i[1]=c;for(var u=2;u<a;u++)i[u]=n[u];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},56681:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>i,default:()=>m,frontMatter:()=>a,metadata:()=>c,toc:()=>u});var r=n(87462),o=(n(67294),n(3905));const a={},i="\u8def\u7531\u5b88\u536b",c={unversionedId:"web/route_ guard",id:"web/route_ guard",title:"\u8def\u7531\u5b88\u536b",description:"\u8def\u7531\u5b88\u536b\uff08Route Guard\uff09\u662f\u524d\u7aef\u8def\u7531\u7ba1\u7406\u4e2d\u7684\u4e00\u4e2a\u6982\u5ff5\uff0c\u4e3b\u8981\u7528\u4e8e\u63a7\u5236\u7528\u6237\u5bf9\u7279\u5b9a\u8def\u7531\u7684\u8bbf\u95ee\u6743\u9650\u3002\u5b83\u5728\u5355\u9875\u5e94\u7528\uff08SPA\uff09\u5982Angular\u3001React\u3001Vue.js\u7b49\u6846\u67b6\u4e2d\u5c24\u4e3a\u5e38\u89c1\uff0c\u7528\u4e8e\u5728\u8def\u7531\u7ea7\u522b\u5904\u7406\u8ba4\u8bc1\u548c\u6388\u6743\u3002",source:"@site/docs/web/route_ guard.md",sourceDirName:"web",slug:"/web/route_ guard",permalink:"/yiwen-blog-website/en/docs/web/route_ guard",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/web/route_ guard.md",tags:[],version:"current",frontMatter:{},sidebar:"webSidebar",previous:{title:"\u63a8\u8350\u5b57\u4f53",permalink:"/yiwen-blog-website/en/docs/web/recommand_fontFamily"},next:{title:"Taro",permalink:"/yiwen-blog-website/en/docs/category/taro"}},l={},u=[{value:"\u8def\u7531\u5b88\u536b\u7684\u4f5c\u7528",id:"\u8def\u7531\u5b88\u536b\u7684\u4f5c\u7528",level:2},{value:"\u793a\u4f8b\uff1aReact\u8def\u7531\u5b88\u536b",id:"\u793a\u4f8breact\u8def\u7531\u5b88\u536b",level:2},{value:"\u793a\u4f8b\uff1aUmi\u8def\u7531\u5b88\u536b",id:"\u793a\u4f8bumi\u8def\u7531\u5b88\u536b",level:2}],p={toc:u},s="wrapper";function m(e){let{components:t,...n}=e;return(0,o.kt)(s,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"\u8def\u7531\u5b88\u536b"},"\u8def\u7531\u5b88\u536b"),(0,o.kt)("p",null,"\u8def\u7531\u5b88\u536b\uff08Route Guard\uff09\u662f\u524d\u7aef\u8def\u7531\u7ba1\u7406\u4e2d\u7684\u4e00\u4e2a\u6982\u5ff5\uff0c\u4e3b\u8981\u7528\u4e8e\u63a7\u5236\u7528\u6237\u5bf9\u7279\u5b9a\u8def\u7531\u7684\u8bbf\u95ee\u6743\u9650\u3002\u5b83\u5728\u5355\u9875\u5e94\u7528\uff08SPA\uff09\u5982Angular\u3001React\u3001Vue.js\u7b49\u6846\u67b6\u4e2d\u5c24\u4e3a\u5e38\u89c1\uff0c\u7528\u4e8e\u5728\u8def\u7531\u7ea7\u522b\u5904\u7406\u8ba4\u8bc1\u548c\u6388\u6743\u3002"),(0,o.kt)("h2",{id:"\u8def\u7531\u5b88\u536b\u7684\u4f5c\u7528"},"\u8def\u7531\u5b88\u536b\u7684\u4f5c\u7528"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"\u8bbf\u95ee\u63a7\u5236\uff1a\u9632\u6b62\u672a\u7ecf\u6388\u6743\u7684\u7528\u6237\u8bbf\u95ee\u67d0\u4e9b\u8def\u7531\u3002\u4f8b\u5982\uff0c\u963b\u6b62\u672a\u767b\u5f55\u7528\u6237\u8bbf\u95ee\u9700\u8981\u8ba4\u8bc1\u7684\u9875\u9762\u3002"),(0,o.kt)("li",{parentName:"ol"},"\u91cd\u5b9a\u5411\uff1a\u6839\u636e\u7528\u6237\u7684\u767b\u5f55\u72b6\u6001\u6216\u6743\u9650\uff0c\u81ea\u52a8\u91cd\u5b9a\u5411\u5230\u4e0d\u540c\u7684\u8def\u7531\u3002"),(0,o.kt)("li",{parentName:"ol"},"\u6570\u636e\u9884\u52a0\u8f7d\uff1a\u5728\u8def\u7531\u6fc0\u6d3b\u4e4b\u524d\u9884\u5148\u52a0\u8f7d\u6240\u9700\u6570\u636e\u3002"),(0,o.kt)("li",{parentName:"ol"},"\u6761\u4ef6\u6e32\u67d3\uff1a\u57fa\u4e8e\u7279\u5b9a\u6761\u4ef6\uff08\u5982\u7528\u6237\u89d2\u8272\u3001\u529f\u80fd\u5f00\u5173\uff09\u51b3\u5b9a\u662f\u5426\u6e32\u67d3\u67d0\u4e2a\u8def\u7531\u3002")),(0,o.kt)("h2",{id:"\u793a\u4f8breact\u8def\u7531\u5b88\u536b"},"\u793a\u4f8b\uff1aReact\u8def\u7531\u5b88\u536b"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-js"},"// \u5728React\u4e2d\uff0c\u8def\u7531\u5b88\u536b\u53ef\u4ee5\u901a\u8fc7\u521b\u5efa\u4e00\u4e2a\u9ad8\u9636\u7ec4\u4ef6\u6765\u5b9e\u73b0\uff1a\nimport React from 'react';\nimport { Redirect, Route } from 'react-router-dom';\n\nconst ProtectedRoute = ({ component: Component, ...rest }) => {\n    const isAuthenticated = // \u903b\u8f91\u5224\u65ad\u7528\u6237\u662f\u5426\u8ba4\u8bc1\n    return (\n        <Route\n            {...rest}\n            render={props =>\n                isAuthenticated ? <Component {...props} /> : <Redirect to=\"/login\" />\n            }\n        />\n    );\n};\n\n// \u5728\u8def\u7531\u914d\u7f6e\u4e2d\u4f7f\u7528\n<ProtectedRoute path=\"/protected\" component={ProtectedComponent} />\n")),(0,o.kt)("h2",{id:"\u793a\u4f8bumi\u8def\u7531\u5b88\u536b"},"\u793a\u4f8b\uff1aUmi\u8def\u7531\u5b88\u536b"),(0,o.kt)("p",null,"\u5728",(0,o.kt)("inlineCode",{parentName:"p"},"app.ts"),"\u4e2d\u8fdb\u884c\u5982\u4e0b\u914d\u7f6e"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-js"},"export function onRouteChange({ location, routes, action }) {\n  const token = localStorage.getItem('userToken');\n  const isLogin = !!token; // \u6839\u636e\u5b9e\u9645\u60c5\u51b5\u5224\u65ad\u767b\u5f55\u72b6\u6001\n  const isLoginPage = location.pathname === '/login';\n\n  // \u5982\u679c\u7528\u6237\u672a\u767b\u5f55\u4e14\u4e0d\u5728\u767b\u5f55\u9875\u9762\uff0c\u5219\u91cd\u5b9a\u5411\u5230\u767b\u5f55\u9875\u9762\n  if (!isLogin && !isLoginPage) {\n    location.pathname = '/login';\n  }\n}\n")))}m.isMDXComponent=!0}}]);