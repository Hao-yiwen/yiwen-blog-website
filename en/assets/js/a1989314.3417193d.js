"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[2808],{3905:(e,t,r)=>{r.d(t,{Zo:()=>d,kt:()=>b});var n=r(7294);function i(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){i(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function c(e,t){if(null==e)return{};var r,n,i=function(e,t){if(null==e)return{};var r,n,i={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(i[r]=e[r]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(i[r]=e[r])}return i}var l=n.createContext({}),s=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},d=function(e){var t=s(e.components);return n.createElement(l.Provider,{value:t},e.children)},p="mdxType",u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},v=n.forwardRef((function(e,t){var r=e.components,i=e.mdxType,a=e.originalType,l=e.parentName,d=c(e,["components","mdxType","originalType","parentName"]),p=s(r),v=i,b=p["".concat(l,".").concat(v)]||p[v]||u[v]||a;return r?n.createElement(b,o(o({ref:t},d),{},{components:r})):n.createElement(b,o({ref:t},d))}));function b(e,t){var r=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=r.length,o=new Array(a);o[0]=v;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c[p]="string"==typeof e?e:i,o[1]=c;for(var s=2;s<a;s++)o[s]=r[s];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}v.displayName="MDXCreateElement"},2015:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>o,default:()=>u,frontMatter:()=>a,metadata:()=>c,toc:()=>s});var n=r(7462),i=(r(7294),r(3905));const a={},o="\u5982\u4f55\u5728android\u4e2d\u4f7f\u7528svg\u56fe\u6807",c={unversionedId:"native/android/basic/svg",id:"native/android/basic/svg",title:"\u5982\u4f55\u5728android\u4e2d\u4f7f\u7528svg\u56fe\u6807",description:"\u8f6c\u6362svg\u4e3aVector Drawable",source:"@site/docs/native/android/basic/svg.md",sourceDirName:"native/android/basic",slug:"/native/android/basic/svg",permalink:"/yiwen-blog-website/en/docs/native/android/basic/svg",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/native/android/basic/svg.md",tags:[],version:"current",frontMatter:{},sidebar:"nativeSidebar",previous:{title:"Androud\u4e2d\u7684\u5c4f\u5e55\u65b9\u5411",permalink:"/yiwen-blog-website/en/docs/native/android/basic/orientation"},next:{title:"viewBinding",permalink:"/yiwen-blog-website/en/docs/native/android/basic/viewBinding"}},l={},s=[{value:"\u8f6c\u6362svg\u4e3aVector Drawable",id:"\u8f6c\u6362svg\u4e3avector-drawable",level:2},{value:"\u4f7f\u7528Image\u7ec4\u4ef6\u6dfb\u52a0",id:"\u4f7f\u7528image\u7ec4\u4ef6\u6dfb\u52a0",level:2}],d={toc:s},p="wrapper";function u(e){let{components:t,...r}=e;return(0,i.kt)(p,(0,n.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"\u5982\u4f55\u5728android\u4e2d\u4f7f\u7528svg\u56fe\u6807"},"\u5982\u4f55\u5728android\u4e2d\u4f7f\u7528svg\u56fe\u6807"),(0,i.kt)("h2",{id:"\u8f6c\u6362svg\u4e3avector-drawable"},"\u8f6c\u6362svg\u4e3aVector Drawable"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u9009\u62e9",(0,i.kt)("inlineCode",{parentName:"li"},"New > Vector Asset"),"\u8fdb\u884c\u8f6c\u6362")),(0,i.kt)("h2",{id:"\u4f7f\u7528image\u7ec4\u4ef6\u6dfb\u52a0"},"\u4f7f\u7528Image\u7ec4\u4ef6\u6dfb\u52a0"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-kotlin"},'Image(\n    painter = painterResource(id = R.drawable.vector_drawable), // \u66ff\u6362\u4e3a\u4f60\u7684\u8d44\u6e90ID\n    contentDescription = "\u63cf\u8ff0\u4f60\u7684\u56fe\u5f62", // \u63d0\u4f9b\u65e0\u969c\u788d\u652f\u6301\u7684\u6587\u672c\u63cf\u8ff0\n    modifier = Modifier.fillMaxSize() // \u6216\u5176\u4ed6Modifier\u6765\u8c03\u6574\u663e\u793a\u65b9\u5f0f\n)\n')))}u.isMDXComponent=!0}}]);