"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[1253],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>y});var n=r(67294);function l(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){l(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,l=function(e,t){if(null==e)return{};var r,n,l={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(l[r]=e[r]);return l}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(l[r]=e[r])}return l}var s=n.createContext({}),c=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},u=function(e){var t=c(e.components);return n.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,l=e.mdxType,a=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),p=c(r),m=l,y=p["".concat(s,".").concat(m)]||p[m]||d[m]||a;return r?n.createElement(y,o(o({ref:t},u),{},{components:r})):n.createElement(y,o({ref:t},u))}));function y(e,t){var r=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var a=r.length,o=new Array(a);o[0]=m;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i[p]="string"==typeof e?e:l,o[1]=i;for(var c=2;c<a;c++)o[c]=r[c];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},55895:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>d,frontMatter:()=>a,metadata:()=>i,toc:()=>c});var n=r(87462),l=(r(67294),r(3905));const a={},o="\u6570\u636e\u5e93\u7d22\u5f15",i={unversionedId:"backend/mysql/index",id:"backend/mysql/index",title:"\u6570\u636e\u5e93\u7d22\u5f15",description:"\u4ecb\u7ecd",source:"@site/docs/backend/mysql/index.md",sourceDirName:"backend/mysql",slug:"/backend/mysql/",permalink:"/yiwen-blog-website/en/docs/backend/mysql/",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/backend/mysql/index.md",tags:[],version:"current",frontMatter:{},sidebar:"backendSidebar",previous:{title:"Alter\u64cd\u4f5c\u7b26",permalink:"/yiwen-blog-website/en/docs/backend/mysql/alter"},next:{title:"INNER JOIN\u548cOUTER JOIN",permalink:"/yiwen-blog-website/en/docs/backend/mysql/inner_outer_join"}},s={},c=[{value:"\u4ecb\u7ecd",id:"\u4ecb\u7ecd",level:2},{value:"\u5355\u5217\u7d22\u5f15",id:"\u5355\u5217\u7d22\u5f15",level:2},{value:"\u590d\u5408\u7d22\u5f15",id:"\u590d\u5408\u7d22\u5f15",level:2},{value:"\u7d22\u5f15\u5fc5\u987b\u662f\u4e3b\u952e\u5417",id:"\u7d22\u5f15\u5fc5\u987b\u662f\u4e3b\u952e\u5417",level:2},{value:"\u7d22\u5f15\u5fc5\u987b\u552f\u4e00\u5417",id:"\u7d22\u5f15\u5fc5\u987b\u552f\u4e00\u5417",level:2},{value:"\u9700\u8981\u6bcf\u8fc7\u4e00\u6bb5\u65f6\u95f4\u8bbe\u7f6e\u4e00\u6b21\u7d22\u5f15\u5417",id:"\u9700\u8981\u6bcf\u8fc7\u4e00\u6bb5\u65f6\u95f4\u8bbe\u7f6e\u4e00\u6b21\u7d22\u5f15\u5417",level:2},{value:"\u590d\u5408\u7d22\u5f15\u7684\u4f18\u70b9",id:"\u590d\u5408\u7d22\u5f15\u7684\u4f18\u70b9",level:2},{value:"\u590d\u5408\u7d22\u5f15\u793a\u4f8b",id:"\u590d\u5408\u7d22\u5f15\u793a\u4f8b",level:2}],u={toc:c},p="wrapper";function d(e){let{components:t,...r}=e;return(0,l.kt)(p,(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"\u6570\u636e\u5e93\u7d22\u5f15"},"\u6570\u636e\u5e93\u7d22\u5f15"),(0,l.kt)("h2",{id:"\u4ecb\u7ecd"},"\u4ecb\u7ecd"),(0,l.kt)("p",null,"\u6570\u636e\u5e93\u7d22\u5f15\u7c7b\u4f3c\u4e8e\u4e66\u7c4d\u7684\u76ee\u5f55\uff0c\u5b83\u5e2e\u52a9\u6570\u636e\u5e93\u5feb\u901f\u5b9a\u4f4d\u548c\u68c0\u7d22\u6570\u636e\u3002\u5728MySQL\uff08\u5173\u7cfb\u578b\u6570\u636e\u5e93\uff09\u548cMongoDB\uff08\u975e\u5173\u7cfb\u578b\u6570\u636e\u5e93\uff09\u4e2d\uff0c\u7d22\u5f15\u90fd\u662f\u7528\u6765\u63d0\u9ad8\u67e5\u8be2\u6548\u7387\u7684\u3002\u5b83\u4eec\u901a\u8fc7\u4e3a\u6570\u636e\u5e93\u8868\u6216\u96c6\u5408\u4e2d\u7684\u4e00\u4e2a\u6216\u591a\u4e2a\u5b57\u6bb5\u521b\u5efa\u7d22\u5f15\uff0c\u51cf\u5c11\u4e86\u6570\u636e\u5e93\u641c\u7d22\u6574\u4e2a\u8868\u6216\u96c6\u5408\u7684\u9700\u8981\u3002\u7d22\u5f15\u7279\u522b\u6709\u7528\u4e8e\u5904\u7406\u5927\u91cf\u6570\u636e\u548c\u6267\u884c\u590d\u6742\u67e5\u8be2\u7684\u573a\u666f\uff0c\u56e0\u4e3a\u6ca1\u6709\u7d22\u5f15\u7684\u67e5\u8be2\u53ef\u80fd\u4f1a\u5bfc\u81f4\u5168\u8868\u626b\u63cf\uff0c\u8fd9\u5728\u5927\u578b\u6570\u636e\u5e93\u4e2d\u4f1a\u975e\u5e38\u8017\u65f6\u3002\u56e0\u6b64\uff0c\u5408\u7406\u4f7f\u7528\u7d22\u5f15\u662f\u4f18\u5316\u6570\u636e\u5e93\u6027\u80fd\u7684\u5173\u952e\u3002"),(0,l.kt)("h2",{id:"\u5355\u5217\u7d22\u5f15"},"\u5355\u5217\u7d22\u5f15"),(0,l.kt)("p",null,"\u5728MySQL\u4e2d\u8bbe\u7f6e\u7d22\u5f15\u901a\u5e38\u6d89\u53ca\u4f7f\u7528CREATE INDEX\u8bed\u53e5\u6216\u5728CREATE TABLE\u6216ALTER TABLE\u8bed\u53e5\u4e2d\u6307\u5b9a\u7d22\u5f15\u3002\u4f8b\u5982\uff0c\u4e3a\u8868\u4e2d\u7684\u67d0\u4e2a\u5217\u521b\u5efa\u7d22\u5f15\u7684\u8bed\u53e5\u5982\u4e0b\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-sql"},"CREATE INDEX \u7d22\u5f15\u540d ON \u8868\u540d(\u5217\u540d);\n")),(0,l.kt)("p",null,"\u7d22\u5f15\u7684\u539f\u7406\u662f\u901a\u8fc7\u4e3a\u8868\u4e2d\u7684\u4e00\u5217\u6216\u591a\u5217\u6570\u636e\u521b\u5efa\u4e00\u4e2a\u5feb\u901f\u67e5\u627e\u7684\u53c2\u7167\u7cfb\u7edf\u3002\u5b83\u7c7b\u4f3c\u4e8e\u4e66\u7c4d\u7684\u76ee\u5f55\uff0c\u4f7f\u6570\u636e\u5e93\u80fd\u591f\u5feb\u901f\u627e\u5230\u6570\u636e\uff0c\u800c\u4e0d\u662f\u626b\u63cf\u6574\u4e2a\u8868\u3002\u7d22\u5f15\u901a\u5e38\u4ee5B\u6811\uff08\u5e73\u8861\u6811\uff09\u6216\u54c8\u5e0c\u8868\u7684\u5f62\u5f0f\u5b9e\u73b0\uff0c\u8fd9\u4e9b\u7ed3\u6784\u4f7f\u5f97\u67e5\u627e\u3001\u63d2\u5165\u548c\u5220\u9664\u64cd\u4f5c\u66f4\u52a0\u9ad8\u6548\u3002\u6b63\u786e\u4f7f\u7528\u7d22\u5f15\u53ef\u4ee5\u663e\u8457\u63d0\u9ad8\u6570\u636e\u5e93\u7684\u67e5\u8be2\u6027\u80fd\uff0c\u7279\u522b\u662f\u5728\u5904\u7406\u5927\u91cf\u6570\u636e\u65f6\u3002"),(0,l.kt)("h2",{id:"\u590d\u5408\u7d22\u5f15"},"\u590d\u5408\u7d22\u5f15"),(0,l.kt)("p",null,"\u5728MySQL\u4e2d\uff0c\u8981\u521b\u5efa\u5305\u542b\u591a\u5217\u7684\u590d\u5408\u7d22\u5f15\uff0c\u60a8\u53ef\u4ee5\u5728CREATE INDEX\u547d\u4ee4\u4e2d\u6307\u5b9a\u591a\u4e2a\u5217\u540d\u3002\u4f8b\u5982\uff0c\u5982\u679c\u60a8\u6709\u4e00\u4e2a\u5305\u542bfirstName\u548clastName\u4e24\u5217\u7684employees\u8868\uff0c\u60f3\u8981\u57fa\u4e8e\u8fd9\u4e24\u5217\u521b\u5efa\u7d22\u5f15\uff0c\u53ef\u4ee5\u8fd9\u6837\u505a\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-sql"},"CREATE INDEX idx_name ON employees (firstName, lastName);\n")),(0,l.kt)("p",null,"\u8fd9\u91cc\uff0cidx_name\u662f\u7d22\u5f15\u7684\u540d\u79f0\uff0c\u800cemployees\u662f\u8868\u540d\u3002\u8fd9\u4e2a\u547d\u4ee4\u4f1a\u5728employees\u8868\u4e0a\u521b\u5efa\u4e00\u4e2a\u65b0\u7684\u7d22\u5f15idx_name\uff0c\u8be5\u7d22\u5f15\u57fa\u4e8efirstName\u548clastName\u4e24\u5217\u3002\u590d\u5408\u7d22\u5f15\u53ef\u4ee5\u63d0\u9ad8\u6d89\u53ca\u8fd9\u4e9b\u5217\u7684\u67e5\u8be2\u7684\u6548\u7387\uff0c\u4f46\u8981\u6ce8\u610f\u9009\u62e9\u6027\u548c\u67e5\u8be2\u6a21\u5f0f\uff0c\u4ee5\u786e\u4fdd\u7d22\u5f15\u7684\u6709\u6548\u6027\u3002"),(0,l.kt)("h2",{id:"\u7d22\u5f15\u5fc5\u987b\u662f\u4e3b\u952e\u5417"},"\u7d22\u5f15\u5fc5\u987b\u662f\u4e3b\u952e\u5417"),(0,l.kt)("p",null,"\u5982\u679c\u60a8\u5728MySQL\u4e2d\u5bf9\u67d0\u5217\u8bbe\u7f6e\u4e86\u7d22\u5f15\uff0c\u90a3\u4e48\u67e5\u8be2\u8fd9\u4e00\u5217\u7684\u901f\u5ea6\u901a\u5e38\u4f1a\u66f4\u5feb\uff0c\u56e0\u4e3a\u6570\u636e\u5e93\u53ef\u4ee5\u5229\u7528\u7d22\u5f15\u5feb\u901f\u5b9a\u4f4d\u6570\u636e\uff0c\u800c\u4e0d\u9700\u8981\u626b\u63cf\u6574\u4e2a\u8868\u3002\u7d22\u5f15\u5e76\u4e0d\u5fc5\u987b\u662f\u4e3b\u952e\u3002\u867d\u7136\u4e3b\u952e\u81ea\u52a8\u6210\u4e3a\u7d22\u5f15\uff0c\u4f46\u60a8\u53ef\u4ee5\u5728\u975e\u4e3b\u952e\u7684\u5217\u4e0a\u521b\u5efa\u7d22\u5f15\u3002\u5728\u9009\u62e9\u662f\u5426\u521b\u5efa\u7d22\u5f15\u65f6\uff0c\u5e94\u8003\u8651\u67e5\u8be2\u7684\u9700\u6c42\u548c\u8868\u7684\u6570\u636e\u7ed3\u6784\uff0c\u56e0\u4e3a\u7d22\u5f15\u867d\u7136\u53ef\u4ee5\u63d0\u9ad8\u67e5\u8be2\u901f\u5ea6\uff0c\u4f46\u4e5f\u4f1a\u589e\u52a0\u5199\u64cd\u4f5c\u7684\u5f00\u9500\uff0c\u5e76\u5360\u7528\u66f4\u591a\u5b58\u50a8\u7a7a\u95f4\u3002"),(0,l.kt)("h2",{id:"\u7d22\u5f15\u5fc5\u987b\u552f\u4e00\u5417"},"\u7d22\u5f15\u5fc5\u987b\u552f\u4e00\u5417"),(0,l.kt)("p",null,"\u60a8\u53ef\u4ee5\u5c06\u4ea7\u54c1\u8868\u7684name\u5217\u8bbe\u7f6e\u4e3a\u7d22\u5f15\uff0c\u8fd9\u6837\u67e5\u8be2\u8be5\u5217\u65f6\u901f\u5ea6\u901a\u5e38\u4f1a\u66f4\u5feb\u3002\u7d22\u5f15\u5e76\u4e0d\u8981\u6c42\u552f\u4e00\uff0c\u9664\u975e\u5b83\u662f\u4e00\u4e2a\u552f\u4e00\u7d22\u5f15\u6216\u4e3b\u952e\u7d22\u5f15\u3002\u666e\u901a\u7d22\u5f15\u5141\u8bb8\u91cd\u590d\u7684\u503c\u3002\u7d22\u5f15\u7684\u4e3b\u8981\u8981\u6c42\u5305\u62ec\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u9002\u7528\u6027\uff1a\u9002\u5408\u7528\u4e8e\u7ecf\u5e38\u4f5c\u4e3a\u67e5\u8be2\u6761\u4ef6\u7684\u5217\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u9009\u62e9\u6027\uff1a\u5177\u6709\u9ad8\u5ea6\u552f\u4e00\u6027\u7684\u5217\u66f4\u9002\u5408\u7d22\u5f15\uff0c\u56e0\u4e3a\u5b83\u4eec\u63d0\u4f9b\u66f4\u597d\u7684\u8fc7\u6ee4\u6548\u679c\u3002\n\u8bf7\u6ce8\u610f\uff0c\u867d\u7136\u7d22\u5f15\u53ef\u4ee5\u52a0\u901f\u67e5\u8be2\uff0c\u4f46\u5b83\u4eec\u4e5f\u4f1a\u589e\u52a0\u6570\u636e\u63d2\u5165\u3001\u66f4\u65b0\u548c\u5220\u9664\u7684\u5f00\u9500\uff0c\u5e76\u5360\u7528\u989d\u5916\u7684\u5b58\u50a8\u7a7a\u95f4\u3002\u56e0\u6b64\uff0c\u5728\u51b3\u5b9a\u5bf9\u54ea\u4e9b\u5217\u521b\u5efa\u7d22\u5f15\u65f6\uff0c\u9700\u8981\u6743\u8861\u5229\u5f0a\u3002")),(0,l.kt)("h2",{id:"\u9700\u8981\u6bcf\u8fc7\u4e00\u6bb5\u65f6\u95f4\u8bbe\u7f6e\u4e00\u6b21\u7d22\u5f15\u5417"},"\u9700\u8981\u6bcf\u8fc7\u4e00\u6bb5\u65f6\u95f4\u8bbe\u7f6e\u4e00\u6b21\u7d22\u5f15\u5417"),(0,l.kt)("p",null,"\u521b\u5efa\u7d22\u5f15\u901a\u5e38\u662f\u4e00\u6b21\u6027\u7684\u64cd\u4f5c\uff0c\u800c\u4e0d\u662f\u9700\u8981\u5b9a\u671f\u91cd\u590d\u7684\u4efb\u52a1\u3002\u4e00\u65e6\u5728\u8868\u7684\u5217\u4e0a\u521b\u5efa\u4e86\u7d22\u5f15\uff0c\u5b83\u4f1a\u968f\u7740\u8868\u4e2d\u6570\u636e\u7684\u63d2\u5165\u3001\u66f4\u65b0\u548c\u5220\u9664\u81ea\u52a8\u7ef4\u62a4\u3002\u4e0d\u8fc7\uff0c\u6709\u51e0\u79cd\u60c5\u51b5\u53ef\u80fd\u9700\u8981\u60a8\u91cd\u65b0\u8003\u8651\u548c\u8c03\u6574\u7d22\u5f15\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u6570\u636e\u53d8\u5316\uff1a\u5982\u679c\u8868\u7684\u6570\u636e\u5206\u5e03\u6216\u67e5\u8be2\u6a21\u5f0f\u53d1\u751f\u663e\u8457\u53d8\u5316\uff0c\u53ef\u80fd\u9700\u8981\u66f4\u65b0\u6216\u4f18\u5316\u7d22\u5f15\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u6027\u80fd\u76d1\u63a7\uff1a\u5b9a\u671f\u76d1\u63a7\u6570\u636e\u5e93\u6027\u80fd\u53ef\u4ee5\u5e2e\u52a9\u8bc6\u522b\u662f\u5426\u9700\u8981\u5bf9\u73b0\u6709\u7d22\u5f15\u8fdb\u884c\u8c03\u6574\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u7cfb\u7edf\u5347\u7ea7\uff1a\u5728\u8fdb\u884c\u6570\u636e\u5e93\u7cfb\u7edf\u5347\u7ea7\u6216\u66f4\u6539\u65f6\uff0c\u53ef\u80fd\u9700\u8981\u91cd\u65b0\u5ba1\u89c6\u7d22\u5f15\u7b56\u7565\u3002")),(0,l.kt)("h2",{id:"\u590d\u5408\u7d22\u5f15\u7684\u4f18\u70b9"},"\u590d\u5408\u7d22\u5f15\u7684\u4f18\u70b9"),(0,l.kt)("p",null,"\u590d\u5408\u7d22\u5f15\u76f8\u6bd4\u5355\u5217\u7d22\u5f15\u7684\u4f18\u70b9\u4e3b\u8981\u4f53\u73b0\u5728\u591a\u5217\u67e5\u8be2\u4e0a\u3002\u5f53\u67e5\u8be2\u6761\u4ef6\u6d89\u53ca\u590d\u5408\u7d22\u5f15\u4e2d\u7684\u591a\u4e2a\u5217\u65f6\uff0c\u590d\u5408\u7d22\u5f15\u53ef\u4ee5\u63d0\u4f9b\u66f4\u9ad8\u7684\u67e5\u8be2\u6548\u7387\u3002\u4f8b\u5982\uff0c\u5728\u4e00\u4e2a\u7531\u4e24\u5217\u7ec4\u6210\u7684\u590d\u5408\u7d22\u5f15\u4e2d\uff0c\u5982\u679c\u67e5\u8be2\u6761\u4ef6\u540c\u65f6\u4f7f\u7528\u8fd9\u4e24\u5217\uff0c\u67e5\u8be2\u5c06\u975e\u5e38\u5feb\u901f\u3002",(0,l.kt)("strong",{parentName:"p"},"\u4f46\u662f\uff0c\u590d\u5408\u7d22\u5f15\u7684\u6548\u679c\u53d6\u51b3\u4e8e\u67e5\u8be2\u6761\u4ef6\u548c\u7d22\u5f15\u4e2d\u5217\u7684\u987a\u5e8f\u3002\u4f7f\u7528\u590d\u5408\u7d22\u5f15\u65f6\uff0c\u4e00\u822c\u662f\u67e5\u8be2\u6761\u4ef6\u4ece\u5de6\u5230\u53f3\u5339\u914d\u7d22\u5f15\u5217\u65f6\u6548\u679c\u6700\u4f73\u3002\u5982\u679c\u67e5\u8be2\u53ea\u4f7f\u7528\u7d22\u5f15\u4e2d\u7684\u7b2c\u4e00\u4e2a\u5217\uff0c\u4ecd\u7136\u53ef\u4ee5\u5229\u7528\u7d22\u5f15\uff0c\u4f46\u5982\u679c\u67e5\u8be2\u4f7f\u7528\u7684\u662f\u7d22\u5f15\u540e\u9762\u7684\u5217\u800c\u8df3\u8fc7\u4e86\u524d\u9762\u7684\u5217\uff0c\u7d22\u5f15\u53ef\u80fd\u5c31\u4e0d\u4f1a\u88ab\u4f7f\u7528")),(0,l.kt)("h2",{id:"\u590d\u5408\u7d22\u5f15\u793a\u4f8b"},"\u590d\u5408\u7d22\u5f15\u793a\u4f8b"),(0,l.kt)("p",null,"\u4f8b\u5982\uff0c\u5047\u8bbe\u60a8\u6709\u4e00\u4e2a\u540d\u4e3aOrders\u7684\u8868\uff0c\u5b83\u5305\u542bcustomerID\u548corderDate\u4e24\u4e2a\u5b57\u6bb5\u3002\u60a8\u60f3\u8981\u6839\u636e\u8fd9\u4e24\u4e2a\u5b57\u6bb5\u5feb\u901f\u67e5\u8be2\u8ba2\u5355\u3002\u60a8\u53ef\u4ee5\u521b\u5efa\u4e00\u4e2a\u590d\u5408\u7d22\u5f15\uff0c\u540c\u65f6\u8986\u76d6\u8fd9\u4e24\u4e2a\u5b57\u6bb5\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-sql"},"CREATE INDEX idx_customer_order ON Orders (customerID, orderDate);\n")),(0,l.kt)("p",null,"\u8981\u6709\u6548\u5229\u7528customerID\u548corderDate\u7684\u590d\u5408\u7d22\u5f15\uff0c\u60a8\u7684\u67e5\u8be2\u5e94\u8be5\u9996\u5148\u4f7f\u7528customerID\uff0c\u53ef\u9009\u5730\u518d\u4f7f\u7528orderDate\u3002\u4f8b\u5982\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-sql"},"SELECT * FROM Orders WHERE customerID = 123;\n\nSELECT * FROM Orders WHERE customerID = 123 AND orderDate = '2021-01-01';\n\nSELECT * FROM Orders WHERE customerID = 123 ORDER BY orderDate;\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u8fd9\u4e9b\u67e5\u8be2\u90fd\u4f1a\u4f18\u5148\u4f7f\u7528customerID\uff0c\u8fd9\u4e0e\u590d\u5408\u7d22\u5f15\u7684\u7b2c\u4e00\u4e2a\u5b57\u6bb5\u5339\u914d\u3002\u5982\u679c\u60a8\u7684\u67e5\u8be2\u53ea\u6d89\u53caorderDate\u800c\u4e0d\u662fcustomerID\uff0c\u5219\u7d22\u5f15\u4e0d\u4f1a\u88ab\u6709\u6548\u5229\u7528\u3002")))}d.isMDXComponent=!0}}]);