"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[2688],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>c});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var p=r.createContext({}),m=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},s=function(e){var t=m(e.components);return r.createElement(p.Provider,{value:t},e.children)},d="mdxType",N={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,l=e.originalType,p=e.parentName,s=i(e,["components","mdxType","originalType","parentName"]),d=m(n),u=a,c=d["".concat(p,".").concat(u)]||d[u]||N[u]||l;return n?r.createElement(c,o(o({ref:t},s),{},{components:n})):r.createElement(c,o({ref:t},s))}));function c(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=n.length,o=new Array(l);o[0]=u;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i[d]="string"==typeof e?e:a,o[1]=i;for(var m=2;m<l;m++)o[m]=n[m];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},7013:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>o,default:()=>N,frontMatter:()=>l,metadata:()=>i,toc:()=>m});var r=n(7462),a=(n(7294),n(3905));const l={},o="INNER JOIN\u548cOUTER JOIN",i={unversionedId:"backend/mysql/inner_outer_join",id:"backend/mysql/inner_outer_join",title:"INNER JOIN\u548cOUTER JOIN",description:"\u4ecb\u7ecd",source:"@site/docs/backend/mysql/inner_outer_join.md",sourceDirName:"backend/mysql",slug:"/backend/mysql/inner_outer_join",permalink:"/yiwen-blog-website/docs/backend/mysql/inner_outer_join",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/backend/mysql/inner_outer_join.md",tags:[],version:"current",frontMatter:{},sidebar:"backendSidebar",previous:{title:"\u6570\u636e\u5e93\u7d22\u5f15",permalink:"/yiwen-blog-website/docs/backend/mysql/"},next:{title:"Mysql\u4ecb\u7ecd",permalink:"/yiwen-blog-website/docs/backend/mysql/intro"}},p={},m=[{value:"\u4ecb\u7ecd",id:"\u4ecb\u7ecd",level:2},{value:"\u793a\u4f8b",id:"\u793a\u4f8b",level:2},{value:"INNER JOIN",id:"inner-join",level:3},{value:"OUTER JOIN",id:"outer-join",level:3},{value:"\u4f7f\u7528\u573a\u666f",id:"\u4f7f\u7528\u573a\u666f",level:2},{value:"\u6a21\u62df\u5668 FULL OUTER JOIN",id:"\u6a21\u62df\u5668-full-outer-join",level:2}],s={toc:m},d="wrapper";function N(e){let{components:t,...n}=e;return(0,a.kt)(d,(0,r.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"inner-join\u548couter-join"},"INNER JOIN\u548cOUTER JOIN"),(0,a.kt)("h2",{id:"\u4ecb\u7ecd"},"\u4ecb\u7ecd"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"INNER JOIN\u548cOUTER JOIN\u662fSQL\u4e2d\u4e24\u79cd\u4e0d\u540c\u7684\u8868\u5173\u8054\u65b9\u5f0f\uff1a")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"INNER JOIN\uff1a\u4ec5\u8fd4\u56de\u4e24\u4e2a\u8868\u4e2d\u5339\u914d\u7684\u8bb0\u5f55\u3002\u5982\u679c\u8868A\u548c\u8868B\u4e4b\u95f4\u505aINNER JOIN\uff0c\u7ed3\u679c\u4e2d\u53ea\u5305\u542bA\u548cB\u90fd\u6709\u7684\u90a3\u90e8\u5206\u6570\u636e\u3002OUTER JOIN\uff1a\u53ef\u4ee5\u5206\u4e3a\u4e09\u79cd\uff1aLEFT OUTER JOIN\u3001RIGHT OUTER JOIN\u548cFULL OUTER JOIN\u3002"),(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"LEFT OUTER JOIN\uff08\u5de6\u8fde\u63a5\uff09\uff1a\u8fd4\u56de\u5de6\u8868\uff08JOIN\u8bed\u53e5\u524d\u7684\u8868\uff09\u7684\u6240\u6709\u8bb0\u5f55\uff0c\u5373\u4f7f\u5728\u53f3\u8868\u4e2d\u6ca1\u6709\u5339\u914d\u7684\u8bb0\u5f55\u3002\u53f3\u8868\u4e2d\u6ca1\u6709\u5339\u914d\u7684\u90e8\u5206\u4f1a\u663e\u793a\u4e3aNULL\u3002"),(0,a.kt)("li",{parentName:"ul"},"RIGHT OUTER JOIN\uff08\u53f3\u8fde\u63a5\uff09\uff1a\u8fd4\u56de\u53f3\u8868\u7684\u6240\u6709\u8bb0\u5f55\uff0c\u5373\u4f7f\u5728\u5de6\u8868\u4e2d\u6ca1\u6709\u5339\u914d\u7684\u8bb0\u5f55\u3002\u5de6\u8868\u4e2d\u6ca1\u6709\u5339\u914d\u7684\u90e8\u5206\u4f1a\u663e\u793a\u4e3aNULL\u3002"),(0,a.kt)("li",{parentName:"ul"},"FULL OUTER JOIN\uff08\u5168\u5916\u8fde\u63a5","[mysql\u4e2d\u6682\u4e0d\u652f\u6301]","\uff09\uff1a\u8fd4\u56de\u4e24\u4e2a\u8868\u4e2d\u6240\u6709\u7684\u8bb0\u5f55\uff0c\u4e0d\u8bba\u5b83\u4eec\u4e4b\u95f4\u662f\u5426\u5339\u914d\u3002\u4e0d\u5339\u914d\u7684\u90e8\u5206\u4f1a\u663e\u793a\u4e3aNULL\u3002")))),(0,a.kt)("h2",{id:"\u793a\u4f8b"},"\u793a\u4f8b"),(0,a.kt)("p",null,"\u8868\u7ed3\u6784"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sql"},"-- \u5047\u8bbe\u7684\u8868\u7ed3\u6784\nemployees: id, name, department_id\ndepartments: id, department_name\n")),(0,a.kt)("h3",{id:"inner-join"},"INNER JOIN"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sql"},"SELECT employees.name, departments.department_name\nFROM employees\nINNER JOIN departments ON employees.department_id = departments.id;\n")),(0,a.kt)("p",null,"\u5c06\u8fd4\u56de\u5728employees\u548cdepartments\u8868\u4e2d\u90fd\u6709\u5339\u914d\u7684\u8bb0\u5f55\u3002"),(0,a.kt)("h3",{id:"outer-join"},"OUTER JOIN"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sql"},"SELECT employees.name, departments.department_name\nFROM employees\nLEFT OUTER JOIN departments ON employees.department_id = departments.id;\n")),(0,a.kt)("p",null,"\u8fd9\u5c06\u8fd4\u56de\u6240\u6709employees\u8bb0\u5f55\u548c\u5b83\u4eec\u5bf9\u5e94\u7684departments\u8bb0\u5f55\u3002\u5982\u679c\u67d0\u4e2a\u5458\u5de5\u6ca1\u6709\u5bf9\u5e94\u7684\u90e8\u95e8\uff0c\u90e8\u95e8\u540d\u5c06\u663e\u793a\u4e3aNULL\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sql"},"SELECT employees.name, departments.department_name\nFROM employees\nRIGHT JOIN departments ON employees.department_id = departments.id;\n")),(0,a.kt)("p",null,"\u8fd9\u5c06\u8fd4\u56de\u6240\u6709employees\u8bb0\u5f55\u548c\u5b83\u4eec\u5bf9\u5e94\u7684departments\u8bb0\u5f55\u3002\u5982\u679c\u67d0\u4e2a\u90e8\u95e8\u6ca1\u6709\u5bf9\u5e94\u7684\u5458\u5de5\uff0c\u5458\u5de5\u540d\u5c06\u663e\u793a\u4e3aNULL\u3002"),(0,a.kt)("h2",{id:"\u4f7f\u7528\u573a\u666f"},"\u4f7f\u7528\u573a\u666f"),(0,a.kt)("p",null,"\u5728\u6570\u636e\u5e93\u67e5\u8be2\u4e2d\uff0cINNER JOIN\u548cOUTER JOIN\u7684\u4f7f\u7528\u573a\u666f\u901a\u5e38\u53d6\u51b3\u4e8e\u4f60\u9700\u8981\u4ece\u5173\u8054\u8868\u4e2d\u68c0\u7d22\u54ea\u4e9b\u6570\u636e\uff1a"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"INNER JOIN\uff1a\u9002\u7528\u4e8e\u4f60\u53ea\u5bf9\u4e24\u4e2a\u8868\u4e2d\u90fd\u5b58\u5728\u5339\u914d\u7684\u6570\u636e\u611f\u5174\u8da3\u7684\u60c5\u51b5\u3002\u5b83\u53ea\u8fd4\u56de\u5728\u4e24\u4e2a\u8868\u4e2d\u90fd\u6709\u76f8\u5339\u914d\u8bb0\u5f55\u7684\u7ed3\u679c\u3002\u4f8b\u5982\uff0c\u4f60\u53ea\u60f3\u8981\u90a3\u4e9b\u5728\u5458\u5de5\u8868\u548c\u90e8\u95e8\u8868\u4e2d\u90fd\u6709\u8bb0\u5f55\u7684\u5458\u5de5\u4fe1\u606f\u3002")),(0,a.kt)("li",{parentName:"ol"},(0,a.kt)("p",{parentName:"li"},"OUTER JOIN\uff1a\u9002\u7528\u4e8e\u4f60\u9700\u8981\u4ece\u4e00\u4e2a\u8868\u4e2d\u83b7\u53d6\u6240\u6709\u8bb0\u5f55\uff0c\u5e76\u4ece\u53e6\u4e00\u4e2a\u8868\u4e2d\u83b7\u53d6\u5339\u914d\u7684\u8bb0\u5f55\uff08\u5982\u679c\u5b58\u5728\uff09\u7684\u60c5\u51b5\u3002\u5982\u679c\u7b2c\u4e8c\u4e2a\u8868\u4e2d\u6ca1\u6709\u5339\u914d\u8bb0\u5f55\uff0c\u4ecd\u7136\u4f1a\u8fd4\u56de\u7b2c\u4e00\u4e2a\u8868\u4e2d\u7684\u8bb0\u5f55\u3002"),(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"LEFT OUTER JOIN\uff1a\u8fd4\u56de\u5de6\u8868\u7684\u6240\u6709\u8bb0\u5f55\uff0c\u4ee5\u53ca\u4e0e\u53f3\u8868\u5339\u914d\u7684\u8bb0\u5f55\u3002"),(0,a.kt)("li",{parentName:"ul"},"RIGHT OUTER JOIN\uff1a\u8fd4\u56de\u53f3\u8868\u7684\u6240\u6709\u8bb0\u5f55\uff0c\u4ee5\u53ca\u4e0e\u5de6\u8868\u5339\u914d\u7684\u8bb0\u5f55\u3002"),(0,a.kt)("li",{parentName:"ul"},"FULL OUTER JOIN\uff08\u4e0d\u662f\u6240\u6709\u6570\u636e\u5e93\u7cfb\u7edf\u90fd\u652f\u6301\uff09\uff1a\u8fd4\u56de\u4e24\u4e2a\u8868\u4e2d\u7684\u6240\u6709\u8bb0\u5f55\uff0c\u4e0d\u8bba\u5b83\u4eec\u4e4b\u95f4\u662f\u5426\u5339\u914d\u3002")))),(0,a.kt)("h2",{id:"\u6a21\u62df\u5668-full-outer-join"},"\u6a21\u62df\u5668 FULL OUTER JOIN"),(0,a.kt)("admonition",{type:"info"},(0,a.kt)("p",{parentName:"admonition"},(0,a.kt)("inlineCode",{parentName:"p"},"mysql"),"\u4e0d\u652f\u6301",(0,a.kt)("inlineCode",{parentName:"p"},"full join"),",\u4f46\u662f\u53ef\u4ee5\u6a21\u62df\u5b9e\u73b0")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-sql"},"SELECT employees.name, departments.department_name\nFROM employees\nLEFT JOIN departments ON employees.department_id = departments.id\nUNION\nSELECT employees.name, departments.department_name\nFROM employees\nRIGHT JOIN departments ON employees.department_id = departments.id;\n")))}N.isMDXComponent=!0}}]);