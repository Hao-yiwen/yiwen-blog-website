"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[6881],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>b});var r=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=r.createContext({}),p=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},c="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,a=e.originalType,s=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),c=p(n),m=i,b=c["".concat(s,".").concat(m)]||c[m]||d[m]||a;return n?r.createElement(b,l(l({ref:t},u),{},{components:n})):r.createElement(b,l({ref:t},u))}));function b(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var a=n.length,l=new Array(a);l[0]=m;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o[c]="string"==typeof e?e:i,l[1]=o;for(var p=2;p<a;p++)l[p]=n[p];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},48443:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>l,default:()=>d,frontMatter:()=>a,metadata:()=>o,toc:()=>p});var r=n(87462),i=(n(67294),n(3905));const a={},l="Build Phase",o={unversionedId:"native/ios/build_phase",id:"native/ios/build_phase",title:"Build Phase",description:"\u5728 Xcode \u4e2d\uff0cBuild Phases \u5b9a\u4e49\u4e86\u5728\u9879\u76ee\u6784\u5efa\u8fc7\u7a0b\u4e2d\u7684\u5404\u4e2a\u9636\u6bb5\u5e94\u8be5\u6267\u884c\u7684\u4efb\u52a1\u3002\u8fd9\u4e9b\u4efb\u52a1\u5305\u62ec\u7f16\u8bd1\u6e90\u4ee3\u7801\u3001\u590d\u5236\u8d44\u6e90\u6587\u4ef6\u3001\u94fe\u63a5\u5e93\u6587\u4ef6\u7b49\u3002Build Phases \u5728\u6574\u4e2a\u6784\u5efa\u8fc7\u7a0b\u4e2d\u6709\u7279\u5b9a\u7684\u6267\u884c\u987a\u5e8f\uff0c\u65e2\u5305\u62ec\u5728\u6784\u5efa\u4e4b\u524d\uff0c\u4e5f\u5305\u62ec\u5728\u6784\u5efa\u4e4b\u540e\u6267\u884c\u7684\u4efb\u52a1\u3002",source:"@site/docs/native/ios/build_phase.md",sourceDirName:"native/ios",slug:"/native/ios/build_phase",permalink:"/yiwen-blog-website/en/docs/native/ios/build_phase",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/native/ios/build_phase.md",tags:[],version:"current",frontMatter:{},sidebar:"nativeSidebar",previous:{title:"FAQ",permalink:"/yiwen-blog-website/en/docs/native/ios/faq"},next:{title:"ios\u5b57\u4f53",permalink:"/yiwen-blog-website/en/docs/native/ios/fontfamily"}},s={},p=[{value:"Build Phases \u7684\u987a\u5e8f\u548c\u7c7b\u578b\uff1a",id:"build-phases-\u7684\u987a\u5e8f\u548c\u7c7b\u578b",level:2},{value:"\u603b\u7ed3\uff1a",id:"\u603b\u7ed3",level:2}],u={toc:p},c="wrapper";function d(e){let{components:t,...n}=e;return(0,i.kt)(c,(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"build-phase"},"Build Phase"),(0,i.kt)("p",null,"\u5728 Xcode \u4e2d\uff0cBuild Phases \u5b9a\u4e49\u4e86\u5728\u9879\u76ee\u6784\u5efa\u8fc7\u7a0b\u4e2d\u7684\u5404\u4e2a\u9636\u6bb5\u5e94\u8be5\u6267\u884c\u7684\u4efb\u52a1\u3002\u8fd9\u4e9b\u4efb\u52a1\u5305\u62ec\u7f16\u8bd1\u6e90\u4ee3\u7801\u3001\u590d\u5236\u8d44\u6e90\u6587\u4ef6\u3001\u94fe\u63a5\u5e93\u6587\u4ef6\u7b49\u3002Build Phases \u5728\u6574\u4e2a\u6784\u5efa\u8fc7\u7a0b\u4e2d\u6709\u7279\u5b9a\u7684\u6267\u884c\u987a\u5e8f\uff0c\u65e2\u5305\u62ec\u5728\u6784\u5efa\u4e4b\u524d\uff0c\u4e5f\u5305\u62ec\u5728\u6784\u5efa\u4e4b\u540e\u6267\u884c\u7684\u4efb\u52a1\u3002"),(0,i.kt)("h2",{id:"build-phases-\u7684\u987a\u5e8f\u548c\u7c7b\u578b"},"Build Phases \u7684\u987a\u5e8f\u548c\u7c7b\u578b\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Pre-actions\uff08\u6784\u5efa\u4e4b\u524d\uff09: \u5728 Build Phases \u7684\u754c\u9762\u4e2d\u4e0d\u76f4\u63a5\u663e\u793a\uff0c\u4f46\u53ef\u4ee5\u5728 Xcode \u7684 scheme \u8bbe\u7f6e\u4e2d\u5b9a\u4e49\u3002\u8fd9\u4e9b\u52a8\u4f5c\u5728\u6784\u5efa\u8fc7\u7a0b\u5f00\u59cb\u4e4b\u524d\u6267\u884c\uff0c\u901a\u5e38\u7528\u4e8e\u8bbe\u7f6e\u73af\u5883\u53d8\u91cf\u6216\u6267\u884c\u811a\u672c\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Target Dependencies\uff08\u76ee\u6807\u4f9d\u8d56\uff09: \u786e\u5b9a\u5f53\u524d\u6784\u5efa\u76ee\u6807\uff08target\uff09\u4f9d\u8d56\u4e8e\u9879\u76ee\u4e2d\u7684\u5176\u4ed6\u76ee\u6807\u3002\u5982\u679c\u6709\u4f9d\u8d56\u5173\u7cfb\uff0c\u90a3\u4e48\u8fd9\u4e9b\u4f9d\u8d56\u76ee\u6807\u4f1a\u5148\u88ab\u6784\u5efa\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Compile Sources\uff08\u7f16\u8bd1\u6e90\u7801\uff09: \u7f16\u8bd1\u9879\u76ee\u4e2d\u7684\u6e90\u4ee3\u7801\u6587\u4ef6\u3002\u8fd9\u662f\u6784\u5efa\u8fc7\u7a0b\u7684\u6838\u5fc3\u6b65\u9aa4\u4e4b\u4e00\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Link Binary With Libraries\uff08\u94fe\u63a5\u4e8c\u8fdb\u5236\u4e0e\u5e93\uff09: \u5c06\u7f16\u8bd1\u540e\u7684\u4ee3\u7801\u4e0e\u6240\u9700\u7684\u5e93\u6587\u4ef6\uff08\u5305\u62ec\u7cfb\u7edf\u5e93\u548c\u7b2c\u4e09\u65b9\u5e93\uff09\u94fe\u63a5\u8d77\u6765\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Copy Bundle Resources\uff08\u590d\u5236\u6346\u7ed1\u8d44\u6e90\uff09: \u5c06\u56fe\u7247\u3001\u97f3\u9891\u3001xib\u6587\u4ef6\u7b49\u8d44\u6e90\u590d\u5236\u5230\u6784\u5efa\u7684\u5e94\u7528\u7a0b\u5e8f\u5305\u4e2d\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Run Script\uff08\u8fd0\u884c\u811a\u672c\uff09: \u5728\u8fd9\u4e2a\u9636\u6bb5\uff0c\u4f60\u53ef\u4ee5\u6dfb\u52a0\u81ea\u5b9a\u4e49\u811a\u672c\u6765\u6267\u884c\u7279\u5b9a\u7684\u4efb\u52a1\uff0c\u5982\u4ee3\u7801\u7b7e\u540d\u3001\u4fee\u6539\u8d44\u6e90\u6587\u4ef6\u7b49\u3002\u8fd9\u4e9b\u811a\u672c\u53ef\u4ee5\u914d\u7f6e\u4e3a\u5728\u7f16\u8bd1\u524d\u6216\u7f16\u8bd1\u540e\u6267\u884c\u3002")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},"Post-actions\uff08\u6784\u5efa\u4e4b\u540e\uff09: \u548c Pre-actions \u7c7b\u4f3c\uff0c\u8fd9\u4e9b\u4e5f\u662f\u5728 scheme \u8bbe\u7f6e\u4e2d\u5b9a\u4e49\u7684\u52a8\u4f5c\uff0c\u5b83\u4eec\u5728\u6574\u4e2a\u6784\u5efa\u8fc7\u7a0b\u5b8c\u6210\u540e\u6267\u884c\u3002"))),(0,i.kt)("h2",{id:"\u603b\u7ed3"},"\u603b\u7ed3\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Build Phases \u5305\u62ec\u4e86\u5728\u6784\u5efa\u8fc7\u7a0b\u7684\u4e0d\u540c\u9636\u6bb5\u6267\u884c\u7684\u4efb\u52a1\uff0c\u5b83\u4eec\u6309\u7167\u7279\u5b9a\u7684\u987a\u5e8f\u8fdb\u884c\u3002"),(0,i.kt)("li",{parentName:"ul"},"Run Script \u9636\u6bb5\u7279\u522b\u7075\u6d3b\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u914d\u7f6e\u811a\u672c\u5728\u6784\u5efa\u4e4b\u524d\u6216\u6784\u5efa\u4e4b\u540e\u6267\u884c\uff0c\u53d6\u51b3\u4e8e\u5177\u4f53\u7684\u9700\u6c42\u3002"),(0,i.kt)("li",{parentName:"ul"},"\u901a\u8fc7\u5408\u7406\u914d\u7f6e\u548c\u4f7f\u7528 Build Phases\uff0c\u53ef\u4ee5\u6709\u6548\u5730\u63a7\u5236\u548c\u81ea\u52a8\u5316\u6784\u5efa\u8fc7\u7a0b\uff0c\u4f18\u5316\u5f00\u53d1\u6d41\u7a0b")))}d.isMDXComponent=!0}}]);