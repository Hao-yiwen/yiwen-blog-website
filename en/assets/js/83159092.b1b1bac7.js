"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[6821],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>f});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var i=r.createContext({}),s=function(e){var t=r.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=s(e.components);return r.createElement(i.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},b=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,i=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),u=s(n),b=a,f=u["".concat(i,".").concat(b)]||u[b]||m[b]||o;return n?r.createElement(f,l(l({ref:t},p),{},{components:n})):r.createElement(f,l({ref:t},p))}));function f(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,l=new Array(o);l[0]=b;var c={};for(var i in t)hasOwnProperty.call(t,i)&&(c[i]=t[i]);c.originalType=e,c[u]="string"==typeof e?e:a,l[1]=c;for(var s=2;s<o;s++)l[s]=n[s];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}b.displayName="MDXCreateElement"},57673:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>i,contentTitle:()=>l,default:()=>m,frontMatter:()=>o,metadata:()=>c,toc:()=>s});var r=n(87462),a=(n(67294),n(3905));const o={},l="c/oc/java/js\u7684\u5185\u5b58\u7ba1\u7406\u673a\u5236",c={unversionedId:"cs_base/memory_management",id:"cs_base/memory_management",title:"c/oc/java/js\u7684\u5185\u5b58\u7ba1\u7406\u673a\u5236",description:"C \u8bed\u8a00\u4f7f\u7528\u624b\u52a8\u7ba1\u7406",source:"@site/docs/cs_base/memory_management.md",sourceDirName:"cs_base",slug:"/cs_base/memory_management",permalink:"/yiwen-blog-website/en/docs/cs_base/memory_management",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/cs_base/memory_management.md",tags:[],version:"current",frontMatter:{},sidebar:"csSidebar",previous:{title:"\u603b\u7ed3",permalink:"/yiwen-blog-website/en/docs/cs_base/web_pattern/override"}},i={},s=[{value:"C \u8bed\u8a00\u4f7f\u7528\u624b\u52a8\u7ba1\u7406",id:"c-\u8bed\u8a00\u4f7f\u7528\u624b\u52a8\u7ba1\u7406",level:2},{value:"oc\u4f7f\u7528\u5f15\u7528\u8ba1\u6570",id:"oc\u4f7f\u7528\u5f15\u7528\u8ba1\u6570",level:2},{value:"Java \u7684\u5185\u5b58\u7ba1\u7406",id:"java-\u7684\u5185\u5b58\u7ba1\u7406",level:2},{value:"js\u5185\u5b58\u7ba1\u7406",id:"js\u5185\u5b58\u7ba1\u7406",level:2}],p={toc:s},u="wrapper";function m(e){let{components:t,...n}=e;return(0,a.kt)(u,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"cocjavajs\u7684\u5185\u5b58\u7ba1\u7406\u673a\u5236"},"c/oc/java/js\u7684\u5185\u5b58\u7ba1\u7406\u673a\u5236"),(0,a.kt)("h2",{id:"c-\u8bed\u8a00\u4f7f\u7528\u624b\u52a8\u7ba1\u7406"},"C \u8bed\u8a00\u4f7f\u7528\u624b\u52a8\u7ba1\u7406"),(0,a.kt)("p",null,"\u5728 C \u8bed\u8a00\u4e2d\uff0c\u5185\u5b58\u7ba1\u7406\u662f\u624b\u52a8\u7684\u3002\u8fd9\u610f\u5473\u7740\u7a0b\u5e8f\u5458\u9700\u8981\u81ea\u5df1\u5206\u914d\u548c\u91ca\u653e\u5185\u5b58\u3002\u5e38\u7528\u7684\u5185\u5b58\u7ba1\u7406\u51fd\u6570\u5305\u62ec malloc\u3001calloc\u3001realloc \u548c free\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},'int *p = (int *)malloc(sizeof(int) * 10); // \u5206\u914d\u5185\u5b58\nif (p == NULL) {\n    fprintf(stderr, "Memory allocation failed\\n");\n    return 1;\n}\n')),(0,a.kt)("h2",{id:"oc\u4f7f\u7528\u5f15\u7528\u8ba1\u6570"},"oc\u4f7f\u7528\u5f15\u7528\u8ba1\u6570"),(0,a.kt)("p",null,"Objective-C \u4f7f\u7528\u5f15\u7528\u8ba1\u6570\uff08Reference Counting\uff09\u6765\u7ba1\u7406\u5185\u5b58\u3002\u5728\u624b\u52a8\u5f15\u7528\u8ba1\u6570\uff08MRC\uff09\u4e0b\uff0c\u5f00\u53d1\u8005\u9700\u8981\u663e\u5f0f\u5730\u8c03\u7528 retain \u548c release \u65b9\u6cd5\u6765\u7ba1\u7406\u5bf9\u8c61\u7684\u751f\u547d\u5468\u671f\u3002\u73b0\u5728\uff0cObjective-C \u5927\u591a\u4f7f\u7528\u81ea\u52a8\u5f15\u7528\u8ba1\u6570\uff08ARC\uff09\uff0c\u7531\u7f16\u8bd1\u5668\u81ea\u52a8\u63d2\u5165 retain \u548c release \u8c03\u7528\uff0c\u7b80\u5316\u4e86\u5185\u5b58\u7ba1\u7406\u3002"),(0,a.kt)("p",null,"\u81ea\u52a8\u5f15\u7528\u8ba1\u6570\uff08ARC\uff09"),(0,a.kt)("p",null,"\u5728 ARC \u4e0b\uff0c\u7f16\u8bd1\u5668\u81ea\u52a8\u5904\u7406\u5f15\u7528\u8ba1\u6570\uff0c\u5f00\u53d1\u8005\u65e0\u9700\u624b\u52a8\u8c03\u7528 retain \u548c release\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"@autoreleasepool {\n    NSString *str = [[NSString alloc] init]; // \u7f16\u8bd1\u5668\u81ea\u52a8\u63d2\u5165 retain \u548c release\n}\n")),(0,a.kt)("h2",{id:"java-\u7684\u5185\u5b58\u7ba1\u7406"},"Java \u7684\u5185\u5b58\u7ba1\u7406"),(0,a.kt)("p",null,"Java \u4f7f\u7528\u5783\u573e\u56de\u6536\uff08Garbage Collection, GC\uff09\u673a\u5236\u6765\u7ba1\u7406\u5185\u5b58\u3002\u5783\u573e\u56de\u6536\u5668\u8d1f\u8d23\u81ea\u52a8\u56de\u6536\u4e0d\u518d\u4f7f\u7528\u7684\u5bf9\u8c61\uff0c\u5f00\u53d1\u8005\u4e0d\u9700\u8981\u663e\u5f0f\u5730\u91ca\u653e\u5185\u5b58\u3002Java \u7684\u5783\u573e\u56de\u6536\u7b97\u6cd5\u901a\u5e38\u662f\u6807\u8bb0-\u6e05\u9664\uff08Mark-and-Sweep\uff09\u6216\u5176\u6539\u8fdb\u7248\u672c\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-java"},'public class Main {\n    public static void main(String[] args) {\n        String str = new String("Hello, World!");\n        // \u5783\u573e\u56de\u6536\u5668\u81ea\u52a8\u56de\u6536\u4e0d\u518d\u4f7f\u7528\u7684\u5bf9\u8c61\n    }\n}\n')),(0,a.kt)("h2",{id:"js\u5185\u5b58\u7ba1\u7406"},"js\u5185\u5b58\u7ba1\u7406"),(0,a.kt)("p",null,"JavaScript \u7684\u5185\u5b58\u7ba1\u7406\u4e3b\u8981\u901a\u8fc7\u5783\u573e\u56de\u6536\u673a\u5236\uff08Garbage Collection, GC\uff09\u6765\u81ea\u52a8\u7ba1\u7406\u5185\u5b58\u3002\u5f00\u53d1\u8005\u4e0d\u9700\u8981\u663e\u5f0f\u5730\u5206\u914d\u548c\u91ca\u653e\u5185\u5b58\u3002"),(0,a.kt)("p",null,"JavaScript \u901a\u8fc7\u5783\u573e\u56de\u6536\u673a\u5236\u81ea\u52a8\u7ba1\u7406\u5185\u5b58\uff0c\u4e3b\u8981\u4f7f\u7528\u6807\u8bb0-\u6e05\u9664\u7b97\u6cd5\u3002\u5c3d\u7ba1\u5f00\u53d1\u8005\u4e0d\u9700\u8981\u624b\u52a8\u7ba1\u7406\u5185\u5b58\uff0c\u4f46\u4ecd\u9700\u6ce8\u610f\u907f\u514d\u5185\u5b58\u6cc4\u6f0f\u3002\u901a\u8fc7\u826f\u597d\u7684\u7f16\u7801\u5b9e\u8df5\uff0c\u53ef\u4ee5\u786e\u4fdd JavaScript \u5e94\u7528\u9ad8\u6548\u8fd0\u884c\uff0c\u5185\u5b58\u4f7f\u7528\u5408\u7406\u3002"))}m.isMDXComponent=!0}}]);