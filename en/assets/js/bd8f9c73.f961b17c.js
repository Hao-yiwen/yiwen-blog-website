"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[7851],{3905:(e,t,n)=>{n.d(t,{Zo:()=>c,kt:()=>d});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var p=r.createContext({}),s=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},c=function(e){var t=s(e.components);return r.createElement(p.Provider,{value:t},e.children)},u="mdxType",y={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,l=e.originalType,p=e.parentName,c=i(e,["components","mdxType","originalType","parentName"]),u=s(n),m=a,d=u["".concat(p,".").concat(m)]||u[m]||y[m]||l;return n?r.createElement(d,o(o({ref:t},c),{},{components:n})):r.createElement(d,o({ref:t},c))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=n.length,o=new Array(l);o[0]=m;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i[u]="string"==typeof e?e:a,o[1]=i;for(var s=2;s<l;s++)o[s]=n[s];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},5957:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>p,contentTitle:()=>o,default:()=>y,frontMatter:()=>l,metadata:()=>i,toc:()=>s});var r=n(7462),a=(n(7294),n(3905));const l={},o="By\u5173\u952e\u5b57",i={unversionedId:"native/kotlin/delagate",id:"native/kotlin/delagate",title:"By\u5173\u952e\u5b57",description:"\u5728Kotlin\u4e2d\uff0cby\u5173\u952e\u5b57\u4e3b\u8981\u7528\u4e8e\u4e24\u4e2a\u573a\u666f\uff1a\u59d4\u6258\u5c5e\u6027\uff08Property Delegation\uff09\u548c\u7c7b\u59d4\u6258\uff08Class Delegation\uff09\u3002\u8fd9\u91cc\u6211\u5c06\u63d0\u4f9b\u4e24\u4e2a\u793a\u4f8b\uff0c\u5206\u522b\u5c55\u793a\u8fd9\u4e24\u79cd\u7528\u6cd5\u3002",source:"@site/docs/native/kotlin/delagate.md",sourceDirName:"native/kotlin",slug:"/native/kotlin/delagate",permalink:"/yiwen-blog-website/en/docs/native/kotlin/delagate",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/native/kotlin/delagate.md",tags:[],version:"current",frontMatter:{},sidebar:"nativeSidebar",previous:{title:"kotlin\u4e2d\u7684\u534f\u6210",permalink:"/yiwen-blog-website/en/docs/native/kotlin/coroutines"},next:{title:"kotlin\u4e2d\u7684\u679a\u4e3e",permalink:"/yiwen-blog-website/en/docs/native/kotlin/enum"}},p={},s=[{value:"1. \u59d4\u6258\u5c5e\u6027\uff08Property Delegation\uff09",id:"1-\u59d4\u6258\u5c5e\u6027property-delegation",level:2},{value:"2.\u7c7b\u59d4\u6258\uff08Class Delegation\uff09",id:"2\u7c7b\u59d4\u6258class-delegation",level:2}],c={toc:s},u="wrapper";function y(e){let{components:t,...n}=e;return(0,a.kt)(u,(0,r.Z)({},c,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"by\u5173\u952e\u5b57"},"By\u5173\u952e\u5b57"),(0,a.kt)("p",null,"\u5728Kotlin\u4e2d\uff0cby\u5173\u952e\u5b57\u4e3b\u8981\u7528\u4e8e\u4e24\u4e2a\u573a\u666f\uff1a\u59d4\u6258\u5c5e\u6027\uff08Property Delegation\uff09\u548c\u7c7b\u59d4\u6258\uff08Class Delegation\uff09\u3002\u8fd9\u91cc\u6211\u5c06\u63d0\u4f9b\u4e24\u4e2a\u793a\u4f8b\uff0c\u5206\u522b\u5c55\u793a\u8fd9\u4e24\u79cd\u7528\u6cd5\u3002"),(0,a.kt)("h2",{id:"1-\u59d4\u6258\u5c5e\u6027property-delegation"},"1. \u59d4\u6258\u5c5e\u6027\uff08Property Delegation\uff09"),(0,a.kt)("p",null,"Kotlin\u7684\u59d4\u6258\u5c5e\u6027\u5141\u8bb8\u4f60\u5c06\u5c5e\u6027\u7684\u83b7\u53d6\uff08get\uff09\u548c\u8bbe\u7f6e\uff08set\uff09\u64cd\u4f5c\u59d4\u6258\u7ed9\u53e6\u4e00\u4e2a\u5bf9\u8c61\u3002\u8fd9\u5bf9\u4e8e\u5c06\u5c5e\u6027\u7684\u884c\u4e3a\u59d4\u6258\u7ed9\u6846\u67b6\u6216\u5e93\u4e2d\u7684\u901a\u7528\u4ee3\u7801\u975e\u5e38\u6709\u7528\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-kt"},'import kotlin.reflect.KProperty\n\nclass Delegate {\n    operator fun getValue(thisRef: Any?, property: KProperty<*>): String {\n        return "$thisRef, thank you for delegating \'${property.name}\' to me!"\n    }\n\n    operator fun setValue(thisRef: Any?, property: KProperty<*>, value: String) {\n        println("$value has been assigned to \'${property.name}\' in $thisRef.")\n    }\n}\n\nclass Example {\n    var p: String by Delegate()\n}\n\nfun main() {\n    val example = Example()\n    println(example.p) // \u8c03\u7528getValue()\n\n    example.p = "New value" // \u8c03\u7528setValue()\n}\n')),(0,a.kt)("h2",{id:"2\u7c7b\u59d4\u6258class-delegation"},"2.\u7c7b\u59d4\u6258\uff08Class Delegation\uff09"),(0,a.kt)("p",null,"Kotlin\u7684\u7c7b\u59d4\u6258\u662f\u4e00\u79cd\u8bbe\u8ba1\u6a21\u5f0f\u7684\u5b9e\u73b0\uff0c\u5b83\u5141\u8bb8\u4f60\u5c06\u4e00\u4e2a\u63a5\u53e3\u7684\u5b9e\u73b0\u59d4\u6258\u7ed9\u53e6\u4e00\u4e2a\u5bf9\u8c61\u3002\u8fd9\u662f\u4e00\u79cd\u907f\u514d\u7ee7\u627f\u7684\u65b9\u5f0f\uff0c\u53ef\u4ee5\u4f7f\u5f97\u4ee3\u7801\u66f4\u7075\u6d3b\u3001\u66f4\u6613\u4e8e\u7ef4\u62a4\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-kt"},"interface Base {\n    val number: Int\n    fun print()\n}\n\nclass BaseImpl(val x: Int) : Base {\n    override fun print() { print(x) }\n}\n\nclass Derived(b: Base) : Base by b\n\nfun main() {\n    val b = BaseImpl(10)\n    // \u5c06\u59d4\u6258\u7684\u5bf9\u8c61\u4f20\u8fdb\u6765 \u7136\u540e\u590d\u5236\u8be5\u5bf9\u8c61\u5176\u4ed6\u5c5e\u6027\u548c\u65b9\u6cd5\n    val b = classDelegate(10)\n    var x = Derived(b)\n    x.print()\n    println(x.number)\n}\n")),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u5982\u679c\u6211\u4eec\u60f3\u8ba9Derived\u7c7b\u4e5f\u5b9e\u73b0Base\u63a5\u53e3\uff0c\u4f46\u4e0d\u60f3\u5728Derived\u7c7b\u4e2d\u624b\u52a8\u5b9e\u73b0\u6240\u6709Base\u63a5\u53e3\u7684\u65b9\u6cd5\uff0c\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528\u59d4\u6258\uff1a"),(0,a.kt)("li",{parentName:"ol"},"\u8fd9\u6837\uff0cDerived\u7c7b\u7684\u6240\u6709Base\u63a5\u53e3\u65b9\u6cd5\u8c03\u7528\u90fd\u4f1a\u88ab\u59d4\u6258\u7ed9\u5bf9\u8c61b\u3002\u8fd9\u91cc\u662f\u5982\u4f55\u4f7f\u7528\u5b83\u7684\uff1a"),(0,a.kt)("li",{parentName:"ol"},"\u5728\u8fd9\u4e2a\u793a\u4f8b\u4e2d\uff0c\u5f53\u6211\u4eec\u8c03\u7528Derived(b).print()\u65f6\uff0c\u8fd9\u4e2a\u8c03\u7528\u88ab\u59d4\u6258\u7ed9\u4e86b\u5bf9\u8c61\uff0c\u5373BaseImpl\u7684\u5b9e\u4f8b\u3002\u56e0\u6b64\uff0c\u8f93\u51fa\u7684\u662f10\uff0c\u5373BaseImpl\u4e2dx\u7684\u503c\u3002")))}y.isMDXComponent=!0}}]);