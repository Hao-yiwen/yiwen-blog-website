"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[7587],{3905:(e,n,t)=>{t.d(n,{Zo:()=>w,kt:()=>g});var o=t(67294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);n&&(o=o.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,o)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function a(e,n){if(null==e)return{};var t,o,r=function(e,n){if(null==e)return{};var t,o,r={},i=Object.keys(e);for(o=0;o<i.length;o++)t=i[o],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(o=0;o<i.length;o++)t=i[o],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var c=o.createContext({}),s=function(e){var n=o.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},w=function(e){var n=s(e.components);return o.createElement(c.Provider,{value:n},e.children)},u="mdxType",v={inlineCode:"code",wrapper:function(e){var n=e.children;return o.createElement(o.Fragment,{},n)}},p=o.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,c=e.parentName,w=a(e,["components","mdxType","originalType","parentName"]),u=s(t),p=r,g=u["".concat(c,".").concat(p)]||u[p]||v[p]||i;return t?o.createElement(g,l(l({ref:n},w),{},{components:t})):o.createElement(g,l({ref:n},w))}));function g(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,l=new Array(i);l[0]=p;var a={};for(var c in n)hasOwnProperty.call(n,c)&&(a[c]=n[c]);a.originalType=e,a[u]="string"==typeof e?e:r,l[1]=a;for(var s=2;s<i;s++)l[s]=t[s];return o.createElement.apply(null,l)}return o.createElement.apply(null,t)}p.displayName="MDXCreateElement"},49804:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>l,default:()=>v,frontMatter:()=>i,metadata:()=>a,toc:()=>s});var o=t(87462),r=(t(67294),t(3905));const i={},l="\u4ece\u4efb\u4f55\u5730\u65b9\u83b7\u53d6UIKit\u4e2dnavigationController\u7684\u529e\u6cd5",a={unversionedId:"native/ios/uikit/root_navigation",id:"native/ios/uikit/root_navigation",title:"\u4ece\u4efb\u4f55\u5730\u65b9\u83b7\u53d6UIKit\u4e2dnavigationController\u7684\u529e\u6cd5",description:"\u4eceoc\u548cswift\u83b7\u53d6navigationController\u7684\u65b9\u6cd5",source:"@site/docs/native/ios/uikit/root_navigation.md",sourceDirName:"native/ios/uikit",slug:"/native/ios/uikit/root_navigation",permalink:"/yiwen-blog-website/en/docs/native/ios/uikit/root_navigation",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/native/ios/uikit/root_navigation.md",tags:[],version:"current",frontMatter:{},sidebar:"nativeSidebar",previous:{title:"\u4f7f\u7528navigation\u5bfc\u822a\u65f6\u6ca1\u6709\u5bfc\u822a\u5934\u4f46\u662f\u80fd\u5de6\u6ed1",permalink:"/yiwen-blog-website/en/docs/native/ios/uikit/navigationcontroller"},next:{title:"\u5728interfacebuilder\u4e2d\u6dfb\u52a0UIScrollview",permalink:"/yiwen-blog-website/en/docs/native/ios/uikit/scrollview"}},c={},s=[{value:"\u4eceoc\u548cswift\u83b7\u53d6navigationController\u7684\u65b9\u6cd5",id:"\u4eceoc\u548cswift\u83b7\u53d6navigationcontroller\u7684\u65b9\u6cd5",level:2}],w={toc:s},u="wrapper";function v(e){let{components:n,...t}=e;return(0,r.kt)(u,(0,o.Z)({},w,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"\u4ece\u4efb\u4f55\u5730\u65b9\u83b7\u53d6uikit\u4e2dnavigationcontroller\u7684\u529e\u6cd5"},"\u4ece\u4efb\u4f55\u5730\u65b9\u83b7\u53d6UIKit\u4e2dnavigationController\u7684\u529e\u6cd5"),(0,r.kt)("h2",{id:"\u4eceoc\u548cswift\u83b7\u53d6navigationcontroller\u7684\u65b9\u6cd5"},"\u4eceoc\u548cswift\u83b7\u53d6navigationController\u7684\u65b9\u6cd5"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-swift"},"if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,\n    let window = windowScene.windows.first,\n    let rootViewController = window.rootViewController {\n    if let navigationController = rootViewController as? UINavigationController {\n        navigationController.popViewController(animated: true)\n    } else {\n        rootViewController.navigationController?.popViewController(animated: true)\n    }\n}\n")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-objectivec"},'UIWindow *window = nil;\nfor (UIWindowScene *windowScene in [UIApplication sharedApplication].connectedScenes) {\n    if (windowScene.activationState == UISceneActivationStateForegroundActive) {\n        window = windowScene.windows.firstObject;\n        break;\n    }\n}\n\nUIViewController *rootViewController = window.rootViewController;\n\nif ([rootViewController isKindOfClass:[UINavigationController class]]) {\n    UINavigationController *rootNavigationController = (UINavigationController *)rootViewController;\n    // \u521b\u5efa\u5e76\u8bbe\u7f6e WKWebViewScreenController\n    WKWebViewScreenController *webViewController = [[WKWebViewScreenController alloc] init];\n    webViewController.urlString = url;\n    \n    // \u4f7f\u7528 navigationController \u8fdb\u884c\u9875\u9762\u8df3\u8f6c\n    [rootNavigationController pushViewController:webViewController animated:YES];\n} else if (rootViewController.navigationController) {\n    // \u5982\u679c rootViewController \u4e0d\u662f UINavigationController\uff0c\u4f46\u6709 navigationController\n    UINavigationController *navigationController = rootViewController.navigationController;\n    WKWebViewScreenController *webViewController = [[WKWebViewScreenController alloc] init];\n    webViewController.urlString = url;\n    \n    [navigationController pushViewController:webViewController animated:YES];\n} else {\n    NSLog(@"Root view controller is not a navigation controller and has no navigation controller");\n}\n')))}v.isMDXComponent=!0}}]);