"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[8017],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>h});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),i=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=i(e.components);return r.createElement(c.Provider,{value:t},e.children)},u="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=i(n),d=a,h=u["".concat(c,".").concat(d)]||u[d]||m[d]||o;return n?r.createElement(h,l(l({ref:t},p),{},{components:n})):r.createElement(h,l({ref:t},p))}));function h(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,l=new Array(o);l[0]=d;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s[u]="string"==typeof e?e:a,l[1]=s;for(var i=2;i<o;i++)l[i]=n[i];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},26933:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>l,default:()=>m,frontMatter:()=>o,metadata:()=>s,toc:()=>i});var r=n(87462),a=(n(67294),n(3905));const o={title:"Centos\u4ee3\u7406"},l="Centos\u4ee3\u7406",s={permalink:"/yiwen-blog-website/en/blog/linux_clash",editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/blog/linux_clash.md",source:"@site/blog/linux_clash.md",title:"Centos\u4ee3\u7406",description:"\u5728\u670d\u52a1\u5668\u4f7f\u7528\u8fc7\u7a0b\u4e2d\u6211\u4eec\u4f1a\u7ecf\u5e38\u53d1\u73b0\u5404\u7c7b\u95ee\u9898\uff0c\u4f46\u662f\u8fd9\u7c7b\u95ee\u9898\u90fd\u6709\u4e00\u4e2a\u516c\u5171\u95ee\u9898\uff0c\u5c31\u662f\u5f88\u591a\u65f6\u5019\u6ca1\u6cd5\u6b63\u5e38\u8bbf\u95eegithub\u8d44\u6e90\u6216\u8005google\u8d44\u6e90\uff0c\u4ece\u800c\u5bfc\u81f4\u90e8\u7f72\u5361\u58f3\u3002\u90a3\u4e48\u5982\u4f55\u5904\u7406\u6b64\u7c7b\u95ee\u9898\u5c31\u663e\u5f97\u5c24\u4e3a\u91cd\u8981\uff0c\u5728\u8fd9\u91cc\u6211\u4ecb\u7ecd\u4e00\u79cd\u6211\u4f7f\u7528\u7684clash + centos\u7684\u89e3\u51b3\u65b9\u6848\u3002",date:"2025-01-12T17:04:42.000Z",formattedDate:"January 12, 2025",tags:[],readingTime:2.1,hasTruncateMarker:!1,authors:[],frontMatter:{title:"Centos\u4ee3\u7406"},prevItem:{title:"githubPackages\u4e0a\u4f20\u8e29\u5751",permalink:"/yiwen-blog-website/en/blog/githubpackages"},nextItem:{title:"oh-my-zsh\u4ee3\u7801\u63d0\u793a\u548c\u8865\u5168",permalink:"/yiwen-blog-website/en/blog/mac_use"}},c={authorsImageUrls:[]},i=[{value:"clash\u4e0b\u8f7d",id:"clash\u4e0b\u8f7d",level:2},{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u9b54\u6cd5\u4f7f\u7528\u5b8c\u6bd5",id:"\u9b54\u6cd5\u4f7f\u7528\u5b8c\u6bd5",level:2},{value:"docker\u4e2d\u4f7f\u7528\u4ee3\u7406",id:"docker\u4e2d\u4f7f\u7528\u4ee3\u7406",level:2},{value:"\u89e3\u51b3\u65b9\u6848",id:"\u89e3\u51b3\u65b9\u6848",level:3}],p={toc:i},u="wrapper";function m(e){let{components:t,...n}=e;return(0,a.kt)(u,(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"\u5728\u670d\u52a1\u5668\u4f7f\u7528\u8fc7\u7a0b\u4e2d\u6211\u4eec\u4f1a\u7ecf\u5e38\u53d1\u73b0\u5404\u7c7b\u95ee\u9898\uff0c\u4f46\u662f\u8fd9\u7c7b\u95ee\u9898\u90fd\u6709\u4e00\u4e2a\u516c\u5171\u95ee\u9898\uff0c\u5c31\u662f\u5f88\u591a\u65f6\u5019\u6ca1\u6cd5\u6b63\u5e38\u8bbf\u95ee",(0,a.kt)("inlineCode",{parentName:"p"},"github"),"\u8d44\u6e90\u6216\u8005",(0,a.kt)("inlineCode",{parentName:"p"},"google"),"\u8d44\u6e90\uff0c\u4ece\u800c\u5bfc\u81f4\u90e8\u7f72\u5361\u58f3\u3002\u90a3\u4e48\u5982\u4f55\u5904\u7406\u6b64\u7c7b\u95ee\u9898\u5c31\u663e\u5f97\u5c24\u4e3a\u91cd\u8981\uff0c\u5728\u8fd9\u91cc\u6211\u4ecb\u7ecd\u4e00\u79cd\u6211\u4f7f\u7528\u7684",(0,a.kt)("inlineCode",{parentName:"p"},"clash + centos"),"\u7684\u89e3\u51b3\u65b9\u6848\u3002"),(0,a.kt)("h2",{id:"clash\u4e0b\u8f7d"},"clash\u4e0b\u8f7d"),(0,a.kt)("admonition",{type:"tip"},(0,a.kt)("p",{parentName:"admonition"},"\u5b89\u88c5 ",(0,a.kt)("inlineCode",{parentName:"p"},"clash-linux-amd64-latest.gz"))),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://drive.google.com/drive/folders/1mhKMWAcS5661t_TWSp9wm4WNj32NFbZK"},"\u8c37\u6b4c\u4e91\u76d8\u4e0b\u8f7d")),(0,a.kt)("h2",{id:"\u5b89\u88c5"},"\u5b89\u88c5"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"# \u89e3\u538b\u538b\u7f29\u6587\u4ef6\uff0c\u4f1a\u751f\u6210\u4e00\u4e2a\u6ca1\u6709 .gz \u7684\u540c\u540d\u6587\u4ef6\ngzip -d clash-linux-amd64-latest.gz\n\n# \uff08\u53ef\u9009\u9879\uff09\u4fee\u6539\u7a0b\u5e8f\u6587\u4ef6\u540d\nmv clash-linux-amd64-latest.gz clash\n\n# \u6dfb\u52a0\u8fd0\u884c\u6743\u9650\nchmod +x clash\n\n# \u5148\u8fd0\u884c\u670d\u52a1\n# \u5728\u8fd0\u884c\u94b1\u8bf7\u5c06\u81ea\u5df1\u7684 config.yaml \u590d\u5236\u5230 /root/.config/clash \u4e0b\u9762\n# \u5982\u679c\u6709Country.mmdb\u62a5\u9519 \u90a3\u53ef\u80fd\u9700\u8981 wget -O Country.mmdb https://www.sub-speeder.com/client-download/Country.mmdb \u89e3\u51b3\n./clash\n\n//\u6dfb\u52a0\u5b88\u62a4\u8fdb\u7a0b\ncp clash /usr/local/bin\n\n// \u6dfb\u52a0\u914d\u7f6e\nvim /etc/systemd/system/clash.service\n\n# start\n[Unit]\nDescription=Clash daemon, A rule-based proxy in Go.\nAfter=network.target\n\n[Service]\nType=simple\nRestart=always\nExecStart=/usr/local/bin/clash -d /root/.config/clash\n\n[Install]\nWantedBy=multi-user.target\n# end\n\n# \u7cfb\u7edf\u542f\u52a8\u65f6\u8fd0\u884c\nsystemctl enable clash\n\n# \u7acb\u5373\u8fd0\u884c clash \u670d\u52a1\nsystemctl start clash\n\n# \u67e5\u770b clash \u670d\u52a1\u8fd0\u884c\u72b6\u6001\nsystemctl status clash\n\n# \u67e5\u770b\u8fd0\u884c\u65e5\u5fd7\njournalctl -xe\n\n# \u4f7f\u7528\u65f6\u4e34\u65f6\u4fee\u6539\nexport http_proxy=http://127.0.0.1:7890\nexport https_proxy=http://127.0.0.1:7890\n\n# \u6d4b\u8bd5\ncurl https://www.google.com\n")),(0,a.kt)("h2",{id:"\u9b54\u6cd5\u4f7f\u7528\u5b8c\u6bd5"},"\u9b54\u6cd5\u4f7f\u7528\u5b8c\u6bd5"),(0,a.kt)("h2",{id:"docker\u4e2d\u4f7f\u7528\u4ee3\u7406"},"docker\u4e2d\u4f7f\u7528\u4ee3\u7406"),(0,a.kt)("p",null,"docker hub\u5728\u56fd\u5185\u4f7f\u7528\u7684\u65f6\u5019\u4f1a\u6709\u51e0\u7387\u51fa\u73b0\u94fe\u63a5\u8d85\u65f6\u95ee\u9898\uff0c\u8fd9\u65f6\u9700\u8981\u7ed9docker\u8bbe\u7f6e\u4ee3\u7406\u6765\u89e3\u51b3\u8fd9\u4e2a\u95ee\u9898\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"Get https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)\n")),(0,a.kt)("h3",{id:"\u89e3\u51b3\u65b9\u6848"},"\u89e3\u51b3\u65b9\u6848"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://stackoverflow.com/questions/48056365/error-get-https-registry-1-docker-io-v2-net-http-request-canceled-while-b"},"https://stackoverflow.com/questions/48056365/error-get-https-registry-1-docker-io-v2-net-http-request-canceled-while-b")),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://docs.docker.com/engine/daemon/proxy/"},"https://docs.docker.com/engine/daemon/proxy/")),(0,a.kt)("p",null,"\u4ece\u4e24\u8005\u6765\u770b\u9700\u8981\u624b\u52a8\u8bbe\u7f6e\u4ee3\u7406"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"\u521b\u5efa\u6216\u8005\u7f16\u8f91",(0,a.kt)("inlineCode",{parentName:"li"},"/etc/systemd/system/docker.service.d/http-proxy.conf"))),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},'[Service]\nEnvironment="HTTP_PROXY=http://127.0.0.1:7890"\nEnvironment="HTTPS_PROXY=http://127.0.0.1:7890"\n')),(0,a.kt)("ol",{start:2},(0,a.kt)("li",{parentName:"ol"},"\u91cd\u542f")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bash"},"sudo systemctl daemon-reload                            \nsudo systemctl restart docker\n")))}m.isMDXComponent=!0}}]);