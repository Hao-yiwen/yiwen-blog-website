"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[3070],{3905:(e,n,t)=>{t.d(n,{Zo:()=>p,kt:()=>g});var o=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);n&&(o=o.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,o)}return t}function a(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,o,r=function(e,n){if(null==e)return{};var t,o,r={},l=Object.keys(e);for(o=0;o<l.length;o++)t=l[o],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(o=0;o<l.length;o++)t=l[o],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var c=o.createContext({}),s=function(e){var n=o.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):a(a({},n),e)),t},p=function(e){var n=s(e.components);return o.createElement(c.Provider,{value:n},e.children)},d="mdxType",u={inlineCode:"code",wrapper:function(e){var n=e.children;return o.createElement(o.Fragment,{},n)}},m=o.forwardRef((function(e,n){var t=e.components,r=e.mdxType,l=e.originalType,c=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),d=s(t),m=r,g=d["".concat(c,".").concat(m)]||d[m]||u[m]||l;return t?o.createElement(g,a(a({ref:n},p),{},{components:t})):o.createElement(g,a({ref:n},p))}));function g(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var l=t.length,a=new Array(l);a[0]=m;var i={};for(var c in n)hasOwnProperty.call(n,c)&&(i[c]=n[c]);i.originalType=e,i[d]="string"==typeof e?e:r,a[1]=i;for(var s=2;s<l;s++)a[s]=t[s];return o.createElement.apply(null,a)}return o.createElement.apply(null,t)}m.displayName="MDXCreateElement"},46:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>a,default:()=>u,frontMatter:()=>l,metadata:()=>i,toc:()=>s});var o=t(7462),r=(t(7294),t(3905));const l={sidebar_position:3},a="MongoDB",i={unversionedId:"backend/mongodb",id:"backend/mongodb",title:"MongoDB",description:"MongoDB \u662f\u4e00\u4e2a\u6d41\u884c\u7684\u5f00\u6e90 NoSQL \u6570\u636e\u5e93\uff0c\u7531 MongoDB Inc. \u5f00\u53d1\u3002\u5b83\u662f\u4e00\u4e2a\u6587\u6863\u6570\u636e\u5e93\uff0c\u610f\u5473\u7740\u5b83\u4ee5\u6587\u6863\u7684\u5f62\u5f0f\u5b58\u50a8\u6570\u636e\uff0c\u8fd9\u4e9b\u6587\u6863\u7684\u683c\u5f0f\u7c7b\u4f3c\u4e8e JSON\u3002",source:"@site/docs/backend/mongodb.md",sourceDirName:"backend",slug:"/backend/mongodb",permalink:"/yiwen-blog-website/docs/backend/mongodb",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/backend/mongodb.md",tags:[],version:"current",sidebarPosition:3,frontMatter:{sidebar_position:3},sidebar:"backendSidebar",previous:{title:"\u4e3a\u4ec0\u4e48\u4f20\u8f93\u6570\u636e\u9700\u8981\u5e8f\u5217\u5316\uff0c\u800c\u4e0d\u80fd\u76f4\u63a5\u4f20\u8f93\uff1f",permalink:"/yiwen-blog-website/docs/backend/java/whysequence"},next:{title:"Mysql",permalink:"/yiwen-blog-website/docs/category/mysql"}},c={},s=[{value:"\u4f18\u70b9",id:"\u4f18\u70b9",level:2},{value:"\u6587\u6863\u5bfc\u5411",id:"\u6587\u6863\u5bfc\u5411",level:3},{value:"\u67e5\u8be2\u8bed\u8a00",id:"\u67e5\u8be2\u8bed\u8a00",level:3},{value:"\u7d22\u5f15",id:"\u7d22\u5f15",level:3},{value:"\u5e94\u7528\u573a\u666f",id:"\u5e94\u7528\u573a\u666f",level:3},{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u4f7f\u7528",id:"\u4f7f\u7528",level:2},{value:"\u548cmysql\u7684\u6bd4\u8f83",id:"\u548cmysql\u7684\u6bd4\u8f83",level:2},{value:"mongoDB",id:"mongodb-1",level:3},{value:"mysql",id:"mysql",level:3}],p={toc:s},d="wrapper";function u(e){let{components:n,...t}=e;return(0,r.kt)(d,(0,o.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"mongodb"},"MongoDB"),(0,r.kt)("p",null,"MongoDB \u662f\u4e00\u4e2a\u6d41\u884c\u7684\u5f00\u6e90 NoSQL \u6570\u636e\u5e93\uff0c\u7531 MongoDB Inc. \u5f00\u53d1\u3002\u5b83\u662f\u4e00\u4e2a\u6587\u6863\u6570\u636e\u5e93\uff0c\u610f\u5473\u7740\u5b83\u4ee5\u6587\u6863\u7684\u5f62\u5f0f\u5b58\u50a8\u6570\u636e\uff0c\u8fd9\u4e9b\u6587\u6863\u7684\u683c\u5f0f\u7c7b\u4f3c\u4e8e JSON\u3002"),(0,r.kt)("p",null,"MongoDB \u5728\u53ef\u6269\u5c55\u6027\u3001\u6027\u80fd\u548c\u7075\u6d3b\u6027\u65b9\u9762\u8868\u73b0\u51fa\u8272\uff0c\u662f\u5f00\u53d1\u73b0\u4ee3\u5e94\u7528\u7a0b\u5e8f\u7684\u70ed\u95e8\u9009\u62e9\u4e4b\u4e00\u3002\u4ee5\u4e0b\u662f MongoDB \u7684\u4e00\u4e9b\u5173\u952e\u7279\u70b9\uff1a"),(0,r.kt)("h2",{id:"\u4f18\u70b9"},"\u4f18\u70b9"),(0,r.kt)("h3",{id:"\u6587\u6863\u5bfc\u5411"},"\u6587\u6863\u5bfc\u5411"),(0,r.kt)("p",null,"MongoDB \u5b58\u50a8\u7684\u6570\u636e\u5355\u5143\u662f\u6587\u6863\uff0c\u8fd9\u4e9b\u6587\u6863\u7ec4\u7ec7\u6210\u96c6\u5408\u3002\u6587\u6863\u7531\u5b57\u6bb5\uff08key\uff09\u548c\u503c\uff08value\uff09\u5bf9\u7ec4\u6210\uff0c\u7c7b\u4f3c\u4e8e JSON \u5bf9\u8c61\u3002"),(0,r.kt)("p",null,"\u6587\u6863\u53ef\u4ee5\u5305\u542b\u4e0d\u540c\u7c7b\u578b\u7684\u6570\u636e\uff0c\u5982\u5b57\u7b26\u4e32\u3001\u6570\u5b57\u3001\u5e03\u5c14\u503c\u3001\u6570\u7ec4\uff0c\u751a\u81f3\u5d4c\u5957\u6587\u6863\u3002"),(0,r.kt)("h3",{id:"\u67e5\u8be2\u8bed\u8a00"},"\u67e5\u8be2\u8bed\u8a00"),(0,r.kt)("p",null,"MongoDB \u63d0\u4f9b\u4e86\u5f3a\u5927\u7684\u67e5\u8be2\u8bed\u8a00\uff0c\u5141\u8bb8\u60a8\u6267\u884c\u5404\u79cd\u590d\u6742\u7684\u67e5\u8be2\u64cd\u4f5c\uff0c\u5305\u62ec\u6587\u6863\u5b57\u6bb5\u7684\u8fc7\u6ee4\u3001\u6587\u6863\u7684\u6392\u5e8f\u548c\u9650\u5236\u8fd4\u56de\u7ed3\u679c\u6570\u91cf\u7b49\u3002"),(0,r.kt)("p",null,"\u5b83\u8fd8\u652f\u6301\u805a\u5408\u64cd\u4f5c\uff0c\u5141\u8bb8\u60a8\u8fdb\u884c\u6570\u636e\u5904\u7406\u548c\u5206\u6790\u3002"),(0,r.kt)("h3",{id:"\u7d22\u5f15"},"\u7d22\u5f15"),(0,r.kt)("p",null,"\u4e3a\u4e86\u63d0\u9ad8\u67e5\u8be2\u6548\u7387\uff0cMongoDB \u652f\u6301\u5bf9\u6587\u6863\u4e2d\u7684\u5b57\u6bb5\u5efa\u7acb\u7d22\u5f15\u3002"),(0,r.kt)("p",null,"\u7d22\u5f15\u53ef\u4ee5\u663e\u8457\u63d0\u9ad8\u67e5\u8be2\u6027\u80fd\uff0c\u7279\u522b\u662f\u5728\u5904\u7406\u5927\u91cf\u6570\u636e\u65f6\u3002"),(0,r.kt)("h3",{id:"\u5e94\u7528\u573a\u666f"},"\u5e94\u7528\u573a\u666f"),(0,r.kt)("p",null,"MongoDB \u9002\u5408\u9700\u8981\u5904\u7406\u5927\u91cf\u6570\u636e\u4e14\u6570\u636e\u7ed3\u6784\u591a\u53d8\u7684\u5e94\u7528\u7a0b\u5e8f\uff0c\u5982\u5185\u5bb9\u7ba1\u7406\u7cfb\u7edf\u3001\u7535\u5b50\u5546\u52a1\u7f51\u7ad9\u3001\u6570\u636e\u4ed3\u5e93\u548c\u5927\u6570\u636e\u5e94\u7528\u3002"),(0,r.kt)("p",null,"MongoDB \u7684\u8fd9\u4e9b\u7279\u6027\u4f7f\u5b83\u6210\u4e3a\u5f53\u4eca\u5e94\u7528\u7a0b\u5e8f\u5f00\u53d1\u4e2d\u5e7f\u6cdb\u4f7f\u7528\u7684\u6570\u636e\u5e93\u4e4b\u4e00\uff0c\u7279\u522b\u662f\u5728\u9700\u8981\u5feb\u901f\u8fed\u4ee3\u548c\u5904\u7406\u975e\u7ed3\u6784\u5316\u6216\u534a\u7ed3\u6784\u5316\u6570\u636e\u7684\u573a\u666f\u4e2d\u3002"),(0,r.kt)("h2",{id:"\u5b89\u88c5"},"\u5b89\u88c5"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},(0,r.kt)("a",{parentName:"li",href:"https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-red-hat/"},"mongoDB\u5728Centos\u5b89\u88c5")),(0,r.kt)("li",{parentName:"ol"},"\u4f7f\u7528navicat\u94fe\u63a5\u4f7f\u7528")),(0,r.kt)("h2",{id:"\u4f7f\u7528"},"\u4f7f\u7528"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-js"},"const express = require('express');\nconst config = require('./config.json');\nconst {MongoClient} = require('mongodb');\nconst app = express();\napp.use(express.json());\n\nconst url = config.mongodb.url;\nconst dbName = config.mongodb.dbName;\n\nconst client = new MongoClient(url, { useUnifiedTopology: true });\n\napp.get('/', (req, res) => {\n    res.send('Hello, World!');\n});\n\napp.post('/add', async (req, res) => {\n    const {name, age} = req.body;\n    try {\n        await client.connect();\n        const db = client.db(dbName);\n        const collection = db.collection('users');\n        const result = await collection.insertOne({name, age});\n        res.json(result);\n    } catch (err) {\n        console.log(err);\n    }\n});\n\napp.get('/users', async (req, res) => {\n    try {\n        await client.connect();\n        const db = client.db(dbName);\n        const collection = db.collection('users');\n        const result = await collection.find({}).toArray();\n        res.json(result);\n    } catch (err) {\n        console.log(err);\n    }\n});\n\napp.get('/users/age', async (req, res) => {\n    const {age} = req.query;\n    try {\n        await client.connect();\n        const db = client.db(dbName);\n        const collection = db.collection('users');\n        const result = await collection.find({ age: { $gt: 25 } }).toArray();\n        res.json(result);\n    } catch (err) {\n        console.log(err);\n    }\n});\n\nconst PORT = 3000;\n\napp.listen(PORT, () => {\n    console.log(`Server is running on http://localhost:${PORT}`);\n});\n")),(0,r.kt)("h2",{id:"\u548cmysql\u7684\u6bd4\u8f83"},"\u548cmysql\u7684\u6bd4\u8f83"),(0,r.kt)("h3",{id:"mongodb-1"},"mongoDB"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"\u7075\u6d3b\u7684\u6570\u636e\u6a21\u578b\uff1aMongoDB \u4f7f\u7528\u6587\u6863\u578b\u6570\u636e\u6a21\u578b\uff0c\u65e0\u9700\u9884\u5b9a\u4e49\u6a21\u5f0f\uff0c\u9002\u5408\u5feb\u901f\u8fed\u4ee3\u548c\u5904\u7406\u591a\u53d8\u7684\u6570\u636e\u7ed3\u6784\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u6c34\u5e73\u6269\u5c55\u6027\uff1a\u901a\u8fc7\u5206\u7247\uff08Sharding\uff09\uff0cMongoDB \u652f\u6301\u5927\u89c4\u6a21\u7684\u6c34\u5e73\u6269\u5c55\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u9ad8\u6548\u7684\u5904\u7406\u5927\u91cf\u6570\u636e\uff1a\u9002\u5408\u5927\u6570\u636e\u5e94\u7528\u548c\u5b9e\u65f6\u5206\u6790\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u5f3a\u5927\u7684\u67e5\u8be2\u8bed\u8a00\uff1a\u652f\u6301\u590d\u6742\u7684\u67e5\u8be2\u548c\u805a\u5408\u64cd\u4f5c\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u9ad8\u53ef\u7528\u6027\u548c\u81ea\u52a8\u6545\u969c\u8f6c\u79fb\uff1a\u901a\u8fc7\u590d\u5236\u96c6\u5b9e\u73b0\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u9002\u5408\u975e\u7ed3\u6784\u5316\u548c\u534a\u7ed3\u6784\u5316\u6570\u636e\uff1a\u5982 JSON \u6216 BSON \u683c\u5f0f\u3002")),(0,r.kt)("h3",{id:"mysql"},"mysql"),(0,r.kt)("ol",null,(0,r.kt)("li",{parentName:"ol"},"\u6210\u719f\u7684\u6280\u672f\uff1a\u4f5c\u4e3a\u5173\u7cfb\u578b\u6570\u636e\u5e93\uff0cMySQL \u6709\u7740\u591a\u5e74\u7684\u53d1\u5c55\u548c\u5e7f\u6cdb\u7684\u5e94\u7528\uff0c\u6280\u672f\u6210\u719f\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u4e25\u683c\u7684\u6570\u636e\u6a21\u5f0f\uff1a\u6709\u52a9\u4e8e\u786e\u4fdd\u6570\u636e\u7684\u5b8c\u6574\u6027\u548c\u4e00\u81f4\u6027\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u5f3a\u5927\u7684\u4e8b\u52a1\u652f\u6301\uff1a\u652f\u6301\u590d\u6742\u7684\u4e8b\u52a1\u5904\u7406\u548c\u9501\u5b9a\u673a\u5236\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u9ad8\u6548\u7684 JOIN \u64cd\u4f5c\uff1a\u5728\u5904\u7406\u5173\u7cfb\u578b\u6570\u636e\u65f6\u8868\u73b0\u51fa\u8272\u3002"),(0,r.kt)("li",{parentName:"ol"},"\u5e7f\u6cdb\u7684\u793e\u533a\u548c\u5de5\u5177\u652f\u6301\uff1a\u6709\u5927\u91cf\u7684\u5de5\u5177\u548c\u793e\u533a\u652f\u6301\u3002")))}u.isMDXComponent=!0}}]);