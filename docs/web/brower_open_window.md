# 浏览器打开新窗口

```js
const windowWidth = 1100; // 新窗口的宽度
const windowHeight = 800; // 新窗口的高度
const windowLeft = (window.screen.width - windowWidth) / 2;
const windowTop = (window.screen.height - windowHeight) / 2;

const windowFeatures = `width=${windowWidth},height=${windowHeight},left=${windowLeft},top=${windowTop},resizable=yes,scrollbars=yes,status=yes`;

window.open(tar.href, '文档窗口', windowFeatures);
```