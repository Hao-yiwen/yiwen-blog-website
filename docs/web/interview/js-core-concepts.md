---
title: JavaScript 核心概念
sidebar_position: 1
tags: [javascript, interview]
---

# 模块一：JavaScript 核心 — 问题清单

## 数据类型 & 类型判断

### 1. JS 有哪些数据类型？基本类型和引用类型的区别？

**基本类型（7种）：** `string`、`number`、`boolean`、`undefined`、`null`、`symbol`、`bigint`

**引用类型：** `Object`（包括 Array、Function、Date、RegExp、Map、Set 等）

**核心区别：**

| 对比项 | 基本类型 | 引用类型 |
|--------|---------|---------|
| 存储位置 | 栈内存 | 堆内存（栈中存引用地址） |
| 赋值行为 | 值拷贝 | 引用拷贝（共享同一对象） |
| 比较方式 | 值比较 | 引用地址比较 |
| 不可变性 | 不可变（immutable） | 可变（mutable） |

```js
// 基本类型 — 值拷贝
let a = 1;
let b = a;
b = 2;
console.log(a); // 1（互不影响）

// 引用类型 — 引用拷贝
let obj1 = { name: 'yiwen' };
let obj2 = obj1;
obj2.name = 'changed';
console.log(obj1.name); // 'changed'（共享同一对象）
```

---

### 2. 类型判断有哪些方式？各自的局限性？

| 方式 | 用法 | 局限性 |
|------|------|--------|
| `typeof` | `typeof 'hello'` → `'string'` | `typeof null` → `'object'`（历史 bug）；无法区分 Array/Object/Date |
| `instanceof` | `[] instanceof Array` → `true` | 不能判断基本类型；跨 iframe 失效（原型链不同） |
| `constructor` | `'hello'.constructor === String` | `null`/`undefined` 没有 constructor；constructor 可被重写 |
| `Object.prototype.toString.call()` | `Object.prototype.toString.call([])` → `'[object Array]'` | **最准确**，无明显局限，推荐使用 |

```js
// 万能类型判断函数
function getType(value) {
    return Object.prototype.toString.call(value).slice(8, -1).toLowerCase();
}

getType([]); // 'array'
getType(null); // 'null'
getType(123); // 'number'
getType(new Date()); // 'date'
```

---

### 3. == 和 === 的区别？隐式类型转换规则？

- `===` **严格相等**：不进行类型转换，类型和值都必须相同
- `==` **宽松相等**：会进行隐式类型转换后再比较

**隐式类型转换规则（==）：**

1. `null == undefined` → `true`（且它们不等于其他任何值）
2. **String vs Number** → String 转 Number
3. **Boolean vs 其他** → Boolean 先转 Number（`true`→`1`，`false`→`0`）
4. **Object vs 基本类型** → Object 调用 `valueOf()` / `toString()` 转基本类型

```js
'' == false       // true（'' → 0, false → 0）
'0' == false      // true（false → 0, '0' → 0）
[] == false       // true（[] → '' → 0, false → 0）
[] == ![]         // true（![] → false → 0, [] → 0）
null == undefined // true
null == 0         // false
NaN == NaN        // false
```

> **最佳实践：** 始终使用 `===`，避免隐式转换带来的 bug。

---

## 原型链 & 继承

### 4. 原型链是什么？

每个对象都有一个内部属性 `[[Prototype]]`（通过 `__proto__` 访问），指向它的原型对象。当访问对象的属性时，如果自身没有，就沿着原型链向上查找，直到 `null` 为止。

```
实例对象 → 构造函数.prototype → Object.prototype → null
```

```js
function Person(name) {
    this.name = name;
}
Person.prototype.sayHi = function () {
    console.log(`Hi, I'm ${this.name}`);
};

const p = new Person('yiwen');

// 原型链查找
p.sayHi();           // 在 Person.prototype 上找到
p.toString();        // 在 Object.prototype 上找到
p.hasOwnProperty('name'); // true（自身属性）
```

**关键等式：**
```js
p.__proto__ === Person.prototype;               // true
Person.prototype.__proto__ === Object.prototype; // true
Object.prototype.__proto__ === null;            // true（原型链终点）
Person.__proto__ === Function.prototype;        // true
```

---

### 5. new 操作符做了什么？手写 new

**new 的四步操作：**
1. 创建一个空对象
2. 将空对象的 `__proto__` 指向构造函数的 `prototype`
3. 将构造函数的 `this` 绑定到新对象并执行
4. 如果构造函数返回对象，则返回该对象；否则返回新创建的对象

```js
function myNew(Constructor, ...args) {
    // 1. 创建空对象，原型指向构造函数的 prototype
    const obj = Object.create(Constructor.prototype);
    // 2. 执行构造函数，绑定 this
    const result = Constructor.apply(obj, args);
    // 3. 如果构造函数返回对象，则返回该对象；否则返回 obj
    return result instanceof Object ? result : obj;
}

// 使用
function Person(name) {
    this.name = name;
}
const p = myNew(Person, 'yiwen');
console.log(p.name); // 'yiwen'
console.log(p instanceof Person); // true
```

---

### 6. 继承有哪几种方式？各自的优缺点？

#### （1）原型链继承
```js
Child.prototype = new Parent();
```
- 缺点：引用类型属性被所有实例共享；无法向父构造函数传参

#### （2）构造函数继承（借用 call）
```js
function Child(...args) {
    Parent.call(this, ...args);
}
```
- 优点：可传参，不共享引用属性
- 缺点：无法继承父类原型上的方法

#### （3）组合继承（最常用的经典方式）
```js
function Child(...args) {
    Parent.call(this, ...args); // 继承属性
}
Child.prototype = new Parent(); // 继承方法
Child.prototype.constructor = Child;
```
- 缺点：父构造函数执行了两次

#### （4）寄生组合继承（最优方案）
```js
function Child(...args) {
    Parent.call(this, ...args);
}
Child.prototype = Object.create(Parent.prototype);
Child.prototype.constructor = Child;
```
- 优点：只调用一次父构造函数，原型链正确

#### （5）ES6 class 继承
```js
class Child extends Parent {
    constructor(...args) {
        super(...args);
    }
}
```
- 本质是寄生组合继承的语法糖

---

## 作用域 & 闭包

### 7. 作用域链是什么？

作用域决定了变量的可访问范围。JS 有三种作用域：

- **全局作用域**
- **函数作用域**
- **块级作用域**（`let`/`const` + `{}`）

**作用域链：** 当访问一个变量时，引擎从当前作用域开始查找，如果找不到就向外层作用域逐级查找，直到全局作用域。这条查找链路就是作用域链。

```js
const global = 'global';

function outer() {
    const outerVar = 'outer';

    function inner() {
        const innerVar = 'inner';
        console.log(innerVar);  // 当前作用域找到
        console.log(outerVar);  // 外层作用域找到
        console.log(global);    // 全局作用域找到
    }

    inner();
}
```

> 作用域链在**函数定义时**确定（词法作用域 / 静态作用域），而非调用时。

---

### 8. 什么是闭包？实际应用场景有哪些？

**闭包 = 函数 + 其定义时的词法环境。** 即一个函数能够访问其外部函数作用域中的变量，即使外部函数已经执行完毕。

```js
function createCounter() {
    let count = 0; // 被闭包"捕获"
    return {
        increment: () => ++count,
        getCount: () => count,
    };
}

const counter = createCounter();
counter.increment();
counter.increment();
console.log(counter.getCount()); // 2
```

**实际应用场景：**

1. **数据封装 / 私有变量**
```js
function createUser(name) {
    let _name = name; // 私有
    return {
        getName: () => _name,
        setName: (n) => { _name = n; },
    };
}
```

2. **防抖 / 节流**
```js
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}
```

3. **柯里化**
```js
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) return fn(...args);
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
}
```

4. **模块模式（IIFE）**
5. **缓存 / 记忆化**

---

### 9. 经典循环闭包陷阱，如何解决？

```js
// 问题代码
for (var i = 0; i < 5; i++) {
    setTimeout(() => console.log(i), 1000);
}
// 输出: 5 5 5 5 5（全部是 5）
```

**原因：** `var` 没有块级作用域，循环结束后 `i = 5`，所有回调共享同一个 `i`。

**解决方案：**

```js
// 方案一：let（推荐）
for (let i = 0; i < 5; i++) {
    setTimeout(() => console.log(i), 1000);
}

// 方案二：IIFE
for (var i = 0; i < 5; i++) {
    (function (j) {
        setTimeout(() => console.log(j), 1000);
    })(i);
}

// 方案三：setTimeout 第三个参数
for (var i = 0; i < 5; i++) {
    setTimeout((j) => console.log(j), 1000, i);
}
```

---

## this 指向

### 10. this 的四种绑定规则是什么？

按优先级从低到高：

| 规则 | 描述 | 示例 |
|------|------|------|
| **默认绑定** | 独立调用，`this` 指向全局（严格模式下为 `undefined`） | `fn()` |
| **隐式绑定** | 由上下文对象调用，`this` 指向该对象 | `obj.fn()` |
| **显式绑定** | 通过 `call`/`apply`/`bind` 指定 | `fn.call(obj)` |
| **new 绑定** | 构造函数调用，`this` 指向新创建的对象 | `new Fn()` |

```js
function sayName() {
    console.log(this.name);
}

const obj = { name: 'yiwen', sayName };

sayName();              // undefined（默认绑定 → window/undefined）
obj.sayName();          // 'yiwen'（隐式绑定）
sayName.call(obj);      // 'yiwen'（显式绑定）
new sayName();          // undefined（new 绑定，this 指向新对象）
```

**隐式丢失问题：**
```js
const fn = obj.sayName; // 赋值后丢失上下文
fn(); // undefined（退化为默认绑定）
```

---

### 11. 箭头函数的 this 和普通函数有什么区别？

| 对比项 | 普通函数 | 箭头函数 |
|--------|---------|---------|
| this 绑定 | 动态，取决于调用方式 | 静态，继承**定义时**外层作用域的 this |
| arguments | 有自己的 arguments 对象 | 没有 arguments，使用 rest 参数 |
| 作为构造函数 | 可以 new | 不可以 new |
| prototype | 有 | 没有 |

```js
const obj = {
    name: 'yiwen',
    // 普通函数：this 取决于调用方式
    greet() {
        setTimeout(function () {
            console.log(this.name); // undefined（this → window）
        }, 100);
    },
    // 箭头函数：继承 greet 的 this
    greetArrow() {
        setTimeout(() => {
            console.log(this.name); // 'yiwen'（this → obj）
        }, 100);
    },
};
```

> **箭头函数的 this 无法被 call/apply/bind 修改。**

---

## 事件循环 & 宏任务/微任务

### 12. 详细说一下事件循环的执行顺序？

```
┌──────────────────────────────────────────┐
│              调用栈 (Call Stack)           │
│  同步代码在此执行                           │
└────────────┬─────────────────────────────┘
             │ 同步代码执行完毕
             ▼
┌──────────────────────────────────────────┐
│           微任务队列 (Microtask Queue)     │
│  Promise.then, queueMicrotask,           │
│  MutationObserver                        │
│  → 全部清空（包括执行中新产生的微任务）       │
└────────────┬─────────────────────────────┘
             │ 微任务队列清空
             ▼
┌──────────────────────────────────────────┐
│           宏任务队列 (Macrotask Queue)     │
│  setTimeout, setInterval, I/O,           │
│  requestAnimationFrame                    │
│  → 取出一个宏任务执行                       │
└────────────┬─────────────────────────────┘
             │ 执行完一个宏任务
             ▼
        回到微任务队列（循环）
```

**执行顺序：**
1. 执行同步代码（主线程 / 当前宏任务）
2. 清空所有微任务（包括执行过程中新产生的微任务）
3. 取出一个宏任务执行
4. 再清空所有微任务
5. 重复 3-4

---

### 13. 微任务和宏任务分别包含哪些？

| 宏任务 (Macrotask) | 微任务 (Microtask) |
|--------------------|--------------------|
| `setTimeout` / `setInterval` | `Promise.then` / `.catch` / `.finally` |
| `setImmediate`（Node） | `queueMicrotask` |
| I/O 操作 | `MutationObserver`（浏览器） |
| `requestAnimationFrame`（浏览器） | `process.nextTick`（Node，优先级最高） |
| UI 渲染（浏览器） | `async/await`（await 之后的代码） |

---

### 14. 经典输出题：async/await + Promise + setTimeout 混合

```js
console.log('1');

setTimeout(() => {
    console.log('2');
}, 0);

new Promise((resolve) => {
    console.log('3');
    resolve();
}).then(() => {
    console.log('4');
});

async function foo() {
    console.log('5');
    await bar();
    console.log('6');
}

async function bar() {
    console.log('7');
}

foo();

console.log('8');

// 输出顺序: 1 → 3 → 5 → 7 → 8 → 4 → 6 → 2
```

**分析：**
- **同步代码：** `1` → `3`（Promise 构造函数同步执行）→ `5` → `7`（await 前同步）→ `8`
- **微任务：** `4`（Promise.then）→ `6`（await 之后，相当于 .then）
- **宏任务：** `2`（setTimeout）

---

## Promise & async/await

### 15. Promise 解决了什么问题？

**核心问题：回调地狱（Callback Hell）**

```js
// 回调地狱
getData(function (a) {
    getMoreData(a, function (b) {
        getEvenMoreData(b, function (c) {
            // 嵌套越来越深...
        });
    });
});

// Promise 链式调用
getData()
    .then((a) => getMoreData(a))
    .then((b) => getEvenMoreData(b))
    .then((c) => {
        // 扁平化
    })
    .catch((err) => {
        // 统一错误处理
    });
```

**Promise 的三种状态：** `pending` → `fulfilled` / `rejected`（不可逆）

---

### 16. Promise.all / allSettled / race / any 的区别？

| 方法 | 行为 | 适用场景 |
|------|------|---------|
| `Promise.all` | 全部成功才成功，一个失败立即失败 | 并发请求，全部需要成功 |
| `Promise.allSettled` | 等所有完成，不论成功失败 | 获取所有结果，不关心是否失败 |
| `Promise.race` | 返回第一个完成的（无论成功或失败） | 超时控制 |
| `Promise.any` | 返回第一个成功的，全部失败才失败 | 多个备选源，取最快成功的 |

```js
// 超时控制示例
function fetchWithTimeout(url, ms) {
    return Promise.race([
        fetch(url),
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Timeout')), ms)
        ),
    ]);
}
```

---

### 17. async/await 如何做错误处理？

```js
// 方式一：try/catch（最常用）
async function fetchData() {
    try {
        const res = await fetch('/api/data');
        const data = await res.json();
        return data;
    } catch (err) {
        console.error('请求失败:', err);
    }
}

// 方式二：.catch() 链式处理
async function fetchData() {
    const data = await fetch('/api/data')
        .then((res) => res.json())
        .catch((err) => {
            console.error(err);
            return null; // 降级处理
        });
    return data;
}

// 方式三：包装函数（Go 风格）
function to(promise) {
    return promise.then((data) => [null, data]).catch((err) => [err, null]);
}

async function fetchData() {
    const [err, data] = await to(fetch('/api/data'));
    if (err) {
        console.error(err);
        return;
    }
    // 使用 data
}
```

---

## ES6+ 特性

### 18. var / let / const 的区别？什么是暂时性死区？

| 特性 | `var` | `let` | `const` |
|------|-------|-------|---------|
| 作用域 | 函数作用域 | 块级作用域 | 块级作用域 |
| 变量提升 | 提升并初始化为 `undefined` | 提升但**不初始化**（TDZ） | 提升但**不初始化**（TDZ） |
| 重复声明 | 允许 | 不允许 | 不允许 |
| 重新赋值 | 允许 | 允许 | 不允许（引用类型内容可变） |

**暂时性死区（Temporal Dead Zone, TDZ）：**

`let`/`const` 声明的变量虽然会提升，但在声明语句之前访问会抛出 `ReferenceError`，这个区域就是 TDZ。

```js
console.log(a); // undefined（var 提升并初始化）
var a = 1;

console.log(b); // ReferenceError（TDZ）
let b = 2;

// TDZ 的边界
{
    // TDZ 开始
    console.log(x); // ReferenceError
    let x = 10;     // TDZ 结束
}
```

---

### 19. 可选链 ?. 和空值合并 ?? 的区别？?? 和 || 的区别？

**可选链 `?.`：** 安全地访问深层属性，遇到 `null`/`undefined` 短路返回 `undefined`

```js
const user = { address: { city: 'Beijing' } };
console.log(user?.address?.city);    // 'Beijing'
console.log(user?.contact?.phone);   // undefined（不会报错）
console.log(user?.getName?.());      // undefined（安全调用方法）
```

**空值合并 `??`：** 仅在左侧为 `null`/`undefined` 时取右侧值

**`??` vs `||` 的区别：**

```js
// || 对所有 falsy 值生效（0, '', false, null, undefined, NaN）
0 || 'default'     // 'default'（0 是 falsy）
'' || 'default'    // 'default'
false || 'default' // 'default'

// ?? 仅对 null/undefined 生效
0 ?? 'default'     // 0 ✅
'' ?? 'default'    // '' ✅
false ?? 'default' // false ✅
null ?? 'default'  // 'default'
undefined ?? 'default' // 'default'
```

> **最佳实践：** 当 `0`、`''`、`false` 是合法值时，使用 `??` 代替 `||`。

---

## 模块化

### 20. CommonJS 和 ESModule 的核心区别？

| 对比项 | CommonJS (CJS) | ESModule (ESM) |
|--------|---------------|----------------|
| 加载方式 | **运行时**加载 | **编译时**静态分析 |
| 输出 | 值的拷贝 | 值的引用（实时绑定） |
| 语法 | `require()` / `module.exports` | `import` / `export` |
| 执行时机 | 同步加载 | 异步加载 |
| 顶层 this | `module.exports` | `undefined` |
| 循环依赖 | 返回已执行部分的拷贝 | 通过引用可正常工作（但需注意初始化顺序） |

```js
// CommonJS — 值的拷贝
// counter.js
let count = 0;
module.exports = { count, increment: () => ++count };

// main.js
const { count, increment } = require('./counter');
increment();
console.log(count); // 0（拿到的是拷贝，不会变）

// ESModule — 值的引用
// counter.mjs
export let count = 0;
export function increment() { count++; }

// main.mjs
import { count, increment } from './counter.mjs';
increment();
console.log(count); // 1（实时绑定，会变）
```

---

### 21. 为什么 ESModule 支持 Tree Shaking 而 CommonJS 不行？

**Tree Shaking：** 打包时移除未使用的代码（Dead Code Elimination）。

**ESModule 支持的原因：**
- `import`/`export` 是**静态语法**，必须在模块顶层，不能在条件语句中
- 打包工具（Webpack/Rollup）可在**编译时**分析依赖关系，确定哪些导出被使用

**CommonJS 不支持的原因：**
- `require()` 是**运行时**函数调用，可以出现在任何位置
- 导出是动态对象，无法在编译时确定哪些属性被使用

```js
// ESM — 可以 Tree Shaking
import { used } from './utils'; // unused 会被移除

// CJS — 无法 Tree Shaking
const { used } = require('./utils'); // 编译时无法确定 require 的结果
```

---

## 手写题

### 手写 instanceof

```js
function myInstanceof(obj, Constructor) {
    if (obj === null || typeof obj !== 'object') return false;
    let proto = Object.getPrototypeOf(obj);
    while (proto !== null) {
        if (proto === Constructor.prototype) return true;
        proto = Object.getPrototypeOf(proto);
    }
    return false;
}
```

---

### 手写 call / apply / bind

```js
// call
Function.prototype.myCall = function (context, ...args) {
    context = context == null ? globalThis : Object(context);
    const key = Symbol();
    context[key] = this;
    const result = context[key](...args);
    delete context[key];
    return result;
};

// apply
Function.prototype.myApply = function (context, args = []) {
    context = context == null ? globalThis : Object(context);
    const key = Symbol();
    context[key] = this;
    const result = context[key](...args);
    delete context[key];
    return result;
};

// bind
Function.prototype.myBind = function (context, ...outerArgs) {
    const fn = this;
    return function boundFn(...innerArgs) {
        // 支持 new 调用
        if (new.target) {
            return new fn(...outerArgs, ...innerArgs);
        }
        return fn.apply(context, [...outerArgs, ...innerArgs]);
    };
};
```

---

### 手写深拷贝（含循环引用）

```js
function deepClone(obj, map = new WeakMap()) {
    // 基本类型直接返回
    if (obj === null || typeof obj !== 'object') return obj;

    // 处理循环引用
    if (map.has(obj)) return map.get(obj);

    // 处理特殊类型
    if (obj instanceof Date) return new Date(obj);
    if (obj instanceof RegExp) return new RegExp(obj);

    const clone = Array.isArray(obj) ? [] : {};
    map.set(obj, clone);

    for (const key of Reflect.ownKeys(obj)) {
        clone[key] = deepClone(obj[key], map);
    }

    return clone;
}

// 测试循环引用
const obj = { a: 1 };
obj.self = obj;
const cloned = deepClone(obj);
console.log(cloned.self === cloned); // true（正确处理循环引用）
console.log(cloned !== obj);         // true（不是同一个对象）
```

---

### 手写防抖 / 节流

```js
// 防抖（debounce）：事件停止触发后延迟执行
function debounce(fn, delay, immediate = false) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        if (immediate && !timer) {
            fn.apply(this, args);
        }
        timer = setTimeout(() => {
            if (!immediate) fn.apply(this, args);
            timer = null;
        }, delay);
    };
}

// 节流（throttle）：固定时间间隔内只执行一次
function throttle(fn, interval) {
    let lastTime = 0;
    return function (...args) {
        const now = Date.now();
        if (now - lastTime >= interval) {
            lastTime = now;
            fn.apply(this, args);
        }
    };
}
```

---

### 手写 Promise

```js
class MyPromise {
    #state = 'pending';
    #value = undefined;
    #callbacks = [];

    constructor(executor) {
        const resolve = (value) => {
            if (this.#state !== 'pending') return;
            this.#state = 'fulfilled';
            this.#value = value;
            this.#callbacks.forEach((cb) => cb.onFulfilled(value));
        };

        const reject = (reason) => {
            if (this.#state !== 'pending') return;
            this.#state = 'rejected';
            this.#value = reason;
            this.#callbacks.forEach((cb) => cb.onRejected(reason));
        };

        try {
            executor(resolve, reject);
        } catch (err) {
            reject(err);
        }
    }

    then(onFulfilled, onRejected) {
        onFulfilled =
            typeof onFulfilled === 'function' ? onFulfilled : (v) => v;
        onRejected =
            typeof onRejected === 'function'
                ? onRejected
                : (e) => {
                      throw e;
                  };

        return new MyPromise((resolve, reject) => {
            const handle = (callback, value) => {
                queueMicrotask(() => {
                    try {
                        const result = callback(value);
                        if (result instanceof MyPromise) {
                            result.then(resolve, reject);
                        } else {
                            resolve(result);
                        }
                    } catch (err) {
                        reject(err);
                    }
                });
            };

            if (this.#state === 'fulfilled') {
                handle(onFulfilled, this.#value);
            } else if (this.#state === 'rejected') {
                handle(onRejected, this.#value);
            } else {
                this.#callbacks.push({
                    onFulfilled: (v) => handle(onFulfilled, v),
                    onRejected: (e) => handle(onRejected, e),
                });
            }
        });
    }

    catch(onRejected) {
        return this.then(null, onRejected);
    }

    finally(callback) {
        return this.then(
            (value) =>
                MyPromise.resolve(callback()).then(() => value),
            (reason) =>
                MyPromise.resolve(callback()).then(() => {
                    throw reason;
                })
        );
    }

    static resolve(value) {
        if (value instanceof MyPromise) return value;
        return new MyPromise((resolve) => resolve(value));
    }

    static reject(reason) {
        return new MyPromise((_, reject) => reject(reason));
    }

    static all(promises) {
        return new MyPromise((resolve, reject) => {
            const results = [];
            let count = 0;
            promises.forEach((p, i) => {
                MyPromise.resolve(p).then((value) => {
                    results[i] = value;
                    if (++count === promises.length) resolve(results);
                }, reject);
            });
        });
    }

    static race(promises) {
        return new MyPromise((resolve, reject) => {
            promises.forEach((p) => {
                MyPromise.resolve(p).then(resolve, reject);
            });
        });
    }
}
```
