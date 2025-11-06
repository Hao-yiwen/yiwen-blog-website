---
title: map和forof
sidebar_label: map和forof
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# map和forof

今晚碰到一个离谱的事情，发现foreach和map中的内容如果是异步的，竟然没法顺序执行，感觉非常的离谱，所以写了个demo放在这里。

```js
// map没法同步执行测试
const arr1 = [1, 2, 3, 4, 5];
const arr2 = [6, 7, 8, 9, 10];

const promiseDev = number => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            console.log('Dev', number);
            resolve('Dev');
        }, 500);
    });
};

// MAP和foreach

arr1.forEach(async (item, index) => {
    await promiseDev(item);
});

arr2.forEach(async (item, index) => {
    await promiseDev(item);
});

// forof
async function main() {
    for (let item of arr1) {
        await promiseDev(item);
    }

    for (let item of arr2) {
        await promiseDev(item);
    }
}

main();
```

## 为什么map和foreach没法同步执行

```
Array.prototype.myMap = function(callback) {
  let result = [];
  for (let i = 0; i < this.length; i++) {
    result.push(callback(this[i], i, this));
  }
  return result;
};
```

为了js的性能非阻塞性质，所以map和foreach是非阻塞的，即使添加async/await也没用。此事还得用forof处理，因为它底层是迭代器。

```js title="forof"
function* generatorFunction(array) {
    for (let item of array) {
        yield item;
    }
}

async function processArray(array) {
    const iterator = generatorFunction(array);
    let result = iterator.next();
    while (!result.done) {
        await asyncProcess(result.value);
        result = iterator.next();
    }
}
```
