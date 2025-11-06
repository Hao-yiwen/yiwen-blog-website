---
title: 回调函数和参数传递
sidebar_label: 回调函数和参数传递
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 回调函数和参数传递

在react和react native中经常会遇到参数传递和回调函数。在这里，我们记录两种传递方式，及其使用场景。

## 函数组件中的回调函数

```js
// ChildComponent.js
import React, { useEffect } from 'react';

function ChildComponent({ passMethodToParent }) {
  // 定义需要传递给父组件的方法
  const childMethod = () => {
    console.log('Method called from the parent component');
  };

  // 使用 useEffect Hook 在组件挂载时执行
  useEffect(() => {
    // 将方法传递给父组件
    passMethodToParent(childMethod);
  }, [passMethodToParent]); // 确保当 passMethodToParent 改变时重新执行

  return <div>Child Component</div>;
}

export default ChildComponent;

// parent.js
import React, { useState, useRef } from 'react';
import ChildComponent from './ChildComponent';

function ParentComponent() {
  const childMethod = useRef();

  // 定义接收子组件方法的函数
  const receiveChildMethod = (method) => {
    childMethod.current = method;
  };

  // 调用子组件传递的方法
  const callChildMethod = () => {
    if (childMethod.current) {
      childMethod.current();
    }
  };

  return (
    <div>
      <ChildComponent passMethodToParent={receiveChildMethod} />
      <button onClick={callChildMethod}>Call Child Method</button>
    </div>
  );
}

export default ParentComponent;
```

## 类组件中的回调函数

```js
// child.js
import React, { Component } from 'react';

class ChildComponent extends Component {
  constructor(props) {
    super(props);
    // 绑定方法，确保在方法中的 this 指向当前组件实例
    this.childMethod = this.childMethod.bind(this);
  }

  childMethod() {
    console.log('Method called from the parent component');
  }

  componentDidMount() {
    // 通过 props 将方法传递给父组件
    this.props.passMethodToParent(this.childMethod);
  }

  render() {
    return <div>Child Component</div>;
  }
}

export default ChildComponent;

// parent.js
import React, { Component } from 'react';
import ChildComponent from './ChildComponent';

class ParentComponent extends Component {
  constructor(props) {
    super(props);
  }

  setChildMethod = (method) => {
    this.childMethod = method;
  };

  callChildMethod = () => {
    if (this.childMethod) {
      this.childMethod();
    }
  };

  render() {
    return (
      <div>
        <ChildComponent passMethodToParent={this.setChildMethod} />
        <button onClick={this.callChildMethod}>Call Child Method</button>
      </div>
    );
  }
}

export default ParentComponent;
```