# RN事件机制相关API

在React Native中，事件监听和触发是组件间通信的关键机制之一。React Native提供了几种不同的API来帮助开发者在JavaScript和原生代码之间以及不同的组件之间进行事件的监听和触发。

1. DeviceEventEmitter
   DeviceEventEmitter是一个全局的事件广播系统，用于监听来自原生模块的事件。这是一个在JavaScript代码中使用的API，允许你订阅原生事件并对它们做出反应。

使用示例:

```js
import { DeviceEventEmitter } from 'react-native';

// 订阅事件
const subscription = DeviceEventEmitter.addListener('eventName', eventData => {
    console.log(eventData);
});

// 在适当的时候取消订阅
subscription.remove();
```

2. NativeEventEmitter
   NativeEventEmitter提供了一种更模块化的方式来处理来自特定原生模块的事件。你需要为特定的原生模块创建一个NativeEventEmitter实例。这对于封装和使用自定义原生模块非常有用。

```js
import { NativeEventEmitter, NativeModules } from 'react-native';
const { MyNativeModule } = NativeModules;

const myNativeModuleEmitter = new NativeEventEmitter(MyNativeModule);

const subscription = myNativeModuleEmitter.addListener(
    'eventName',
    eventData => {
        console.log(eventData);
    }
);

// 取消订阅
subscription.remove();
```
