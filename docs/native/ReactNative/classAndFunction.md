---
sidebar_position: 3
---

# QA

## class组件和function组件在状态变量中的区别

React Hooks，包括 useState，在设计时就考虑到了在函数组件中保留状态的需求。当你在函数组件中调用 useState 时，React 会为该组件实例创建一个新的状态变量，并在组件的每次重新渲染时保留这个状态变量的值。

这是通过 React 的闭包机制实现的。每次渲染都有它自己的 Props 和 State，它们形成了一个 "快照"，并且每个 useEffect 和 useState 都能看到它们自己的这个 "快照"。这就是为什么默认情况下，每次渲染都有它自己的事件处理函数和副作用函数。

然而，类组件的工作方式不同。在类组件中，状态是直接挂载到类的实例上的（即 this.state 和 this.setState），并且在每次重新渲染时，这个状态都会被新的状态替换。这就是为什么在类组件中，你会看到状态在每次重新渲染时都被重置的原因。

[Hooks实现](https://github.com/facebook/react/blob/main/packages/react-reconciler/src/ReactFiberHooks.js)

[class实现](https://github.com/facebook/react/blob/main/packages/react-reconciler/src/ReactFiberClassComponent.js)

## rn为什么尺寸没有单位

在 React Native 中，所有的尺寸单位都是以密度无关像素（dp）为单位的，这意味着在不同的设备上，相同的 dp 值会被渲染为相同的视觉大小，而不是相同的物理像素数量。

因此，如果你在 React Native 中设置一个元素的宽度为 100，那么无论在哪种 iPhone（或任何其他设备）上，这个元素的视觉宽度都会是一样的。然而，这个宽度对应的物理像素数量可能会因设备的屏幕密度（每英寸像素数量，PPI）不同而不同。

例如，iPhone 8 的屏幕密度为 326 PPI，而 iPhone 12 Pro 的屏幕密度为 460 PPI。因此，一个宽度为 100 dp 的元素在 iPhone 8 上可能对应 326 物理像素，而在 iPhone 12 Pro 上可能对应 460 物理像素。但无论如何，这个元素在两种设备上的视觉宽度都是一样的。
