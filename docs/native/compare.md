---
sidebar_position: 3
---

import SVG from "./anatomy.svg"

# Flutter和ReactNative比较

## 介绍

### React Native

React Native (RN) 是由 Facebook 推出的一种流行的移动应用跨平台开发框架。它极大地简化了移动应用的开发过程，因为开发者可以使用JavaScript进行开发，而不必分别用Java或Swift/Objective-C来为Android和iOS编写原生代码。React Native最引人注目的优势之一在于其动态更新能力。它允许开发者绕过应用商店的审核过程，直接向用户的设备动态下发更新。这一机制已经在许多公司得到广泛实施，无论是从用户体验还是开发成本角度来看，都已经十分成熟。

### Flutter

Flutter 是由谷歌开发的一个开源移动应用开发框架，它使用谷歌开发的Dart语言。Flutter在开发过程中支持热重载，即使开发者对代码进行了更改，也无需重新启动应用即可看到更新效果，这降低了学习曲线，并提高了开发效率。在性能方面，得益于其高效的Skia图形引擎，Flutter在各项性能测试中通常优于React Native。虽然Flutter在许多跨平台解决方案中占有一席之地，并且它的社区也在持续增长，但到目前为止，Flutter官方尚未提供官方的动态更新方案。因此，每次发布新版本仍需遵循应用商店的审核流程。

## 原理解析

### React Native

React Native 是基于 JavaScript 的框架，允许开发者使用类似于 React 的方式来构建跨平台的移动应用。在这个框架中，开发者可以使用 JSX（类似 HTML 的语法）来设计界面。Facebook 开发的 Metro 打包器负责将这种 JSX 以及其他 JavaScript 代码转换成可以在移动设备上运行的代码包，这大大简化了开发过程。

React Native 的运行时基于 Hermes/JavaScriptCore 引擎，这是一个用 C++ 编写的轻量级 JavaScript 引擎。它通过桥接（Bridge）机制与原生平台（Android 或 iOS）进行通信，调用原生组件或 API，实现跨平台功能。当应用中的视图需要更新时，React Native 使用类似 React 的虚拟 DOM 策略来计算最高效的更新路径。更新指令随后通过 Hermes/JavaScriptCore 发送到原生端，触发 UI 的实际更新。最近三年，ReactNative 推出了新架构 Fabric，它使得 React Native 的 UI 可以更快地构建和渲染，因为它提供了 JSI(JavaScript Interface) 来同步调用原生方法，减少了 JavaScript 和原生平台之间的通信开销。

此流程确保了 React Native 应用能够使用 JavaScript 进行逻辑处理，同时保持接近原生应用的性能。通过动态加载 JavaScript 代码的方式，React Native 还能实现绕过应用商店直接向用户推送更新的能力。

### ReactNative底层解析图

![RN底层图](https://reactnative.dev/assets/images/xplat-implementation-diagram-7611cf9dfb6d15667365630147d83ca5.png)

:::tip
因为 iOS 和 Android 原生组件在表现和交互上可能存在差异，所以在使用 React Native（RN）开发时，需要注意两个平台的差异。目前 Facebook 主要负责维护 RN 的核心库（包括核心 API 和组件、Metro 打包器和 Hermes 引擎），而许多其他的组件和 API 都是由社区来维护和提供的。这意味着如果你想要使用 RN 进行开发，你可能需要自己封装一些组件或者使用第三方库。Expo 是一个较为知名的第三方提供商，它为快速开发 RN 应用提供了丰富的预封装组件和开发工具，但还有其他选项可供开发者选择。
:::

### Flutter

Flutter是基于Dart语言开发的，这是一种现代的编程语言，支持即时编译（JIT）和预编译（AOT）模式，能够在提高开发效率的同时保证应用性能。Flutter提供debug、profile和release三种开发模式：debug模式采用JIT编译，以支持热重载功能，极大提高了开发效率；而release模式则采用AOT编译，将Dart代码编译成机器码，优化了应用的运行性能。Flutter框架自带依赖管理和资源打包工具，简化了构建流程，使开发者能专注于业务逻辑。经过多次迭代更新，Flutter官方提供了大量用C++编写的跨平台API，这减少了对第三方库的依赖。它的核心亮点之一是Skia渲染引擎，这个自研引擎能实现跨平台UI的一致性，提供了独立于原生组件的渲染方式。

尽管Flutter有这些优势，但在动态化方面它面临一定的挑战。因为Flutter的release版本是编译成机器码的，这限制了它的动态更新能力。虽然Dart语言天生支持JIT模式，但出于对安全性和性能的考虑，Flutter官方尚未推出官方的动态化方案。尽管如此，许多大公司正在探索适合Flutter的动态化技术。在学习资源方面，Flutter的中文社区非常活跃，这使得学习Flutter变得更加容易，成为一项优秀的跨平台开发解决方案。

Flutter的官方文档十分完善，从原理到底层实现，官方文档都有记述，并且Flutter还有非常强大的调试工具Dart devtools,使得代码调试变得非常强大和较为简单。

### Flutter底层解析图

<SVG />

## 对比

### 编译层面

Flutter 的工作原理与 React Native 不同，因此不需要像 Metro 打包器或 Hermes JavaScript 引擎那样的组件。在 Flutter 中，Dart 代码会被 Dart 编译器直接编译成 ARM 代码或 x86/x64 代码，以便直接在不同平台的设备上运行。

1. DartVM：
   在开发过程中，Flutter 使用 DartVM 来提供热重载功能，这允许开发者在不重新启动应用的情况下，快速测试和构建 UI。
   DartVM 支持 JIT（Just-In-Time）编译，这意味着它可以在开发过程中实时编译 Dart 代码，提高开发效率。

2. AOT 编译：
   当构建发布版本的应用时，Flutter 使用 AOT（Ahead-Of-Time）编译。Dart 代码会被编译成本地机器代码（ARM 或 x86/x64），这样在应用运行时就不再需要 DartVM。
   AOT 编译提供了更好的性能和启动时间，因为代码是直接编译成机器码的。

3. 渲染引擎：
   Flutter 使用 Skia 作为其渲染引擎，所有的绘制都是由 Skia 完成的，这意味着 Flutter 可以在任何平台上提供一致的视觉体验。
   由于 Flutter 不使用平台的原生组件，而是直接在画布上绘制 UI，这样就避免了对平台特定的 UI 框架的依赖。

因此，Flutter 提供了一种不同的架构来构建和部署跨平台的移动应用。这种方法允许高性能和高定制性，而无需 JavaScript 的打包器或特定的执行引擎。Dart 的编译过程和 Flutter 的渲染机制使它能够在设备上提供接近原生性能的体验。
