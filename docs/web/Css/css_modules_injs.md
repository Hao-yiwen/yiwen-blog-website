---
sidebar_position: 4
---

# CSS-in-JS和CSS Modules

## CSS-in-JS

由来：
CSS-in-JS 是一个现代前端架构中用于处理组件范围或局部范围 CSS 的方法。这种方法与 React, Angular, Vue 等前端框架/库结合得非常紧密。CSS-in-JS 库如 styled-components, Emotion, JSS 等具有不同的 API，但核心概念是相似的。

特性：

-   动态样式：可以使用 JavaScript 的全部能力生成样式。
-   组件范围：样式直接与组件绑定，避免全局样式的污染。
-   优化打包：只包括用到的样式，而不是整个样式表。
-   主题和样式共享：可以轻易地通过 JavaScript 对象、props、state 等动态调整样式。

```jsx
import styled from 'styled-components';

const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'white'};
  color: ${props => props.primary ? 'white' : 'black'};
`;

<Button primary>Primary</Button>
<Button>Secondary</Button>
```

## CSS Modules

### 由来：

CSS Modules 是一种使类名局部作用域化的方法，以避免全局样式污染。虽然它主要用于 Webpack 中，但也可以在其他构建工具中使用。CSS Modules 不需要依赖 JavaScript，因此它更像是对传统 CSS 的改进。

### 特性：

-   局部范围：默认地，所有的类名都只在组件范围内起作用。
-   简单的管理：通过 JavaScript 导入样式，并将其作为对象使用。
-   依然是 CSS：你可以使用纯 CSS，或预处理器，如 Sass 或 Less。

### 代码示例：

```jsx
// styles.module.css
.button {
  background: blue;
  color: white;
}

// Component.js
import styles from './styles.module.css';

const Button = () => <button className={styles.button}>Click Me</button>;
```

## 对比

### CSS-in-JS

-   在涉及到与 JavaScript 框架（如 React、Vue 或 Angular）紧密集成的项目中更受欢迎。
-   更方便地支持动态和响应式样式。
-   有广泛使用的库，如 styled-components 和 Emotion。

### CSS Modules

-   常用于不需要动态样式，或者不想增加额外依赖的项目。
-   与 Webpack 和其他模块打包工具良好集成。
-   更靠近传统的 CSS，因此对于不想使用 JavaScript 来处理样式的团队可能更容易接受。
