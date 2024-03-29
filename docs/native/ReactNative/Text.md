# Text中的adjustsFontSizeToFit属性

今天正好在帮业务查问题，顺手就查了一个Text的adjustsFontSizeToFit属性，该属性的作用是自使用字体大小，但是在使用的时候还是遇到了一点问题，还是觉得要记录一下。

## 代码

```tsx
import React from 'react';
import {View, Text, Button} from 'react-native';

const App: React.FC<any> = () => {
  return (
    <View style={{flex: 1}}>
      <View style={{width: 7000, height: 20}}>
        <Text adjustsFontSizeToFit={true} style={{fontSize: 40}}>
          测试adjustsFontSizeToFit
        </Text>
      </View>
    </View>
  );
};

export default App;
```

## 注意事项
1. adjustsFontSizeToFit主要是为了容器较小而字体大的时候字体自适应容器诞生的，字体通常不会变大。
2. 如果开始的时候就没有给Text设置字体大小，那么字体并不会因为容器变小，只有给字体设置大小的，字体才能变大或者变小。
4. 只有字体本身很大，然后容器较小，字体才会自适应容器。