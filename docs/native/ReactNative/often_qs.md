# 常见RN问题

1. 在RN样式中，zIndex层级在绝对定位上面。

2. 在使用新架构的时候以下的代码会少渲染一次。。

```js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { View, Text, Button, TextInput } from 'react-native';

// 模拟复杂异步操作
const simulateAsyncOperation = () => {
    return new Promise(resolve => {
        setTimeout(() => {
            resolve('done');
        }, 500); // 模拟500ms的异步操作
    });
};

// 第一层组件
const FirstLevelComponent = () => {
    const [firstLevelState, setFirstLevelState] = useState('1111111111');

    useEffect(() => {
        console.log(performance.now() + ' FirstLevelComponent render time');
        return () => {
            console.log(
                performance.now() + ' FirstLevelComponent remove render time'
            );
        };
    });

    return (
        <View
            style={{
                padding: 20,
                backgroundColor: '#f0f0f0',
                marginBottom: 10,
            }}
        >
            <Text>First Level Component</Text>
            <Text>{firstLevelState}</Text>
            <SecondLevelComponent setFirstLevelState={setFirstLevelState} />
        </View>
    );
};

// 第二层组件
const SecondLevelComponent = ({ setFirstLevelState }) => {
    const [secondLevelState, setSecondLevelState] = useState('1111111111');
    const thirdInputRef = useRef(null);
    const thirdViewRef = useRef(null);
    const [isShowThirdLevel, setIsShowThirdLevel] = useState(true);

    const updateFirstLevelState = useCallback(
        async (ref, viewRef) => {
            thirdInputRef.current = ref;
            thirdViewRef.current = viewRef;
            setSecondLevelState('2222222222');
            await simulateAsyncOperation();
            thirdInputRef.current.focus();
            thirdViewRef.current.setNativeProps({
                style: { backgroundColor: 'red' },
            });
            setFirstLevelState('2222222222');
            setIsShowThirdLevel(false);
        },
        [setFirstLevelState]
    );

    useEffect(() => {
        console.log(performance.now() + ' SecondLevelComponent render time');
        return () => {
            console.log(
                performance.now() + ' SecondLevelComponent remove render time'
            );
        };
    });

    return (
        <View
            style={{
                padding: 20,
                backgroundColor: '#d0d0d0',
                marginBottom: 10,
            }}
        >
            <Text>Second Level Component</Text>
            <Text>{secondLevelState}</Text>
            {isShowThirdLevel && (
                <ThirdLevelComponent
                    setSecondLevelState={updateFirstLevelState}
                />
            )}
        </View>
    );
};

// 第三层组件
const ThirdLevelComponent = ({ setSecondLevelState }) => {
    const [thirdLevelState, setThirdLevelState] = useState('11111');
    const inputRef = useRef(null);
    const viewRef = useRef(null);

    const handleClick = () => {
        // 更新第三层组件的 state
        setThirdLevelState('2222222');

        // 同时触发第二层组件的 hook
        setTimeout(() => {
            setSecondLevelState(inputRef.current, viewRef.current);
        }, 100);
    };

    useEffect(() => {
        return () => {
            console.log(
                performance.now() + ' ThirdLevelComponent remove render time'
            );
        };
    });

    return (
        <View
            style={{
                padding: 20,
                backgroundColor: '#b0b0b0',
                marginBottom: 10,
            }}
            ref={viewRef}
        >
            <Text>Third Level Component</Text>
            <Text>{thirdLevelState}</Text>
            <TextInput placeholder="这是一个简单的输入框～" ref={inputRef} />
            <Button title="Update State" onPress={handleClick} />
        </View>
    );
};

// 渲染第一层组件
const App = () => {
    return (
        <View style={{ padding: 50 }}>
            <FirstLevelComponent />
        </View>
    );
};

export default App;
```

```bash
# 新架构
 (NOBRIDGE) LOG  247745574.715766 ThirdLevelComponent remove render time
 (NOBRIDGE) LOG  247745574.988997 ThirdLevelComponent render time
 (NOBRIDGE) LOG  247745683.98892 ThirdLevelComponent remove render time
 (NOBRIDGE) LOG  247745684.341843 SecondLevelComponent remove render time
 (NOBRIDGE) LOG  247745684.498535 ThirdLevelComponent render time
 (NOBRIDGE) LOG  247745684.610459 SecondLevelComponent render time
 (NOBRIDGE) LOG  247746203.461843 ThirdLevelComponent remove render time
 (NOBRIDGE) LOG  247746205.302459 SecondLevelComponent remove render time
 (NOBRIDGE) LOG  247746205.505535 FirstLevelComponent remove render time
 (NOBRIDGE) LOG  247746205.644689 SecondLevelComponent render time
 (NOBRIDGE) LOG  247746205.736612 FirstLevelComponent render time
 # 631ms - 500ms - 100ms = 31ms

# 老架构
 LOG  261633253.512666 ThirdLevelComponent remove render time
 LOG  261633253.633666 ThirdLevelComponent render time
 LOG  261633354.077541 ThirdLevelComponent remove render time
 LOG  261633354.18625 SecondLevelComponent remove render time
 LOG  261633354.270125 ThirdLevelComponent render time
 LOG  261633354.31675 SecondLevelComponent render time
 LOG  261633882.711375 ThirdLevelComponent remove render time
 LOG  261633882.887958 SecondLevelComponent remove render time
 LOG  261633882.9635 FirstLevelComponent remove render time
 LOG  261633883.056125 ThirdLevelComponent render time
 LOG  261633883.109208 SecondLevelComponent render time
 LOG  261633883.156125 FirstLevelComponent render time
 LOG  261633899.212166 ThirdLevelComponent remove render time
 LOG  261633900.514625 SecondLevelComponent remove render time
 LOG  261633900.587916 SecondLevelComponent render time
 # 647ms - 500ms - 100ms = 47ms
```

版本是0.74.2，从这段代码可以看到新架构减少了渲染帧，提高了渲染速度。但是在老架构到新架构的迁移过程中容易导致问题。

但是这个问题的根因是什么导致的?
