# 用原生或者js驱动动画

```tsx
import React, { useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Button,
    Animated,
    Dimensions,
} from 'react-native';

const Index = props => {
    const scrollviewJSDriverValue = useRef(new Animated.Value(0)).current;
    const scrollviewNativeDriverValue = useRef(new Animated.Value(0)).current;
    return (
        <View style={{ flex: 1 }}>
            <Animated.View
                style={{
                    width: 60,
                    height: 60,
                    backgroundColor: scrollviewJSDriverValue.interpolate({
                        inputRange: [0, 5000],
                        outputRange: ['#ff00ff', '#00ff00'],
                    }),
                }}
            ></Animated.View>
            <Animated.View
                style={{
                    width: 60,
                    height: 60,
                    backgroundColor: 'yellow',
                    opacity: scrollviewNativeDriverValue.interpolate({
                        inputRange: [0, 5000],
                        outputRange: [1, 0],
                    }),
                }}
            ></Animated.View>
            <Animated.ScrollView
                onScroll={Animated.event(
                    [
                        {
                            nativeEvent: {
                                contentOffset: {
                                    y: scrollviewNativeDriverValue,
                                },
                            },
                        },
                    ],
                    {
                        useNativeDriver: true,
                        listener: event => {
                            scrollviewJSDriverValue.setValue(
                                event.nativeEvent.contentOffset.y
                            );
                        },
                    }
                )}
            >
                {Array.from({ length: 100 }).map((_, index) => (
                    <View
                        key={index}
                        style={{
                            height: 50,
                            backgroundColor: 'pink',
                            marginBottom: 10,
                            width: Dimensions.get('screen').width,
                        }}
                    >
                        <Text>{index}</Text>
                    </View>
                ))}
            </Animated.ScrollView>
        </View>
    );
};

export default Index;
```

经过测试发现，scrollview每次只能绑定一个原生属性其余的手动赋值的都是js驱动。
