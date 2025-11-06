---
title: Animated
sidebar_label: Animated
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Animated

## 原声驱动常见问题

在使用useNativeDriver: true时，确实，并非所有样式属性都支持通过原生驱动来执行动画。React Native中，原生动画支持的样式属性主要包括transform数组内的变换（如translateX, translateY, scale, 等）和opacity。这是因为这些属性可以直接在UI线程上高效地执行，而无需JavaScript线程的参与，从而提高了动画的性能和流畅度。

至于top、left、bottom和right这样的布局属性，它们并不在useNativeDriver: true支持的属性列表中。尝试通过原生驱动来动画这些属性将不会生效，并且可能会在控制台中看到警告信息。

backgroundColor属性并不受支持进行原生动画。React Native中原生驱动的动画支持主要限于transform属性（如translateX, translateY, scale等）和opacity。

## 动态背景颜色

```jsx
const ScrollAnimatedExample = () => {
    // 创建一个Animated.Value
    const scrollAnimate = useRef(new Animated.Value(0)).current;

    // 创建一个动态背景色，它会根据滚动位置变化
    const backgroundColor = scrollAnimate.interpolate({
        inputRange: [0, 1500], // 假设滚动范围为0到1500
        outputRange: ['rgb(255, 99, 71)', 'rgb(135, 206, 235)'], // 颜色从Tomato到SkyBlue
    });

    return (
        <ScrollView
            style={styles.container}
            onScroll={Animated.event(
                [{ nativeEvent: { contentOffset: { y: scrollAnimate } } }],
                { useNativeDriver: false } // 注意: backgroundColor不支持原生驱动，这里要设为false
            )}
            scrollEventThrottle={16} // 控制onScroll事件调用的频率
        >
            <Animated.View style={[styles.content, { backgroundColor }]}>
                <Text style={styles.text}>滚动我!</Text>
            </Animated.View>
        </ScrollView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    content: {
        height: 3000, // 内容足够高，以便有滚动效果
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        fontSize: 24,
        color: 'white',
    },
});

export default ScrollAnimatedExample;
```

在rn中可以通过animated来实现一些动画效果，例先声明一个Animated.value,然后再用interpolate进行背景颜色的映射。

## 根据scrollview的滚动距离控制右侧Animated.view的显示和移动

```
const App = () => {
  const [expanded, setExpanded] = useState(false);
  const scrollviewRef = React.useRef<ScrollView>(null);
  const scrollviewAnimatedValue = useRef(new Animated.Value(0)).current;

  return (
    <View style={style.container}>
      <Animated.ScrollView
        ref={scrollviewRef}
        scrollEventThrottle={16}
        onScroll={Animated.event(
          [
            {
              nativeEvent: {
                contentOffset: {
                  y: scrollviewAnimatedValue,
                },
              },
            },
          ],
          {
            useNativeDriver: true,
          },
        )}
        style={{
          width: Dimensions.get('screen').width * 0.5,
          overflow: 'hidden',
          height: '100%',
        }}>
        {Array.from({length: 100}).map((_, index) => (
          <View
            key={index}
            style={{
              width: Dimensions.get('screen').width * 0.5,
              height: 50,
              backgroundColor: 'pink',
              borderWidth: 0.5,
              borderColor: '#d6d7da',
              marginBottom: 10,
            }}>
            <Text>{index}</Text>
          </View>
        ))}
      </Animated.ScrollView>
      <View style={{width: Dimensions.get('screen').width * 0.5}}>
        {
          <Animated.View
            style={{
              backgroundColor: 'lightblue',
              /**
               * scrollviewAnimatedValue不能当做一个具体数字使用
               * 需要使用opactiy
               */
              display: scrollviewAnimatedValue > 500 ? 'flex' : 'none',
              transform: [
                {
                  translateY: scrollviewAnimatedValue.interpolate({
                    inputRange: [0, 5000],
                    outputRange: [0, 400],
                  }),
                },
              ],
            }}>
            <Text>Animated.View</Text>
          </Animated.View>
        }
        <TouchableOpacity
          onPress={() => {
            LayoutAnimation.configureNext(LayoutAnimation.Presets.spring);
            setExpanded(!expanded);
          }}>
          <Text>Press me to {expanded ? 'collapse' : 'expand'}!</Text>
        </TouchableOpacity>
        {expanded && (
          <View style={style.tile}>
            <Text>I disappear sometimes!</Text>
          </View>
        )}
        <Button
          title="Press me"
          onPress={() => {
            scrollviewRef.current?.scrollTo({
              y: 0,
              animated: true,
            });
          }}
        />
      </View>
    </View>
  );
};
```

:::danger
首先，要注意Animated.Value本身不会自动转化为一个可以直接在逻辑判断中使用的数字值。因此，scrollviewAnimatedValue > 500这样的比较实际上并不会按预期工作。如果你想根据滚动位置显示或隐藏Animated.View，你需要采用不同的策略，比如利用Animated的插值（interpolation）功能来控制opacity或其他样式属性，而不是display属性。
:::
