# Animated

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