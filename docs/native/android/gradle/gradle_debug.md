import gradle_pic from '@site/static/img/gradle_debug.png'

# Gradle debug调试

每次看到复杂的gradle脚本。我就在思考，难道这些脚本不能调试吗，虽然脚本调试可能会有一点点麻烦，但是理论上来讲，应该问题不大，因为他是基于jvm的gradle啊，所以今天让我们看看在android studio中研究一下如何调试gradle脚本。

## 启动守护线程

```bash
./gradlew assembleDebug -Dorg.gradle.debug=true --no-daemon
```

## 创建attach remote jvm配置

<img src={gradle_pic} width={600}/>

## 点击启动按钮启动attach remote jvm
