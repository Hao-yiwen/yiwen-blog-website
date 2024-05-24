# codepush

Microsoft CodePush 是一个云服务，用于实现 React Native 和 Cordova 应用的即时更新。通过 CodePush，你可以在不通过应用商店发布新版本的情况下，将 JavaScript、HTML 和 CSS 更新推送到用户的设备上。这种能力对于快速修复 Bug、推出新功能和改善用户体验非常有用。

主要功能

1. 即时更新：
   • 允许开发者将新代码和资产立即发布到用户设备，而无需重新提交应用商店审核。
   • 避免了传统发布流程中的延迟。
2. 版本管理：
   • 支持多版本管理，允许你选择性地向特定用户群体推送更新。
   • 通过“目标二进制版本”控制哪些用户接收到更新。
3. 更新策略：
   • Install Modes：定义更新的安装方式，包括立即安装、下次启动时安装或下次恢复时安装。
   • Mandatory vs Optional Updates：控制更新是否必须安装。
4. 更新对话框：
   • 提供自定义更新对话框，允许用户选择何时安装更新。
   • 提供详细的更新说明，帮助用户了解新版本的变化。
5. 回滚支持：
   • 如果更新出现问题，可以快速回滚到之前的版本。

## 文档

[codepush文档](https://learn.microsoft.com/zh-cn/appcenter/distribution/codepush/rn-plugin)

## Demo

[copush demo](https://github.com/Hao-yiwen/codePushDemo)
