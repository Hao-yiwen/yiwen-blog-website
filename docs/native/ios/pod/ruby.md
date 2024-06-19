# ruby安装踩坑

## openssl安装

~~`20231218`因为`openssl`版本问题，导致`rvm`管理ruby版本失败，总结经验发现`openssl`版本需要`1.1.1`~~
~~现在不用rvm安装依赖了，已废弃，使用rbenv~~

## 使用rvm安装ruby@2.7.x版本

```bash
export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@1.1/lib/pkgconfig"

rvm reinstall 2.7.2 --with-openssl-dir=/opt/homebrew/opt/openssl@1.1
```

## faq

![RN底层图](../../../static/img/ruby_error.png)

解决办法：

```bash
bundle install
```

:::tip
为什么`pod install`前要进行`bundle i`?

在 iOS 项目中执行 bundle i（即 bundle install 的缩写形式）是为了安装和管理 Ruby 依赖，这通常与使用 CocoaPods 管理 iOS 项目的依赖有关。以下是为什么要执行 bundle install 的一些原因：

-   版本一致性：bundle install 会根据 Gemfile 中列出的依赖和版本号安装 Ruby gems。这有助于确保项目中使用的 gem 版本在所有开发者和环境中保持一致，避免了“在我机器上可以运行”的问题。

-   CocoaPods 管理：对于 iOS 和 macOS 开发，CocoaPods 是一个常见的依赖管理工具，它用 Ruby 编写。使用 bundle install 确保正确版本的 CocoaPods 和其他相关依赖被安装，这对项目的构建和维护非常重要。

-   自动化和工具链：在复杂的开发环境中，可能会用到其他 Ruby 工具或脚本（例如 Fastlane）。使用 Bundler 管理这些工具的版本可以减少兼容性问题。

-   项目依赖隔离：Bundler 通过创建一个独立的环境（称为 "bundle"）来管理特定项目的依赖，这有助于隔离不同项目之间的依赖，避免版本冲突。

-   易于部署和持续集成：在持续集成（CI）流程中，bundle install 可以确保自动化构建过程中使用的依赖与开发者本地使用的一致。

总之，使用 bundle install 有助于确保 Ruby 依赖的一致性和项目构建的可靠性。对于使用 CocoaPods 的 iOS 项目来说，这是一个重要的步骤。
:::
