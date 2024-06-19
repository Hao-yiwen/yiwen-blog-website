# 创建一个新的Pod模块

在使用 CocoaPods 创建新模块（Pod）时，你需要按照以下步骤操作。这个过程包括创建一个新的 Podspec 文件，配置你的模块，发布到 CocoaPods 仓库等。

## 1.安装rbenv和ruby@3.0.0

[文档](./rbenv.md)

## 2.创建一个新的 Pod

```bash
pod lib create MyNewPod
```

## 3. 配置 Podspec 文件

```ruby
Pod::Spec.new do |s|
  s.name             = 'MyNewPod'
  s.version          = '0.1.0'
  s.summary          = 'A short description of MyNewPod.'
  s.description      = <<-DESC
                       A longer description of MyNewPod in greater detail.
                       DESC
  s.homepage         = 'https://example.com/MyNewPod'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Your Name' => 'you@example.com' }
  s.source           = { :git => 'https://github.com/username/MyNewPod.git', :tag => s.version.to_s }
  s.ios.deployment_target = '10.0'

  s.source_files = 'MyNewPod/Classes/**/*'
  s.resource_bundles = {
    'MyNewPod' => ['MyNewPod/Assets/*.png']
  }

  s.public_header_files = 'MyNewPod/Classes/**/*.h'
  s.dependency 'AFNetworking', '~> 3.0'
end
```

## 4. 添加pod

```ruby
pod 'MyNewPod', :path => '/path/to/MyNewPod'
```
