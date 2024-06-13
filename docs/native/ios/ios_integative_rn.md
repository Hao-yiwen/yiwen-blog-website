# ios集成rn

## 文档

[官方集成文档](https://reactnative.dev/docs/integration-with-existing-apps)

## 官方集成问题

官方集成文档需要将ios项目放在ios文件夹中，需要固定的格式，也就是js代码必须在ios外面。这对原生工程结构是一个挑战。于是现阶段将package.json放在项目工程的根目录，既有了自动连接，而且也能正常添加依赖。

## Podfile

```rb
require Pod::Executable.execute_command('node', ['-p',
  'require.resolve(
    "react-native/scripts/react_native_pods.rb",
    {paths: [process.argv[1]]},
  )', __dir__]).strip

platform :ios, min_ios_version_supported
prepare_react_native_project!

linkage = ENV['USE_FRAMEWORKS']
if linkage != nil
  Pod::UI.puts "Configuring Pod with #{linkage}ally linked Frameworks".green
  use_frameworks! :linkage => linkage.to_sym
end

target 'ios-study' do
  config = use_native_modules!

  use_react_native!(
    :path => "./node_modules/react-native",
    # An absolute path to your application root.
    :app_path => "#{Pod::Config.instance.installation_root}",
    :hermes_enabled => true
  )

  # Pods for ios-study
  pod 'TestModule', :path => './TestModule'
  pod 'SwiftUIModule', :path => './SwiftUIModule'

  post_install do |installer|
    # https://github.com/facebook/react-native/blob/main/packages/react-native/scripts/react_native_pods.rb#L197-L202
    react_native_post_install(
      installer,
      "./node_modules/react-native",
      :mac_catalyst_enabled => false,
      # :ccache_enabled => true
    )
  end
end

```

## 示例工程

(示例工程)[https://github.com/Hao-yiwen/ios-study]
