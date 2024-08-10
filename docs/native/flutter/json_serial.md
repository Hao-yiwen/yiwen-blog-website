# flutter离奇问题

## json_serializable无法生成.g.dart问题

在第一次尝试使用json_serializable时候无法生成.g.dart文件，翻阅了很多问题答案都是没有找到答案，后面揣摩了一下发现一个现象，以此来记录一下。

如果model类是中是`part 'user.g.dart'`,那么这个model类必须叫做`user.dart`，如果是`User.dart`则始终不会生成.g.dart文件。

## if you do not see the flutter application running, it might have crashed. the device logs (e.g. from adb or xcode) might have more details. if you do see the flutter application running on the device, try re-running with --host-vmservice-port to use a specific port known to be available.

今天因为要介入firebase，然后因为网络所以挂了全局代理，然后晚上再跑debug包就一直有这个报错，然后经过长达1个多小时的问题排查，后面定位到了系统代理中。

所以在使用flutter进行本地开发的时候，遇到问题还是要思考一下是否是代理问题。
