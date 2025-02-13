# 小程序获取用户信息

https://developers.weixin.qq.com/community/develop/doc/00022c683e8a80b29bed2142b56c01?highLine=%25E5%25A4%25B4%25E5%2583%258F%25E6%2598%25B5%25E7%25A7%25B0

原本小程序可以在用户授权后获取用户信息，但是2022年11月微信下掉了这个api，现在只能通过wx.login获取唯一标识，但是微信文档写的太不明显了，不好好看完全不知道这个改动。