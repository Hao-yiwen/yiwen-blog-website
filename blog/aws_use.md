# aws中ec2登录

今天因为需要购买了一台ec2服务器，但是在登录实例的时候发现始终无法登录，后面对登录进行了研究，ec2不同于国内的ecs等服务器，ec2的登录只能使用密钥对登录，而如果在创建实例的时候没有设置密钥对，那么则需要自己在服务器中设置，而不是控制面板。以下是详细流程:

```bash
# 本地创建密钥对
ssh-keygen -t rsa -b 2048 -f ~/my-aws-key

# 获取公钥内容
cat ~/my-aws-key.pub

# 将公钥添加到 EC2 实例
vim ~/.ssh/authorized_keys

# 添加公钥内容

# 本地登录
ssh -i ~/my-aws-key ec2-user@instance-ip
```