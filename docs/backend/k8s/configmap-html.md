---
title: 使用 ConfigMap 存储 HTML 页面
sidebar_position: 2
tags: [kubernetes, k8s, configmap, nginx, devops]
---

# 使用 ConfigMap 存储 HTML 页面

这是一个非常标准且经典的 **K8s "Hello World" 进阶版** 配置文件！

它不仅仅是跑了一个 Nginx，还用到了 **ConfigMap** 来挂载自定义页面，这比单纯跑个镜像要高级得多。这意味着你**不需要重新打包 Docker 镜像，就能修改网页内容**。

## 完整 YAML 配置

这个 YAML 文件通过 `---` 分隔符，在一个文件里定义了三个资源：

```yaml
# ConfigMap 存储 HTML 页面
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-html
data:
  index.html: |
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hello Kubernetes!</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                text-align: center;
                padding: 40px;
                background: rgba(255,255,255,0.1);
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }
            h1 { font-size: 3em; margin-bottom: 10px; }
            p { font-size: 1.2em; opacity: 0.9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Hello Kubernetes!</h1>
            <p>Nginx is running on Minikube</p>
            <p>Pod: <strong id="hostname"></strong></p>
        </div>
    </body>
    </html>

---
# Deployment 运行 nginx
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-demo
  labels:
    app: nginx-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx-demo
  template:
    metadata:
      labels:
        app: nginx-demo
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: html-volume
          mountPath: /usr/share/nginx/html
      volumes:
      - name: html-volume
        configMap:
          name: nginx-html

---
# Service 暴露服务
apiVersion: v1
kind: Service
metadata:
  name: nginx-demo
spec:
  type: NodePort
  selector:
    app: nginx-demo
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080
```

---

## 第一部分：ConfigMap（粮草/素材）

**作用**：把网页的 HTML 代码存到 K8s 的数据库里，而不是写死在镜像里。

```yaml
apiVersion: v1
kind: ConfigMap        # 类型：配置字典
metadata:
  name: nginx-html     # 名字叫 nginx-html，记住这个名字，后面要用
data:
  index.html: |        # Key是文件名，Value是文件内容
    <!DOCTYPE html>... # (这里是一段带有紫色背景的精美网页代码)
```

**解读**：
- 通常 Nginx 镜像里自带一个丑丑的"Welcome to nginx!"页面。
- 这里我们把一段漂亮的 HTML 代码定义成了一个变量，名字叫 `index.html`。
- **比喻**：这就像你写了一张**"今日特价菜单"**（index.html），先放在经理的保险柜（K8s ConfigMap）里，还没贴到墙上。

---

## 第二部分：Deployment（厨师/运行逻辑）

**作用**：启动 Nginx 容器，并把上面的"菜单"贴进容器里。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-demo     # 部署的名字
spec:
  replicas: 1          # 只要 1 个副本（1个Pod）
  selector: ...
  template:
    metadata: ...
    spec:
      containers:
      - name: nginx
        image: nginx:alpine  # 使用轻量级的 Nginx 镜像
        ports:
        - containerPort: 80

        # 核心魔法在这里！挂载动作
        volumeMounts:
        - name: html-volume              # 1. 找到下面定义的那个卷
          mountPath: /usr/share/nginx/html # 2. 把它挂载到容器内的这个路径

      # 定义数据卷
      volumes:
      - name: html-volume                # 给这个卷起个内部名叫 html-volume
        configMap:
          name: nginx-html               # 3. 这个卷的内容来源是第一部分的 ConfigMap
```

**核心逻辑（Volume 挂载）**：
- **`volumes`（数据卷）**：告诉 Pod，"我要用第一部分的那个 `nginx-html` ConfigMap，把它变成一个虚拟硬盘"。
- **`volumeMounts`（挂载点）**：告诉容器，"把这个虚拟硬盘插到 `/usr/share/nginx/html` 这个目录下"。
- **结果**：Nginx 启动时，去 `/usr/share/nginx/html` 找网页文件，发现那里正好躺着我们定义的 `index.html`。
- **比喻**：经理告诉服务员（Pod）：去保险柜拿那张"今日特价菜单"（Volume），然后贴到门口的展示架上（MountPath）。

---

## 第三部分：Service（前台/大门）

**作用**：在墙上开个洞，让你可以通过浏览器访问。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-demo     # 服务的名字
spec:
  type: NodePort       # 类型：节点端口（在宿主机开端口）
  selector:
    app: nginx-demo    # 只要标签是 nginx-demo 的 Pod，我都负责转发
  ports:
  - port: 80           # Service 内部的虚拟端口
    targetPort: 80     # Pod 容器里的端口（Nginx默认是80）
    nodePort: 30080    # 对外暴露的物理端口（重点！）
```

**解读**：
- **`type: NodePort`**：这是开发测试常用的方式。它会在 K8s 集群的每一台机器上都打开一个端口。
- **`nodePort: 30080`**：这表示你可以通过 `http://<你的机器IP>:30080` 直接访问这个网页。
- **比喻**：本来餐厅是在大楼内部（ClusterIP），外人进不来。现在你在大楼外墙开了个窗口（NodePort），写着"30080窗口"，顾客从这个窗口就能买到餐。

---

## 总结：数据流向

1. **ConfigMap** 保存了 HTML 代码。
2. **Deployment** 启动 Nginx，并把 ConfigMap 里的 HTML **覆盖** 到了 Nginx 的默认网页目录下。
3. **Service** 监听了宿主机的 **30080** 端口，把流量转发给 Nginx 的 80 端口。

---

## 如何运行它？

既然你已经有这个文件了（假设保存为 `web.yaml`）：

### 1. 应用配置

```bash
kubectl apply -f web.yaml
```

### 2. 查看状态

```bash
kubectl get pod,svc
```

### 3. 访问网页

- 如果你是在本机用 Docker Desktop 或 Minikube，你可以直接浏览器访问：`http://localhost:30080` 或者 `http://127.0.0.1:30080`。
- 如果是 Minikube，有时候需要运行 `minikube service nginx-demo` 来自动打开浏览器。

你可以直接把这段 YAML 保存下来去试一下，你会看到一个紫色的"Hello Kubernetes"网页！
