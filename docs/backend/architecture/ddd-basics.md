---
title: 领域驱动设计 (DDD) 基础
sidebar_label: DDD 基础
date: 2024-12-30
tags: [ddd, architecture, domain-driven-design, 架构设计]
---

# 领域驱动设计 (DDD) 基础

**领域驱动设计 (Domain-Driven Design, 简称 DDD)** 是一种主要用于解决**复杂业务问题**的软件开发思想和方法论。它由 Eric Evans 在 2003 年提出，核心理念是：**软件的核心是其解决的领域问题，而非使用的技术。**

DDD 不是一种特定的框架（如 Spring），而是一种**设计思维**。它试图通过建立一个高度对应业务现实的"领域模型"，来降低系统的复杂度，并让代码与业务保持一致。

为了帮助你全面理解 DDD，本文将从**核心理念**、**战略设计（宏观）**、**战术设计（微观）**以及**架构分层**四个维度进行详细介绍。

---

## 1. 核心理念：为什么要用 DDD？

在传统的"数据驱动设计"（CRUD）中，我们通常先设计数据库表，然后写代码去操作这些表。这在简单系统中很有效，但在复杂业务中（如电商、金融系统）会导致：

- **贫血模型 (Anemic Model)：** 对象只是一堆数据（Getter/Setter），业务逻辑散落在 Service 层，难以复用和维护。
- **语言鸿沟：** 开发人员说"表、字段、外键"，业务人员说"订单、客户、结算"，沟通成本极高。

**DDD 的解决方案：**

1. **统一语言 (Ubiquitous Language)：** 开发人员和业务专家使用同一套语言（术语）沟通，并直接反映在代码中。
2. **业务与技术解耦：** 核心业务逻辑不依赖于数据库、UI 或具体的框架。

---

## 2. 战略设计 (Strategic Design) —— 宏观规划

战略设计主要关注系统的**边界划分**和**顶层架构**，主要用于梳理庞大的业务系统。

### A. 通用语言 (Ubiquitous Language)

这是团队的公共词汇表。如果业务方说"冻结账户"，代码里的方法名就应该是 `freezeAccount()`，而不是 `updateStatus(2)`。

### B. 领域 (Domain) 与 子域 (Subdomain)

- **核心域 (Core Domain)：** 公司的核心竞争力（如淘宝的商品交易、搜索）。这是需要投入最强资源的领域。
- **支撑域 (Supporting Domain)：** 必不可少但非核心（如淘宝的字典数据管理）。
- **通用域 (Generic Domain)：** 行业通用的功能（如认证登录、发送邮件），通常可以购买现成方案。

### C. 界限上下文 (Bounded Context) **(最核心概念)**

这是 DDD 中解决复杂度的关键。同一个名词在不同场景下含义不同，因此需要划分边界。

> **例子：** "商品"这个词。
> - 在**销售上下文**中：关注价格、描述、图片。
> - 在**物流上下文**中：关注重量、体积、仓库位置。
>
> 在 DDD 中，我们不会建立一个包含所有属性的巨大 `Product` 类，而是在不同的上下文中建立不同的模型，通过 ID 关联。

### D. 上下文映射 (Context Mapping)

定义不同上下文之间的集成关系（如：防腐层 ACL、共享内核、遵奉者等）。

---

## 3. 战术设计 (Tactical Design) —— 微观实现

战术设计关注**代码层面**如何构建领域模型。

### A. 实体 (Entity)

- 具有**唯一标识 (ID)** 的对象。
- 其状态会随时间变化（生命周期）。
- **例子：** `User` (用户)，即使名字改了，ID 没变，他还是同一个人。

### B. 值对象 (Value Object)

- **没有唯一标识**，通过属性值来定义。
- 通常是**不可变 (Immutable)** 的。
- **例子：** `Address` (地址)、`Color` (颜色)。如果两个地址省市区街道都一样，它们就是同一个值，不需要 ID 区分。

### C. 聚合 (Aggregate) 与 聚合根 (Aggregate Root)

- **聚合：** 一组相关对象的集合，作为一个整体被修改。
- **聚合根：** 聚合中唯一的入口点（通常是一个实体）。
- **规则：** 外部对象只能引用聚合根，不能直接修改聚合内部的对象。这保证了数据的一致性。
- **例子：** "订单"是聚合根，"订单项"是内部实体。你不能直接去修改"订单项"，必须通过"订单"这个入口来操作（如 `order.removeItem()`），以便订单重新计算总价。

```go
// 聚合根示例
type Order struct {
    ID        OrderID
    Items     []OrderItem  // 内部实体
    TotalPrice Money
}

// 只能通过聚合根修改内部对象
func (o *Order) RemoveItem(itemID ItemID) error {
    // 移除订单项并重新计算总价
    // ...
    o.recalculateTotalPrice()
    return nil
}
```

### D. 领域服务 (Domain Service)

- 当某个逻辑不属于任何单一的实体或值对象时，将其放入领域服务。
- **例子：** "转账"，涉及两个账户实体的变化，适合放在领域服务中。

```go
// 领域服务示例
type TransferService struct {}

func (s *TransferService) Transfer(from, to *Account, amount Money) error {
    if err := from.Withdraw(amount); err != nil {
        return err
    }
    to.Deposit(amount)
    return nil
}
```

### E. 领域事件 (Domain Event)

- 表示业务中发生的**事实**。通常用于解耦，触发后续流程（如"支付成功"后触发"发货"）。

```go
// 领域事件示例
type OrderPaidEvent struct {
    OrderID   OrderID
    Amount    Money
    PaidAt    time.Time
}
```

### F. 仓储 (Repository)

- 用于持久化和检索聚合。它提供了类似集合的接口，对外屏蔽了数据库的具体实现（SQL、NoSQL）。

```go
// 仓储接口定义在领域层
type OrderRepository interface {
    FindByID(id OrderID) (*Order, error)
    Save(order *Order) error
    Delete(id OrderID) error
}

// 具体实现在基础设施层
type PostgresOrderRepository struct {
    db *sql.DB
}
```

---

## 4. DDD 分层架构 (Layered Architecture)

DDD 通常采用四层架构，核心原则是**依赖倒置**：核心域不依赖外部，而是基础设施依赖核心域。

| 层级 | 名称 | 职责 | 备注 |
| --- | --- | --- | --- |
| **1. 用户接口层** | User Interface | 处理用户请求，解析参数，返回视图/JSON。 | Controller 层 |
| **2. 应用层** | Application | **编排**业务流程，不包含核心业务逻辑。 | 调用领域服务、仓储等。非常薄。 |
| **3. 领域层** | **Domain** | **核心心脏**。包含实体、聚合、领域服务、业务规则。 | **不依赖任何框架或数据库技术。** |
| **4. 基础设施层** | Infrastructure | 提供技术实现（数据库、缓存、MQ、文件系统）。 | 实现仓储接口，作为插件支撑领域层。 |

```
┌─────────────────────────────────────┐
│         User Interface Layer        │  ← HTTP/gRPC handlers
├─────────────────────────────────────┤
│         Application Layer           │  ← Use cases, orchestration
├─────────────────────────────────────┤
│           Domain Layer              │  ← Entities, Value Objects,
│      (核心，不依赖外部技术)           │     Domain Services, Repositories
├─────────────────────────────────────┤
│        Infrastructure Layer         │  ← DB, Cache, MQ implementations
└─────────────────────────────────────┘
```

> **注意：** 除了传统分层，**六边形架构 (Hexagonal Architecture)** 和 **洋葱架构 (Onion Architecture)** 也是实现 DDD 的常见架构模式，它们更强调"领域内核"的独立性。

---

## 5. 项目目录结构示例

```
go-ddd/
├── cmd/                    # 应用入口
│   └── server/
│       └── main.go
├── internal/
│   ├── application/        # 应用层
│   │   └── order_service.go
│   ├── domain/             # 领域层（核心）
│   │   ├── order/
│   │   │   ├── entity.go       # 实体定义
│   │   │   ├── value_object.go # 值对象
│   │   │   ├── repository.go   # 仓储接口
│   │   │   └── service.go      # 领域服务
│   │   └── user/
│   │       └── ...
│   ├── infrastructure/     # 基础设施层
│   │   ├── persistence/
│   │   │   └── postgres_order_repo.go
│   │   └── messaging/
│   │       └── rabbitmq_publisher.go
│   └── interfaces/         # 用户接口层
│       ├── http/
│       │   └── order_handler.go
│       └── grpc/
│           └── order_server.go
└── pkg/                    # 公共库
    └── ...
```

---

## 6. 总结：何时使用 DDD？

DDD 并非银弹，它有很高的学习成本和开发复杂度。

### ✅ 适合使用 DDD 的场景

- **业务极其复杂：** 逻辑分支多，规则经常变动。
- **长期维护的项目：** 只有设计良好的模型才能经得起数年的迭代。
- **团队规模较大：** 需要统一语言，划分上下文并行开发。

### ❌ 不适合使用 DDD 的场景

- **简单的 CRUD 系统：** 仅仅是增删改查，强行上 DDD 会造成"过度设计"。
- **性能极其敏感的脚本：** DDD 的分层和抽象可能会带来微小的性能损耗。
- **短期外包项目：** 追求交付速度，而非长期可维护性。

---

## 7. 示例代码

完整的 Go 语言 DDD 实现示例可以参考：

- [go-ddd 示例项目](https://github.com/Hao-yiwen/go-examples/tree/master/go-ddd)

---

## 8. 参考资料

- Eric Evans - *Domain-Driven Design: Tackling Complexity in the Heart of Software* (2003)
- Vaughn Vernon - *Implementing Domain-Driven Design* (2013)
- [DDD Community](https://www.domainlanguage.com/)
