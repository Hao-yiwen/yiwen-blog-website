# OC中的枚举

## c枚举
```
typedef enum {
    Red,
    Green,
    Blue
} Color;
```

## NS_ENUM枚举
```
typedef NS_ENUM(NSInteger, Color1) {
    Red1,
    Green1,
    Blue1
};
```

## 区别
1. 类型检查

- 传统 C 风格枚举：
    - 在传统的 C 风格枚举中，枚举类型 Color 只是一个 int 类型。编译器不会对枚举值进行严格的类型检查。

- NS_ENUM：
    - NS_ENUM 定义的枚举类型在编译器中具有更强的类型检查。这有助于防止意外的类型错误。