---
title: Antd中Menu中选中项移动到屏幕中间位置
sidebar_label: Antd中Menu中选中项移动到屏幕中间位置
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Antd中Menu中选中项移动到屏幕中间位置

```js
useEffect(() => {
    if (selectedKey) {
        const menuElement = document.getElementById(`${selectedKey[0]}`); // 需要再items中添加id字段
        if (menuElement) {
            menuElement.scrollIntoView({
                behavior: 'smooth', // 定义滚动行为为平滑滚动
                block: 'center', // 定义垂直方向滚动时，元素位于视图的中间位置
                inline: 'nearest', // 定义水平方向滚动时，元素位于视图的最近边
            });
        }
    }
}, [selectedKey]);

<Menu
    ref={menuRef}
    mode="inline"
    className={classes.menu}
    openKeys={activeOpenkeys}
    defaultOpenKeys={['@ctrip/xtaro-platform-component']}
    onOpenChange={onOpenChange}
    style={{ height: '100%', overflowY: 'auto' }}
    items={activeComponents}
    selectedKeys={selectedKey}
    onSelect={onSelect}
/>;
```
