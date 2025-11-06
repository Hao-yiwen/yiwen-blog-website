---
title: Android中的动画
sidebar_label: Android中的动画
date: 2024-07-25
last_update:
  date: 2024-07-25
---

# Android中的动画

动画一直是前端和移动端开发中比较头痛的一块内容，因为动画往往需要编写复杂的内容，而其底层也往往非常繁杂。以下是对android中动画的一些总结。

## 视图动画

该动画使用于早期的android开发中，只能对view操作。需要编写大量代码，目前使用不多。

```java
View view2 = findViewById(R.id.view_two);
Animation animation = AnimationUtils.loadAnimation(this, R.anim.translate);
view2.startAnimation(animation);
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<translate xmlns:android="http://schemas.android.com/apk/res/android"
    android:fromXDelta="0%"
    android:toXDelta="100%"
    android:duration="1000" />
```

## 属性动画

属性动画现在是android主推的动画，可对大多数属性进行操作。

```java
ObjectAnimator animator = ObjectAnimator.ofFloat(view, "translationX", 0, 200);
        animator.setDuration(2000);
        animator.start();
```

## 帧动画

每一帧都市不同样式。

```xml
<?xml version="1.0" encoding="utf-8"?>
<animation-list xmlns:android="http://schemas.android.com/apk/res/android"
    android:oneshot="false">

    <item
        android:drawable="@mipmap/bga_refresh_loading01"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading02"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading03"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading04"
        android:duration="100" />


    <item
        android:drawable="@mipmap/bga_refresh_loading05"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading06"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading07"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading08"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading09"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading10"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading11"
        android:duration="100" />

    <item
        android:drawable="@mipmap/bga_refresh_loading12"
        android:duration="100" />

</animation-list>
```

## 矢量图动画

```xml
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="24dp"
    android:height="24dp"
    android:viewportWidth="24"
    android:viewportHeight="24">
    <path
        android:name="heart"
        android:fillColor="#FF0000"
        android:pathData="M12,21.35l-1.45-1.32C5.4,15.36,2,12.28,2,8.5 2,5.42,4.42,3,7.5,3c1.74,0,3.41,0.81,4.5,2.09C13.09,3.81,14.76,3,16.5,3 19.58,3,22,5.42,22,8.5c0,3.78-3.4,6.86-8.55,11.54L12,21.35z"/>
</vector>

<animated-vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:drawable="@drawable/ic_heart">
    <target
        android:name="heart"
        android:animation="@animator/heart_anim" />
</animated-vector>

<?xml version="1.0" encoding="utf-8"?>
<set xmlns:android="http://schemas.android.com/apk/res/android">
    <objectAnimator
        xmlns:android="http://schemas.android.com/apk/res/android"
        android:propertyName="fillAlpha"
        android:valueFrom="0"
        android:valueTo="1"
        android:repeatCount="infinite"
        android:repeatMode="reverse"
        android:duration="1000" />
</set>
```

```java
ImageView img1 = findViewById(R.id.iv_heart);
img1.setImageResource(R.drawable.anim_heart);
AnimatedVectorDrawable drawable = (AnimatedVectorDrawable) img1.getDrawable();
drawable.start();
```