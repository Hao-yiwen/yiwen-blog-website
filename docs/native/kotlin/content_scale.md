# ContentScale

ContentScale 是一个用于图片组件（如 Image）的属性，它指定了图片如何适应或填充其容器的尺寸。

## 介绍

-   `Crop`: 图片将被缩放（保持其宽高比）以完全覆盖容器，如果图片的宽高比与容器不匹配，则图片的某些部分会被裁剪以确保图片填满整个容器。
-   `Fit`: 保持宽高比，确保图片的完整可见性而不裁剪
-   `FillBounds`: 填充整个容器但不保持宽高比，可能导致图片扭曲

## ContentScale.Crop 的工作方式

-   保持宽高比：缩放图片时保持原始宽高比不变。
-   填充容器：图片被缩放到足以完全覆盖容器的尺寸。这意味着图片的某些部分可能会超出容器的边界，从而不在视图中显示。
-   裁剪：为了使图片完全覆盖容器，超出容器边界的图片部分将被裁剪掉。

## 示例

```kt
// 图片将被裁剪以填充Image组件的尺寸
Image(
    painter = painterResource(id = R.drawable.my_image),
    contentDescription = "Example Image",
    contentScale = ContentScale.Crop
)
// 圆形头像
Image(
    modifier = modifier
        .size(dimensionResource(R.dimen.image_size))
        .padding(dimensionResource(R.dimen.padding_small))
        .clip(MaterialTheme.shapes.small),
    contentScale = ContentScale.Crop,
    painter = painterResource(dogIcon),
    contentDescription = null
)
```
