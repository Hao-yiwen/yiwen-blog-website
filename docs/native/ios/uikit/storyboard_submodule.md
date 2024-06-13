import substoryboard_controller from '@site/static/img/substoryboard_controller.png'
import storyboard_copu_bundle from '@site/static/img/storyboard_copu_bundle.png'

# 引用子模块的storyboard

有时候有些storyboard在子模块，而父模块要去使用storyboard refrence去引用这些模块，但是有时候是找不到，有时候各种报错。

## 步骤

-   storyboard ID一定要和Class一样，不然就会有controller找不到问题

<img src={storyboard_copu_bundle} width={500} />

<img src={storyboard_copu_bundle} />