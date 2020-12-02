# yolo3-目标检测

## 一、主要思路

![img](https://upload-images.jianshu.io/upload_images/2709767-bc5def91d05e4d3a.png?imageMogr2/auto-orient/strip|imageView2/2)

​																												整体框架图

## ![https://img-blog.csdnimg.cn/20191020111518954.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center](https://img-blog.csdnimg.cn/20191020111518954.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

YOLOv3相比于之前的yolo1和yolo2，改进较大，主要改进方向有：

1、主干网络修改为darknet53，其重要特点是使用了**残差网络Residual**，darknet53中的残差卷积就是**进行一次3X3、步长为2的卷积，然后保存该卷积layer，再进行一次1X1的卷积和一次3X3的卷积，并把这个结果加上layer作为最后的结果**， 残差网络的特点是**容易优化**，并且能够通过增加相当的**深度来提高准确率**。其内部的**残差块使用了跳跃连接，缓解了在深度神经网络中增加深度带来的梯度消失问题。**代码如下：

```
#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x
```

![image-20201201153700656](C:/Users/YYQ/AppData/Roaming/Typora/typora-user-images/image-20201201153700656.png)

![img](https://upload-images.jianshu.io/upload_images/2709767-bb96f29a37be1651.png?imageMogr2/auto-orient/strip|imageView2/2)

  																								残差网络结构

2、darknet53的每一个卷积部分**使用了特有的DarknetConv2D结构**，每一次卷积的时候进行l2正则化，**完成卷积后进行BatchNormalization标准化与LeakyReLU。**代码如下所示：

```
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

```

![image-20201201154329727](C:/Users/YYQ/AppData/Roaming/Typora/typora-user-images/image-20201201154329727.png)

**普通的ReLU是将所有的负值都设为零，Leaky ReLU则是给所有负值赋予一个非零斜率。以数学的方式我们可以表示为**：

![img](https://img-blog.csdnimg.cn/20191127153502635.png#pic_center) 

### 1.实现主干部分darknet部分：

**注意：文中使用@wraps(Conv2D)函数，简单说就是一种继承父类的算法，对子类方法进行重写的过程，具体可以看一下链接：**

[https://blog.csdn.net/weixin_38495031/article/details/105357313]: @wraps(Conv2D)函数

其余部分只是正常的卷积，主要的还是残差网络部分。

###### 代码实现：

```
from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    # 输入【batch_size,416,416,3】
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 输入【batch_size,416,416,32】
    x = resblock_body(x, 64, 1)
    # 输入【batch_size,208,208,64】
    x = resblock_body(x, 128, 2)
    # 输入【batch_size,104,104,128】
    x = resblock_body(x, 256, 8)
    # 输入【batch_size,52,52,256】
    feat1 = x
    # 输入【batch_size,52,52,256】
    x = resblock_body(x, 512, 8)
    # 输入【batch_size,26,26,512】
    feat2 = x
    # 输入【batch_size,26,26,512】
    x = resblock_body(x, 1024, 4)
    # 输入【batch_size,13,13,1024】
    feat3 = x
    return feat1,feat2,feat3


```

## 二、从特征预测结果

![img](https://img-blog.csdnimg.cn/20191020111518954.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

1、在特征利用部分，yolo3提取**多特征层进行目标检测**，一共**提取三个特征层**，三个特征层位于主干部分darknet53的不同位置，分别位于中间层，中下层，底层，三个特征层的shape分别为(52,52,256)、(26,26,512)、(13,13,1024)。代码如下所示：

```

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    # 输入【batch_size,416,416,3】
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 输入【batch_size,416,416,32】
    x = resblock_body(x, 64, 1)
    # 输入【batch_size,208,208,64】
    x = resblock_body(x, 128, 2)
    # 输入【batch_size,104,104,128】
    x = resblock_body(x, 256, 8)
    # 输入【batch_size,52,52,256】
    feat1 = x
    # 输入【batch_size,52,52,256】
    x = resblock_body(x, 512, 8)
    # 输入【batch_size,26,26,512】
    feat2 = x
    # 输入【batch_size,26,26,512】
    x = resblock_body(x, 1024, 4)
    # 输入【batch_size,13,13,1024】
    feat3 = x
    return feat1,feat2,feat3
```

![image-20201201161159305](C:/Users/YYQ/AppData/Roaming/Typora/typora-user-images/image-20201201161159305.png)

2、三个特征层进行5次卷积处理，处理完后**一部分用于输出该特征层对应的预测结果**，**一部分用于进行反卷积UmSampling2d后与其它特征层进行结合**。代码如下：

```

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)
            
    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    feat1,feat2,feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # 第一个特征层
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    # 得到y1 = 【batch_size,13,13,255】
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    # 得到x=【batch_size,26,26,256】，feat2 = 【batch_size,26,26,512】
    x = Concatenate()([x,feat2])
    # 得到x = 【batch_size,26,26,768】

    # 第二个特征层
    # y2=(batch_size,26,26,3,75)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
    # 得到y2 =【batch_size,26,26,255】，x = 【batch_size,26,26,256】
    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),# 得到x =【batch_size,26,26,128】
            UpSampling2D(2))(x)
    # 得到x =【batch_size,52,52,128】，feat1 = 【batch_size,52,52,256】
    x = Concatenate()([x,feat1])
    # 得到x =【batch_size,52,52,384】

    # 第三个特征层
    # y3=(batch_size,52,52,3,85)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    # 得到y2 =【batch_size,52,52,255】，x = 【batch_size,52,52,128】

    return Model(inputs, [y1,y2,y3])

```

3、输出层的shape分别为(**13,13,75**)，(**26,26,75**)，(**52,52,75**)，**最后一个维度为75是因为该图是基于voc数据集的，它的类为20种，剩下的5是指预选框的x，y，w，h和置信度，yolo3只有针对每一个特征层存在3个先验框，所以最后维度为3x25；如果使用的是coco训练集，类则为80种，最后的维度应该为255 = 3x85**，三个特征层的shape为(**13,13,255**)，(**26,26,255**)，(**52,52,255**)

其实际情况就是，输入N张416x416的图片，在经过多层的运算后，会输出三个shape分别为(**N,13,13,255**)，(**N,26,26,255**)，(**N,52,52,255**)的数据，对应每个图分为13x13、26x26、52x52的网格上3个先验框的位置。·

 实现代码如下：

```
from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from nets.darknet53 import darknet_body
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)
            
    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    feat1,feat2,feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat2])
    # 第二个特征层
    # y2=(batch_size,26,26,3,85)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat1])
    # 第三个特征层
    # y3=(batch_size,52,52,3,85)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

```

## 三、预测结果的解码

​		**其中的anchor表示**

[anchor框]: https://zhuanlan.zhihu.com/p/150332784

由第二步我们可以获得三个特征层的预测结果，shape分别为(**N,13,13,225)，(N,26,26,225)，(N,52,52,225)**的数据，对应每个图分为13x13、26x26、52x52的网格上3个预测框的位置。

![img](https://upload-images.jianshu.io/upload_images/2709767-5cc00a60004e0473.png?imageMogr2/auto-orient/strip|imageView2/2)

​		最后一个维度中的255包含了（4+1+80）*3，前面的4**分别代表x_offset、y_offset、h和w；1表示置信度；80表示分类结果（coco数据是80个种类），最后的3代表每个像素点有三个预选框。**yolo3的解码过程就是**将每个网格点加上它对应的x_offset和y_offset**，加完后的结果就是**预测框的中心**，然后**再利用先验框和h、w结合 计算出预测框的长和宽**。这样就能得到整个预测框的位置了。

![img](https://img-blog.csdn.net/2018091217215138?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![image-20201201175338600](C:/Users/YYQ/AppData/Roaming/Typora/typora-user-images/image-20201201175338600.png)

​		但是这个预测结果并不对应着最终的预测框在图片上的位置，还需要解码才可以完成。

​		此处要讲一下yolo3的预测原理，yolo3的3个特征层分别将整幅图分为13x13、26x26、52x52的网格，每个网络点负责一个区域的检测。感受一下9种先验框的尺寸，下图中蓝色框为聚类得到的先验框。黄色框式ground truth，红框是对象中心点所在的网格。

![img](https://upload-images.jianshu.io/upload_images/2709767-f7e39ba644a583e4.png?imageMogr2/auto-orient/strip|imageView2/2)

​		我们知道特征层的预测结果对应着三个预测框的位置，我们先将其reshape一下，其结果为(**N,13,13,3,85**)，(**N,26,26,3,85**)，(**N,52,52,3,85**)，其中的3代表三个预测框。

![img](https://upload-images.jianshu.io/upload_images/2709767-9b28d0f1c682b80a.png?imageMogr2/auto-orient/strip|imageView2/2)

将网络输出结果进行解码，目标位置预测，代码如下所示：

```

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    # -------------------------------对锚点进行重新定义维度------------------------------
    # 1，13，13，3，85
    num_anchors = len(anchors)
    # [1, 1, 1, 3, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    # ------------------------------方便与图像进行加减乘除计算-------------------------

    # --------------------------------创建13*13的网格-----------------------------------------
    # 获得x，y的网格
    # (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    # 得到grid_y = 【feats.height,feats.width , 1, 1]】
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    # 得到grid_y = 【feats.height, feats.width, 1, 1]】
    grid = K.concatenate([grid_x, grid_y])
    # 得到grid = 【feats.height,feats.width, 1, 2】
    grid = K.cast(grid, K.dtype(feats))
    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # ------------------------------------------------------------------------------

    # ------------------------------------对数据进行解码---------------------------------
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_xy = （每个网格点x与y的值 + x与y方向的偏移量）/13
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    # box_wh = （每个网格点w与h的指数 * 先验框）/416
    box_confidence = K.sigmoid(feats[..., 4:5])
    # 求置信度
    box_class_probs = K.sigmoid(feats[..., 5:])
    # 求种类
    # --------------------------------------------------------------------------------

    # -----------------------------------对求出的x y w h 置信度 种类 进行返回--------------
    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
    # ------------------------------------------------------------------------

```

​		找到特征图上的目标位置，还需将位置转化到真实图片上，代码如下：

```

# ---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # 重新定义参数名，方便计算
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    # 转换类型（float32 方便之后的指数运算）
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    # ----------------------将网络图片尺寸转换为真实图片尺寸（输入图像自动会加上灰边，固定尺寸为416*416）
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    # --------------------------------------------------------------------------------

    # ------------------------计算一个框的左上和右下的坐标点（已知中心点坐标）--------
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    # -------------------------------------------------------------------------------------

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

```

计算每个预测框的得分，代码如下：

```

#---------------------------------------------------#
#   获取每个box和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # --------------------------对网络得到的数据进行解码，得到图片真实的x y w h 置信度和种类-----------
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -------------------------------------------------------------------------------------

    # -------------------------------------计算真实图片上的目标框的坐标---------
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # ----------------------------------------------------------------------------
    # -----------------------------------将计算出的结果进行排列--------------
    # 获得得分和box
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    # ----------------------------------------------------------------------
    return boxes, box_scores

```

将得到的每个预测框和得分，都排成一维数组，使用非极大值抑制法对所有预测框进行抑制，选出得分最高的框，代码如下：

```
# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # 获得特征层的数量
    num_layers = len(yolo_outputs)
    # 特征层1对应的anchor是678（13，13）
    # 特征层2对应的anchor是345（26，26）
    # 特征层3对应的anchor是012（52，52）
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    # 416，416
    boxes = []
    box_scores = []
    # ---------------------------------对每一个特征层的结果进行处理-----------------------
    # 对每个特征层进行处理
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠（全部堆叠成一维）
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    # 每一个点对应80种类别的得分值

    # 通过每个得分和阈值进行比较，满足要求的就留下，并设置为1
    mask = box_scores >= score_threshold
    # 限制每个图片最大的边框数（20）
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，得分与种类
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

```

最后将得到的预测框进行绘制即可，代码如下:

```

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        start = timer()

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[0],self.model_image_size[1])# (416,416)
        # 将照片格式都统一为416，416
        boxed_image = letterbox_image(image, new_image_size)
        # 将图片数据转换为float数组
        image_data = np.array(boxed_image, dtype='float32')
        # 归归一化
        image_data /= 255.
        # 增加维度（416，416，1）
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic=[]

        # ------------------------对结果进行绘制-------------------------------
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

```



[^参考]: https://blog.csdn.net/leviopku/article/details/82660381

