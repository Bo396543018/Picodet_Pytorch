# PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{picodet,
  title={{PP-PicoDet}: A Better Real-Time Object Detector on Mobile Devices},
  author={Guanghua Yu, Qinyao Chang, Wenyu Lv, Chang Xu, Cheng Cui, Wei Ji, Qingqing Dang, Kaipeng Deng, Guanzhong Wang, Yuning Du, Baohua Lai, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma},
  journal={arXiv preprint arXiv:2111.00902},
  year={2021}
}
```

## Backbone Pretrained Weights
- [ESNet_0.75](https://drive.google.com/file/d/1DdIey-J64e8cl17uuIHUUZhfH-g1vXKr/view?usp=sharing)
- [ESNet_1.0x](https://drive.google.com/file/d/1sgjKUQ6tjm-jZoYCM3yCe914m1qnTy1n/view?usp=sharing)
- [ESNet_1.25x](https://drive.google.com/file/d/1bCpdK1GCRX3LzyuRafmtv4_ZT-ZCy5kt/view?usp=sharing)


## Results and Models

| Bakcbone  | size|box AP(ppdet) | Config|Download
|:---------:|:-------:|:-------:|:-------:|:-------:|
|picodet-s|320|26.9(27.1)| [config](https://github.com/Bo396543018/Picodet_Pytorch/tree/picodet/configs/picodet/picodet_s_320_coco.py)|[model](https://drive.google.com/file/d/1o6Vxhs9JpiFc87uZA5woaIp8woDHA_jT/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ctIuAbSl1afuiQW0Bkmm82jNh6CR_SOl/view?usp=sharing)|
|picodet-s|416|30.6(30.6)| [config](https://github.com/Bo396543018/Picodet_Pytorch/tree/picodet/configs/picodet/picodet_s_416_coco.py)|[model](https://drive.google.com/file/d/1mnbQ2Fex1v5Hn_MZbZYMZCq6Ol6Bgp4D/view?usp=sharing) \| [log](https://drive.google.com/file/d/1OzPHakomEPFtSmJhf0_Qh35-njHiRJCB/view?usp=sharing)|
|picodet-m|416|34.2(34.3)| [config](https://github.com/Bo396543018/Picodet_Pytorch/tree/picodet/configs/picodet/picodet_m_416_coco.py)|[model](https://drive.google.com/file/d/17jH2kzNBCuKzD39OOYdkyjWcIw9BMpgo/view?usp=sharing) \| [log](https://drive.google.com/file/d/1gIYzoPqRqmoY2-nydedfNz4Mk3TlrRRw/view?usp=sharing)|
|picodet-l|640|40.4(40.9)| [config](https://github.com/Bo396543018/Picodet_Pytorch/tree/picodet/configs/picodet/picodet_l_640_coco.py)|[model](https://drive.google.com/file/d/13x1uMQf8RXlVIjBen7KUxZMHSuUeAWVe/view?usp=sharing) \| [log](https://drive.google.com/file/d/1RWC0128oWtJt825JQBEy5fMnuHrJT-bi/view?usp=sharing)|


## Usage

### Install MMdetection
Our implementation is based on mmdetection. 
Install mmdetection according to [INSTALL](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

Note: Make sure your mmcv-full version is consistency with mmdet version(we use mmcv==1.4.0)


### Train

1. Download pretrained backbone using the link above

2. training

```
bash tools/dist_train.sh ./configs/picodet/picodet_s_320_coco.py 4
```

### Test
```
bash tools/dist_test.sh $CONFIG_PATH $TRAINED_MODEL_PATH $GPU_NUMS --eval bbox

eg. use picodet-s 320 pretrianed model
bash tools/dist_test.sh ./configs/picodet/picodet_s_320_coco.py $MODEL_DIR/picodet_s_320.26.9.pth 8 --eval bbox

Evaluating bbox...
Loading and preparing results...
DONE (t=1.76s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=43.50s).
Accumulating evaluation results...
DONE (t=14.63s).

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.408
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.138
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.684
```


## Deploy

TODO：
- [ ] mnn deploy