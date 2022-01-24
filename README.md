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
|picodet-s|320|26.9(27.1)| [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/picodet/picodet_s_320_coco.py)|[model](https://drive.google.com/file/d/1o6Vxhs9JpiFc87uZA5woaIp8woDHA_jT/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ctIuAbSl1afuiQW0Bkmm82jNh6CR_SOl/view?usp=sharing)|
|picodet-s|416|30.6(30.6)| [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/picodet/picodet_s_416_coco.py)|[model](https://drive.google.com/file/d/1mnbQ2Fex1v5Hn_MZbZYMZCq6Ol6Bgp4D/view?usp=sharing) \| [log](https://drive.google.com/file/d/1OzPHakomEPFtSmJhf0_Qh35-njHiRJCB/view?usp=sharing)|
|picodet-m|416|34.2(34.3)| [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/picodet/picodet_m_416_coco.py)|[model](https://drive.google.com/file/d/17jH2kzNBCuKzD39OOYdkyjWcIw9BMpgo/view?usp=sharing) \| [log](https://drive.google.com/file/d/1gIYzoPqRqmoY2-nydedfNz4Mk3TlrRRw/view?usp=sharing)|
|picodet-l|640|40.4(40.9)| [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/picodet/picodet_l_640_coco.py)|[model](https://drive.google.com/file/d/13x1uMQf8RXlVIjBen7KUxZMHSuUeAWVe/view?usp=sharing) \| [log](https://drive.google.com/file/d/1RWC0128oWtJt825JQBEy5fMnuHrJT-bi/view?usp=sharing)|