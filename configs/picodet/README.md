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

## Reproduce Step

### Inference Aligned

We convert the official weights to align inference.

| Model  | box AP | 
|:---------:|:-------:|
| [Picodet-s-416(ppdet)](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet)| 30.5(interp=1)| 
| [Picodet-s-416(mmdet)](https://drive.google.com/file/d/1XB8JOPz35fCIDyNcT5146UagX4etfQQf/view?usp=sharing) | 30.5| 

```
bash tools/dist_test.sh configs/picodet/picodet_s_416_coco.py $MODEL_PATH 8 --eval bbox 
```

### Train Aligned

ing. There are still some differences, 0.5 mAP lower

