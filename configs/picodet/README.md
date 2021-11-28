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

#### Backbone Pretrained Weights
- [ESNet_0.75](https://drive.google.com/file/d/1j0Bw8TyTnbwfmGihUdRZ0var4zFLe6W5/view?usp=sharing)
- [ESNet_1.0x](https://drive.google.com/file/d/1oGJTjX0xNzmqgkZWzJsGXRD7_WOZrtkO/view?usp=sharing)

#### Progress
ing. 

| Model  | exp setting| box AP | log | weights
|:---------:|:-------:|:-------:|:-------:|:-------:|
|Picodet-s-416(reproduce-1120)| most aligned <br>lr decay not consistency <br>ema initialized with model param|30.1| [log](https://drive.google.com/file/d/1KfSAYQHxGNz0btn_BoGWq9nPK4t43T_U/view?usp=sharing)|[weights](https://drive.google.com/file/d/181GANlB8vnvQ2ZAL05ufo8quG0a7aZD8/view?usp=sharing) | 
|Picodet-s-416(reproduce-1124)|lr decay not consistency <br>ema initialized with zero|30.3| [log](https://drive.google.com/file/d/1TpOtKmgoZgiG_s5dR92zc1El6ObbYrTh/view?usp=sharing)|[weights](https://drive.google.com/file/d/14wckQPZtRMfXoXR2iwv-67aRkz8bLSvR/view?usp=sharing) | 
|Picodet-m-416(reproduce-1128)|lr decay not consistency |34.2| [log](https://drive.google.com/file/d/1BWBcHj7SPytCValyjUTICNZ1paqXhRgC/view?usp=sharing)|[weights](https://drive.google.com/file/d/1NHoqetZGdZ0PwxWqMs7Jxgp-YTYInzpE/view?usp=sharing) | 