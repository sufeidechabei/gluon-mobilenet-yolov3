# Gluon-Mobilenet-YOLOv3

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam <br>

**Abstract** <br>
We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

[[Paper]](https://arxiv.org/abs/1704.04861) [[Original Implementation]](https://github.com/Zehaos/MobileNet)

## Prerequisites
1. Python 3.6 +
2. Gluoncv 0.3.0
3. Mxnet

## Usage

### Mobilenet
#### voc
```
python3 train_yolo3_mobilenet.py --network mobilenet1_0 --dataset voc --gpus 0,1,2,3,4,5,6,7 --batch-size 64 -j 16 --log-interval 100 --lr-decay-epoch 160,180 --epochs 200 --syncbn --warmup-epochs 4
```
### coco
```
python3 train_yolo3_mobilenet.py --network mobilenet1_0 --dataset coco --gpus 0,1,2,3,4,5,6,7 --batch-size 64 -j 32 --log-interval 100 --lr-decay-epoch 220,250 --epochs 280 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 20 --label-smooth --no-wd

```



## MAP

| Backbone                | GPU     | Dataset    |  Size  |   MAP      | 
| ----------------------- |:--------:|:--------: | :--------: | :--------:|
| Mobilenet               | 8 Tesla v100  | VOC   | random shape   |  76.12       |
| Mobilenet               | 8 Tesla v100 | COCO2017 | random shape | 28.3          |



## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
@article{mobilenets,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Andrew G. Howard, Menglong Zhu, Bo Chen,Dmitry Kalenichenko,Weijun Wang, Tobias Weyand,Marco Andreetto, Hartwig Adam},
  journal = {arXiv},
  year = {2017}
}
```
