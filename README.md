# MMDetection

[TOC]



## 介绍

**主分支使用PyTorch 1.3 to 1.5.**

老分支v1是pytorch1.1-1.4， 但是v2是更快更高更强

支持faster rcnn, mask rcnn, retinanet



## 安装

### 需要

+ Linux or macOS (Windows is not currently officially supported)
+ Python 3.6+
+ PyTorch 1.3+
+ CUDA 9.2+ 
+ GCC 5+
+ [mmcv](https://github.com/open-mmlab/mmcv)

### 安装mmdetection

1. 创建conda虚拟环境并激活

   ```shell
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2.  安装 PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

3. 确保你编译的 CUDA 版本 and 运行 CUDA 版本匹配. 

   例如1：

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

   例如2：

   ```shell
   conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
   ```

4. 拷贝安装环境

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   ```

5. 安装

   ```shell
   pip install -r requirements/build.txt
   pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
   pip install -v -e .  # or "python setup.py develop"
   ```

6. 查看mmdetection安装版本

   ```shell
   PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
   ```

   

## 开始训练

mmdetection的数据格式

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

<font color=red>cityscapes annotations</font> 转 coco 形式 用 `tools/convert_datasets/cityscapes.py`:

```shell
pip install cityscapesscripts
python tools/convert_datasets/cityscapes.py ./data/cityscapes --nproc 8 --out-dir ./data/cityscapes/annotations
```

当前配置文件cityscapes使用COCO预训练权重来初始化。

引用当前新数据

[Tutorials 2: Adding New Dataset](https://github.com/Cauchy-Fan/mm_wheel/blob/master/docs/tutorials/new_dataset.md).

### 引用预训练模型

提供了评测整个数据集(COCO, PASCAL VOC, Cityscapes, etc.) 然后也some high-level apis

### 测试数据集

+ single GPU
+ single node multiple GPU
+ multiple node

使用一下命令测试一个数据集

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

选择参数:

+ `RESULT_FILE`: 输出结果pickle格式，结果将存在文件里。
+ `EVAL_METRICS`: 选择评测e.g., COOC评测使用`proposal_fast`, `proposal`, `bbox`, `segm` ，PASCAL VOC. 使用mAP`,` ``recall` for Cityscapes 
+ `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.
+ `--show-dir`: If specified, detection results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
+ `--show-score-thr`: If specified, detections with score below this threshold will be removed.

举例：

认为你已经将模型下载到了checkpoints文件中了

1. 测试Faster R-CNN和可视化结果，按任意键可以下一张图

   ```shell
   python tools/test.py configs/faster_rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
       --show
   ```

2. 测试Faster R-CNN和保存照片

   ```shell
   python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
       checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
       --show-dir faster_rcnn_r50_fpn_1x_results
   ```

3. 测试Faster R-CNN在PASCAL VOC数据集不保存测试结果并且评测mAP

   ```shell
   python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
       checkpoints/SOME_CHECKPOINT.pth \
       --eval mAP
   ```

4. 测试Mask R-CNN使用8GPU，评测bbox和mask AP

   ```shell
   ./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
       8 --out results.pkl --eval bbox segm
   ```

5. 测试Mask R-CNN使用8GPUs, 评测类别bbox和mask AP.

   ```shell
   ./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
       8 --out results.pkl --eval bbox segm --options "classwise=True"
   ```

### Image demo

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```

#### 例子

```shell
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --device cpu
```



### Webcam demo

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--camera-id ${CAMERA-ID}] [--score-thr ${SCORE_THR}]
```

#### 例子

```shell
python demo/webcam_demo.py configs/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
```



### 高级别API 测试照片

#### Synchronous interface

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```



## 训练模型

#### 使用单GPU训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果你想去特别的工作空间，你能添加参数`--work_dir ${YOUR_WORK_DIR}`.

#### 使用多GPUs训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

+ `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py#L174)) epochs during the training. To disable this behavior, use `--no-validate`.
+ `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
+ `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.



## 有用的工具

tools文件夹中含有很多有用的工具

### 分析logs

首先安装seaborn

![loss curve image](https://github.com/Cauchy-Fan/mm_wheel/raw/master/demo/loss_curve.png)

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```









