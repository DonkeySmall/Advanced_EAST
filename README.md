# Advanced EAST

This repository is edited from https://github.com/huoyijie/AdvancedEAST. Released under MIT license



## Dependence:

- python 3.6.3+
- tensorflow-gpu 1.5.0+(or tensorflow 1.5.0+)
- keras 2.1.4+
- numpy 1.14.1+
- tqdm 4.19.7+



## Train

Simply edit ``origin_image_dir_name ``, ``origin_txt_dir_name `` and ``data_dir `` in ``config.py``. Then, run ``python train.py``

## Test

~~~
python test.py -p demo/img1.jpg
~~~



### File Structure For Training Data

```
├── train_1000                   
│   ├── image_1000  
│            ├── IMG_0.jpg          # IMGs         
│					       ...
│
│   ├── txt_1000   
│            ├── IMG_0.txt          # IMG Labels         
│                   ....			
│
└── ...
```



## About HyperParameters

**cfg.shrink_ratio** and **cfg.shrink_side_ratio** are used to shrink text box.

In  [paper](https://arxiv.org/abs/1704.03155v2), **cfg.shrink_ratio = 0.3**. However, it seems too large for ducumentation text detection.



Here is a graph to help you understand how to shrink the bounding boxes

~~~
x ————————————————————————
|       ↓
|      0.2
|       ↑
|       x           x
|→ 0.2 ←|           |
|     →  0.6   ←    |
|        
~~~



为了适应长文本的识别，[huoyijie](https://github.com/huoyijie) 对loss有着自己的设计思路与哲学。仅个人理解，如有不同见解，欢迎讨论。



### For Training

We firstly regard all text regions as text grids with labels, which means, for a text region whose size is **[height, width],** we regard it as a **[height // pixel_size, width // pixel_size, 7 ]** vector. For the 7-d label vectors at each position, each element has its special meaning as listed below. 



| 0                 | 1                        | 2       | 3                   | 4                   | 5                   | 6                   |
| ----------------- | ------------------------ | ------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Exist Text: 1     | Is side_vertex_pixel: 1  | Tail: 0 | Upper Left X offset | Upper Left Y offset | Lower Left X offset | Lower Left Y offset |
| Non-Exist Text: 0 | Not side_vertex_pixel: 0 | Head: 1 | If y[:,:,1]  == 1   | If y[:,:,1]  == 1   | If y[:,:,1]  == 1   | If y[:,:,1]  == 1   |

1st digit is score map, represents if there exists text ；2nd digit is vertex code，3rd is head/tail flag；4 ~ 6 digit is geo. If 2nd digit == 1, we can use these four digits to predict box region。

> 其中所有像素构成了文本框形状，然后只用边界像素去预测回归顶点坐标。边界像素定义为黄色和绿色框内部所有像素，是用所有的边界像素预测值的**加权平均**来预测头或尾的短边两端的两个顶点。头和尾部分边界像素分别预测2个顶点，最后得到4个顶点坐标。

### For Testing

EAST predict: **y = East\_detect.predict()**

y represents the prediction of text grids from input image. 

We can filter out the regions which contain no potential text regions with **pixel_threshold** on y[:,:,0]. 



### Hyper-Parameter Meaning

| cfg,py                          | meaning                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| **pixel_threshold**             | Judge whether the region is valid text grid                                                                            y[i,j,0] > pixel\_threshold:   (i, j) is a valid text grid |
| **pixel_size**                  | super-pixel size,  used to transform original input image into text grids when training |
| **side_vertex_pixel_threshold** | Judge whether the region is side vertex pixels<br />IF y[i,j,0] > side\_vertex\_pixel\_threshold:   <br />(i, j) is a potential text region<br />else <br />(i, j) is a side_vertex region |
| **trunc_threshold**             | IF trunc\_threshold <= y[i,j,2] < 1- trunc\_threshold , <br />THEN  (i, j) is Not side_vertex_pixel<br /><br />IF y[i,j,2] < trunc\_threshold   , THEN  (i, j) is side_vertex_pixel and classified as head<br /><br />IF y[:,:,2] >= 1-  trunc_threshold   , THEN (i, j) is side_vertex_pixel and classified as tail |
| **segment_region_threshold**    | Used to segement connected regions if two text grids are overlapped |





### TO-DO List

- ~~Try to edit nms.py to increase text detection and merge logics~~
- ~~Train a model based on fake jz data~~