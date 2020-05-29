# EfficientDet-keras-lightversion


Thanks to [xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet)

This repo is a light version of the original one, remove some script.

* change to tf2.1.0
* merge into one file for kaggle kernel
* relace progressbar2 with tqdm
* add earlystop,save best model,reduce lr
* add cocoapi eval

### Test
```
1. d2 bifpn=False pre_score_threshold=0.5 nms=0.5

loss: 0.2411 - classification_loss: 0.1261 - regression_loss: 0.1151 - val_loss: 0.3281 - val_classification_loss: 0.1818 - val_regression_loss: 0.1463

	localtrain: valmap9132   coco6810   

	onlinetrain(dataleak):  
	* pre_score_threshold=0.5 coco7430  board6298
	* pre_score_threshold=0.4 coco7590  board6291

	* py_nms(base): pre_score_threshold=0.5 nms=0.5 coco0.681
    * Weighted-Boxes-Fusion nms: pre_score_threshold=0.5 nms=0.5 coco0.681
	* Weighted-Boxes-Fusion wbf: pre_score_threshold=0.5 nms=0.5 coco0.681  board:0.6215
							     pre_score_threshold=0.4 nms=0.5 coco0.693
                                 pre_score_threshold=0.6 nms=0.5 coco0.664
                                 pre_score_threshold=0.3 nms=0.5 coco0.699  board:0.5877
                                 pre_score_threshold=0.2 nms=0.5 coco0.708
                                 pre_score_threshold=0.1 nms=0.5 coco0.714  board:0.3883
                                 pre_score_threshold=0.05 nms=0.5 coco0.714  
                                 pre_score_threshold=0.0 nms=0.5 coco0.714
                                 pre_score_threshold=0.1 nms=0.4 coco0.672
                                 pre_score_threshold=0.1 nms=0.6 coco0.714


2. d5 512   
	valmap0.85   pre_score_threshold=0.5 coco0.526
				 pre_score_threshold=0.4 coco0.554
                 pre_score_threshold=0.3 coco0.576
	             pre_score_threshold=0.2 coco0.569

3.d5 1280
	valmap0.89   pre_score_threshold=0.5 coco0.154
				 pre_score_threshold=0.7 coco0.135
	             pre_score_threshold=0.2 coco0.013

4.d4 1024 valmap0.9219
	* score_threshold=0.5 nms coco0.693     wbf0.693
    * score_threshold=0.4 nms coco0.703     wbf0.703
	* score_threshold=0.3 nms coco0.684     wbf0.716

5.d3 1024 valmap9251
	* score_threshold=0.6 nms coco0.672
	* score_threshold=0.5 nms coco0.694
    * score_threshold=0.4 nms coco0.693
	* score_threshold=0.3 nms coco0.679

6.d2 1024

```

### Run
1. python setup_gen_overlap.py
2. python setup.py build_ext --inplace
3. run python train.py


### Base Result
1. efficientd0 valmap0.8695 board0.5366 coco0.65
2. efficientd1 valmap0.9037 batchsize16
3. efficientd2 valmap0.9148 batchsize16
3. efficientd3 valmap0.9212 batchsize4
4. efficientd4 valmap0.9219 batchsize2