# EfficientDet-keras-lightversion


Thanks to [xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet)

This repo is a light version of the original one, remove some script.

* change to tf2.1.0
* merge into one file for kaggle kernel
* relace progressbar2 with tqdm
* add earlystop,save best model,reduce lr
* add cocoapi eval

### Test
1. d2 bifpn=False pre_score_threshold=0.5 nms=0.5
loss: 0.2411 - classification_loss: 0.1261 - regression_loss: 0.1151 - val_loss: 0.3281 - val_classification_loss: 0.1818 - val_regression_loss: 0.1463

	localtrain: valmap9132   coco6810   

	onlinetrain(dataleak):  
	* pre_score_threshold=0.5 coco7430  board6298
	* pre_score_threshold=0.4 coco7590  board6291



2. 

### Run
1. python setup_gen_overlap.py
2. python setup.py build_ext --inplace
3. run python train.py


### Base Result
1. efficientd0 valmap0.8695 board0.5366 coco0.65
2. efficientd1 valmap0.9037 batchsize16
3. efficientd2 valmap0.9148 batchsize16
3. efficientd3 valmap0.9212 batchsize4
4. efficientd4 valmap0.9XXX batchsize2