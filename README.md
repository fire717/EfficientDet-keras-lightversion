# EfficientDet-keras-lightversion


Thanks to [xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet)

This repo is a light version of the original one, remove some script.

* change to tf2.1.0
* merge into one file for kaggle kernel
* relace progressbar2 with tqdm
* add earlystop,save best model,reduce lr


### Run
1. python setup_gen_overlap.py
2. python setup.py build_ext --inplace
3. run python train.py


### Result
1. efficientd0 valmap0.8695 board0.5366
2. efficientd1 valmap0.9037
3. efficientd2 valmap0.9148