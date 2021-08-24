Do train the lifting network [Simple](https://github.com/una-dinosauria/3d-pose-baseline) with the Human3.6M dataset loaded by [Moon](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git).


## Tree
Follow the instructions in [link](https://github.com/mks0601/3DMPPE_POSENET_RELEASE/tree/3f92ebaef214a0eb1574b7265e836456fbf3508a#data) to make the 'Human3.6M' & 'MPII' folder and put it under the 'data' directory. 
(MPII for only one-stage method training)

```sh
$ PoseAug_Human36M
| -- ...
| -- data
    | -- Human3.6M
        | -- annotations
        | -- images
        | -- bbox_root
        | -- ...
    | -- MPII
        | -- annotations
        | -- images
| -- ...
```
## Run training code  
* Only baseline network([Simple](https://github.com/una-dinosauria/3d-pose-baseline)) training is available now. 
* Below instructions do the training of 2-stage methods.

```sh
# mlp - GT 2D keypoints
python3 run_baseline_custom.py --note GT --checkpoint ./checkpoint/pretrain_baseline  --keypoints gt

# mlp - 2D keypoints estimated by networks
# before train using the 2D keypoints from the 2D estimation network, do 'run_2d_save.py' first.
# the results will be stored in {args.checkpoint} / mlp / {args.keypoints} / {start_time}_{args.note} / 
python3 run_baseline_custom.py --note resnet_50 --checkpoint ./checkpoint/pretrain_baseline/ --keypoints resnet_50
python3 run_baseline_custom.py --note resnet_101 --checkpoint ./checkpoint/pretrain_baseline/ --keypoints resnet_101
python3 run_baseline_custom.py --note resnet_152 --checkpoint ./checkpoint/pretrain_baseline/ --keypoints resnet_152
python3 run_baseline_custom.py --note resnet --checkpoint ./checkpoint/pretrain_baseline/ --keypoints pelee

``` 
* Below instruction does the training of 1-stage methods.
```sh
# thre result will be saved in {args.save_path_one_stage}/
python run_train_one_stage.py --batch_size 32 --save_path_one_stage {PATH/TO/SAVE/WEIGHT FILE}
```

## Run evaluation code
```sh
# evaluate 2-stage methods
python run_evaluate_custom.py --keypoints gt --evaluate {PATH/TO/WEIGHT}
python run_evaluate_custom.py --keypoints pelee --evaluate {PATH/TO/WEIGHT}
python run_evaluate_custom.py --posenet_name mlp --keypoints resnet_50 --evaluate {PATH/TO/WEIGHT}
python run_evaluate_custom.py --posenet_name mlp --keypoints resnet_101 --evaluate {PATH/TO/WEIGHT}
python run_evaluate_custom.py --posenet_name mlp --keypoints resnet_152 --evaluate {PATH/TO/WEIGHT}
```
```sh
# evaluate the 2-stage method
python run_evaluate_one_stage.py --args.path_one_stage {PATH/TO/WEIGHT}
```

## 2D detection result save 
Get the pre-trained weight from [link](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC) for the resnet-based 2d pose estimation model and [link](https://drive.google.com/drive/folders/1b16iQ9p5jLcICnhGibkhyuFewTYyNLwF?usp=sharing) for peleenet-based model.
```sh
# save the result of the Human3.6M training set using a 2D human pose estimation network
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_50 --path_2d {PATH/TO/WEIGHT} --is_train true 
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_101 --path_2d {PATH/TO/WEIGHT} --is_train true 
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_152 --path_2d {PATH/TO/WEIGHT} --is_train true 
python run_2d_detection_save.py --batch_size 128 --keypoints pelee --path_2d {PATH/TO/WEIGHT} --is_train true 

# save the result of the Human3.6M test set
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_50 --path_2d {PATH/TO/WEIGHT} --is_train false 
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_101 --path_2d {PATH/TO/WEIGHT} --is_train false 
python run_2d_detection_save.py --batch_size 128 --keypoints resnet_152 --path_2d {PATH/TO/WEIGHT} --is_train false
python run_2d_detection_save.py --batch_size 128 --keypoints pelee --path_2d {PATH/TO/WEIGHT} --is_train false 
```

## Saved 2D detection result test
```sh
# Visualize the keypoints detected by 2D estimator
# With GT keypoints -> save_test/{#}_GT.jpg
# With estimated keypoints -> save_test/{#}_pelee.jpg or save_test/{#}_resnet.jpg

python run_2d_save_test.pt --keypoints pelee
python run_2d_save_test.pt --keypoints resnet_50
python run_2d_save_test.pt --keypoints resnet_101
python run_2d_save_test.pt --keypoints resnet_152
```
<!-- 
## 2D estimation network finetune
```sh
# finetune the 2D network to Human3.6M
python run_finetune.py --keypoints resnet_50 --path_2d {PATH/TO/WEIGHT} --batch_size 128
``` -->
## Acknowledgements
This code uses ([SemGCN](https://github.com/garyzhao/SemGCN), [SimpleBL](https://github.com/una-dinosauria/3d-pose-baseline), [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks) and [VPose3D](https://github.com/facebookresearch/VideoPose3D)) as backbone. The integrated contents are from [PoseAug](https://github.com/jfzhang95/PoseAug.git). Human 3.6M dataset is from [Moon's github](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git).
