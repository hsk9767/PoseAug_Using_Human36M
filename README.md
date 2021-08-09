Do 'PoseAug'(CVPR 2021) with the Human3.6M dataset loaded by [Moon](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git).


## Run training code  
* Only baseline network(w/o [SemGCN](https://github.com/garyzhao/SemGCN)) training is available now. 
* 2D inputs are in image coordinate system and the target 3D keypoints are in camera coordinate system with meter unit.
```sh
# videopose
python3 run_baseline_custom.py --note pretrain --lr 1e-3 --posenet_name 'videopose' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
# mlp
python3 run_baseline_custom.py --note pretrain --lr 1e-3 --stages 2 --posenet_name 'mlp' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
# st-gcn
python3 run_baseline_custom.py --note pretrain --dropout -1 --lr 1e-3 --posenet_name 'stgcn' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
``` 

## Acknowledgements
This code uses ([SemGCN](https://github.com/garyzhao/SemGCN), [SimpleBL](https://github.com/una-dinosauria/3d-pose-baseline), [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks) and [VPose3D](https://github.com/facebookresearch/VideoPose3D)) as backbone. The integrated contents are from [PoseAug](https://github.com/jfzhang95/PoseAug.git). Human 3.6M dataset is from [Moon's github](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git).