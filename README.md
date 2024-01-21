# SC_Depth
Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video
## 简介
SC_Depth的非官方实现，主要参考论文[《Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video》](https://arxiv.org/pdf/2105.11610v1.pdf),
官方github为[sc_depth_pl](https://github.com/JiawangBian/sc_depth_pl) ，这也是该repo主要参考的代码，原仓库基于kornia完成基本空间几何的操作(法向量，相机投影，RT变换等)，本仓库为简易起见只依赖pytorch和pytorch-lightning.
另外本仓库只实现kitti数据集上的单目深度估计，并且提供了删减后kitti数据集和必要的gt_depth文件，简化算法跟进成本，方便后续改进和优化。

## 数据集
KITTI（只是针对eigen_zhou和eigen_full两个split进行了抽取）大小大概为7G(原monodepthv2提供的数据大概170G)

百度云盘
```text
简化后的KITTI数据:
链接: https://pan.baidu.com/s/1--ssV6krvvwHK-LT4rE8SQ 提取码: 7s7b 
--来自百度网盘超级会员v6的分享

gt_depth文件:(目前只提供eigen的test_files.txt的gt_depth)
链接: https://pan.baidu.com/s/1ex185AZq_vkhFmbFw72k-Q 提取码: h8ap 
--来自百度网盘超级会员v6的分享

```
如果需要kitti其他split的gt_path请参考monodepthv2的export_gt_depth.py repo地址:[github](https://github.com/nianticlabs/monodepth2) ,同样split的文件也可以在repo的splits下找到。

## API
- 训练
```shell script
python train.py --model_name M --kitti_dir 'your kitti img dir' --train_split '' --val_split '' --gt_path '' --ref_ids [-1,1] --width 832 --height 256
```

**如果配置项gt_path没有对应的gt_depth.npz文件，配置项val_mode请设置为photo，否则无法运行**


- 验证
```shll script
python eval_kitti.py --kitti_dir ‘’ --gt_path ‘’  --ckpt ‘’ --split_path ‘’ --cuda True
```

## 指标
论文中指标

| depth encode | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|resnet18 | 832x256 | 0.119 | 0.857 | 4.950 | 0.197 | 0.863 | 0.957 | 0.981 |
|resnet50 | 832x256 | 0.114 | 0.813 | 4.706 | 0.191 | 0.873 | 0.960 | 0.982 |

该repo的指标

| depth encode | 输出尺寸 | Abs Rel | Sq Rel | RMSE | RMSE log | δ<1.25 | δ<1.25^2 | δ<1.25^3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|resnet18 | 832x256 | 0.115 | 0.784 | 4.805 | 0.192 | 0.870 | 0.960 | 0.982 |
|resnet50 | 832x256 | 0.110 | 0.767 | 4.518 | 0.183 | 0.886 | 0.964 | 0.984 |

resnet18
```text
链接: https://pan.baidu.com/s/1Fr7lAG0k4XgalzucJj6k8A 提取码: tpx3 
--来自百度网盘超级会员v6的分享
```
resnet50
```text
链接: https://pan.baidu.com/s/1Ci1gt2TAT_rEFmlCMP8J4Q 提取码: 5bct 
--来自百度网盘超级会员v6的分享
```
**注: 与δ相关的项目越高越好**
resnet18训练使用3080，显存占用8G,训练时长14h,论文中(TESLA v100 29h).

## 参考
```text
@inproceedings{bian2019neurips,
  title={Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video},
  author={Bian, Jiawang and Li, Zhichao and Wang, Naiyan and Zhan, Huangying and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```