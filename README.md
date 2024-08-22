# Pedestrian-Intention-Prediction
The project is based on the PIE and JAAD datasets and is validated against the benchmark standard.
**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)** (see [citation](#citation) information below).

**Paper :  [T. W. Wang and S. -H. Lai, "Pedestrian Crossing Intention Prediction with Multi-Modal Transformer-Based Model," 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)](https://ieeexplore.ieee.org/abstract/document/10317161/authors#authors)**
<br>

![image](https://github.com/user-attachments/assets/b473c363-59ea-4591-88b8-14744d5b0ddb)

The popularity of autonomous driving and advanced driver assistance systems can potentially reduce thousands of car accidents and casualties. In particular, pedestrian prediction and protection is an urgent development priority for such systems. Prediction of pedestrians’ intentions of crossing the road or their actions can help such systems to assess the risk of pedestrians in front of vehicles in advance. In this paper, we propose a multi-modal pedestrian crossing intention prediction framework based on the transformer model to provide a better solution. Our method takes advantage of the excellent sequential modeling capability of the Transformer, enabling the model to perform stably in this task. We also propose to represent traffic environment information in a novel way, allowing such information can be efficiently exploited. Moreover, we include the lifted 3D human pose and 3D head orientation information estimated from pedestrian image into the model prediction, allowing the model to understand pedestrian posture better. Finally, our experimental results show the proposed system provides state-of-the-art accuracy on benchmarking datasets.

### Train and test models

Use `train_test.py` script with `config_file`:
```
python train_test.py -c <config_file>
```

To train the model in this paper, run:  

```
python train_test.py -c config_files/PIE_TFGRU.yaml
```

<a name="citation"></a>
## Citation

If you use the results, analysis or code for the models presented in the paper, please cite:

```
@inproceedings{kotseruba2021benchmark,
	title={{Benchmark for Evaluating Pedestrian Action Prediction}},
	author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
	booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
	pages={1258--1268},
	year={2021}
}
```

If you use model implementations provided in the benchmark, please cite the corresponding papers

- ATGC [1] 
- C3D [2]
- ConvLSTM [3]
- HierarchicalRNN [4]
- I3D [5]
- MultiRNN [6]
- PCPA [7]
- SFRNN [8] 
- SingleRNN [9]
- StackedRNN [10]
- Two_Stream [11]

[1] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Are they going to cross?  A benchmark dataset and baseline for pedestrian crosswalk behavior.  ICCVW, 2017.

[2] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,and Manohar Paluri. Learning spatiotemporal features with 3D convolutional networks. ICCV, 2015.

[3] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung,Wai-Kin Wong, and Wang-chun Woo. Convolutional LSTM network:  A machine learning approach for precipitation nowcasting. NeurIPS, 2015.

[4] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. CVPR, 2015

[5] Joao Carreira and Andrew Zisserman.  Quo vadis, action recognition?  A new model and the kinetics dataset.  CVPR, 2017.

[6] Apratim Bhattacharyya, Mario Fritz, and Bernt Schiele. Long-term on-board prediction of people in traffic scenes under uncertainty. CVPR, 2018.

[7] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021.

[8] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs. BMVC, 2019

[9] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos.  Do They Want to Cross? Understanding Pedestrian Intention for Behavior Prediction. In IEEE Intelligent Vehicles Symposium (IV), 2020.

[10] Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vi-jayanarasimhan, Oriol Vinyals, Rajat Monga, and GeorgeToderici. Beyond short snippets: Deep networks for video classification. CVPR, 2015.

[11] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. NeurIPS, 2014.

