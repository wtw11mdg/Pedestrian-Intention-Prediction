# Pedestrian-Intention-Prediction
The project is based on the [PIE](https://github.com/aras62/PIE) and [JAAD](https://github.com/aras62/PIE) datasets and is validated against the standard of [Benchmark for Evaluating Pedestrian Action Prediction](#citation).

If you are interested in the details of this project, please refer to the original paper: <br>
[T. W. Wang and S. -H. Lai, "Pedestrian Crossing Intention Prediction with Multi-Modal Transformer-Based Model," 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)](https://ieeexplore.ieee.org/abstract/document/10317161/authors#authors)
<br>
<br>
![image](https://github.com/user-attachments/assets/b473c363-59ea-4591-88b8-14744d5b0ddb)
### Abstract
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
*** The code has not been refined and fully organized. For reference only. ***

<a name="citation"></a>
## Citation

According to the benchmark, this project is based on the code provided in the following benchmark.

**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)**

<!--
```
@inproceedings{kotseruba2021benchmark,
	title={{Benchmark for Evaluating Pedestrian Action Prediction}},
	author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
	booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
	pages={1258--1268},
	year={2021}
}
```
-->

For testing and reference, the SFRNN[1] model is referenced in the code.

[1] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs. BMVC, 2019





