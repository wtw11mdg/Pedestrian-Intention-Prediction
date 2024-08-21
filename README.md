# Pedestrian-Intention-Prediction

**Paper :  [T. W. Wang and S. -H. Lai, "Pedestrian Crossing Intention Prediction with Multi-Modal Transformer-Based Model," 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)](https://ieeexplore.ieee.org/abstract/document/10317161/authors#authors)**

/n

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
