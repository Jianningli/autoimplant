## A Baseline Approach for the MICCAI 2020 Cranial Implant Design Challenge

coarse-to-fine implant prediction
![alt text](https://github.com/Jianningli/autoimplant/blob/master/images/teaser1.png)

The github repository contains codes for the automatic cranial implant design methods described in:

>Jianning Li, Antonio Pepe, Christina Gsaxner, Gord von Campe and Jan Egger. A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge. [arxiv:2006.12449v1(2020)](https://arxiv.org/abs/2006.12449).

### Direct Implant Generation & Volumetric Shape Completion
n1 model is trained for both direction implant prediction, as an intermediate step for n2 (left) and skull shape completion (right), in a down-sampled mode.
The pros and cons of both formulations are described as follows:
 >direct implant prediction
* cannot generalize well to varied defects (e.g., defects shape, position,size).
* can produce clean/high-quality implants directly.
>skull shape completion
* can generalize well to varied defects (defects shape, position), even if trained only on defects of a fixed pattern.
* the subtraction of the defective skull from the completed skull USUALLY won't yield the desired implant without further post- processing.

![alt text](https://github.com/Jianningli/autoimplant/blob/master/images/illustration.png)

### Quantitative and Qualitative Shape Analysis
The predicted implant shape (or the completed skull) can be evaluated quantitatively using **Dice Similarity Score (DSC)** or **Hausdorff Distance (HD)**, or qualitatively by assessing how it matches with the ground truth. However, better and more specialized quantitative metrics (as well as the **loss function** for the shape learning network) can be devised for more accurate evaluation of how two shapes match each other. Another aspect of qualitative evaluation is to visually inspect if the implant is consistent with the defective skull in terms of the bone thichness, shape as well as boundary of the defected region. 

![alt text](https://github.com/Jianningli/autoimplant/blob/master/images/match.png)

### Data
The training and testing set can be found at the AutoImplant challenge website (https://autoimplant.grand-challenge.org/).
The challenge provides 100 data pairs for training and 100 for testing. An additional 10 test data are provided for the evaluation of the algorithms robustness.    


### Codes
The codes run through <ins> Python '3.6.8' with tensorflow '1.4.0' on win10 with GTX Nvidia 1070 GPU </ins>. Other python and tensorflow versions have not been tested.

```
in the main.py: (if no GPU available, set os.environ['CUDA_VISIBLE_DEVICES'] = '-1)
load n1 model: from n1_model import auto_encoder  
load n2 model: from n2_model import auto_encoder
load skull shape completion model: from skull_completion_model import auto_encoder
to train model: model.train()
to test model: model.test()
to run the model (in training or testing mode): python main.py
__________________________________________________________________________________________
to convert the output of n2 to the orignal dimension:  python pred_2_org.py
__________________________________________________________________________________________
to remove the isolated noise automatically: pre_post_processing.py
```
more skull data processing codes can be found [HERE](https://github.com/Jianningli/autoimplant/tree/master/skull-processing).

### License
The codes are licensed under the MIT license. See [LICENSE](https://github.com/Jianningli/autoimplant/blob/master/LICENSE) for details.
If you find our codes useful or use our codes/methods in your research, please cite our paper:
```
@article{li2020baseline,
  title={A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge},
  author = {Jianning Li and Antonio Pepe and Christina Gsaxner and Gord von Campe and Jan Egger},
  journal={arXiv preprint arXiv:2006.12449},
  year={2020}
}
```
### Contact
Jianning Li    
Feel free to drop me an email if you have any questions regarding the paper/code: <ins>jianning.li@icg.tugraz.at</ins>




