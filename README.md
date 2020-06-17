## A Baseline Approach for the MICCAI 2020 Cranial Implant Design Challenge
![alt text](https://github.com/Jianningli/autoimplant/blob/master/images/teaser.png)

The github repository contains codes for the automatic cranial implant design methods described in:
> **Jianning Li, Antonio Pepe, Christina Gsaxner, Gord von Campe, Dieter Schmalstieg, and Jan Egger. A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge (2020).**

### Direct Implant Generation & Volumetric Shape Completion
n1 model is trained for both direction implant prediction, as an intermediate step for n2 (left) and skull shape completion (right), in a down-sampled mode.
The pros and cons of both formulations are described as follows:

>**direct implant prediction**
* insensitive to the shape/position of the defects.
* cannot generalize well to varied defects (defects shape, position).
* can produce clean/high-quality implants directly.

>**skull shape completion**
* can generalize well to varied defects (defects shape, position).
* the subtraction of the defective skull from the completed skull usually won't yield the desired implant without further post- processing.

![alt text](https://github.com/Jianningli/autoimplant/blob/master/images/illustration.png)

### Data
The training and testing set can be found at the AutoImplant challenge website (https://autoimplant.grand-challenge.org/).
The challenge provides 100 data pairs for training and 100 for testing. An additional 10 test data are provided for the evaluation of the algorithms robustness.    


### Codes
The codes run through <ins>Python '3.6.8' with tensorflow '1.4.0' on win10 with GTX Nvidia 1070 GPU</ins>. Other python and tensorflow versions have not been tested.

>in the **main.py:** (if no GPU available, set ```os.environ['CUDA_VISIBLE_DEVICES'] = '-1```)

* **load n1 model:**  ```from n1_model import auto_encoder```   
* **load n2 model:**  ```from n2_model import auto_encoder```
* **load skull shape completion model:**  ```from skull_completion_model import auto_encoder```
* **to train model:**  ```model.train()```
* **to test model:**   ```model.test()```
* **to run the model (in training or testing mode):** ```python main.py```
> **to convert the output of n2 to the orignal dimension:** 
``` python pred_2_org.py```

### License
The codes are licensed under the MIT license. See [LICENSE](https://github.com/Jianningli/autoimplant/blob/master/LICENSE) for details.
If you use our code/model in your research, please cite our paper:
```
Jianning Li, Antonio Pepe, Christina Gsaxner, Gord von Campe, Dieter Schmalstieg, and Jan Egger.
A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge (2020)
```
### Contact
Questions regarding the paper can be communicated to Jianning Li at jianning.li@icg.tugraz.at




