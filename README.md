## A Baseline Approach for the MICCAI 2020 Cranial Implant Design Challenge

The github repository contains codes for the automatic cranial implant design methods described in '**A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge**'
### Data
The training and testing set can be found at the AutoImplant challenge website (https://autoimplant.grand-challenge.org/). 
### Codes
The codes run through Python '3.6.8' with tensorflow '1.4.0' on win10 with GTX Nvidia 1070 GPU. Other python and tensorflow versions have not been tested.

in the **main.py:** (if no GPU available, set os.environ['CUDA_VISIBLE_DEVICES'] = '-1)

* **load n1 model:**  ```from n1_model import auto_encoder```   
* **load n2 model:**  ```from n2_model import auto_encoder```
* **load skull shape completion model:**  ```from skull_completion_model import auto_encoder```
* **to train model:**  ```model.train()```
* **to test model:**   ```model.test()```
* **to run the model (in training or testing mode):** ```python main.py```
* **to convert the output of n2 to the orignal dimension:**  ``` python pred_2_org.py```

### License
The codes are licensed under a CC BY-NC. See [!LICENSE](https://github.com/Jianningli/autoimplant/blob/master/LICENSE) for details.

### Contact
Questions regarding the paper can be communicated to Jianning Li at jianning.li@icg.tugraz.at
### Citation (bibtex)



