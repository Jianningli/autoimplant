## A Baseline Approach for the MICCAI 2020 Cranial Implant Design Challenge

The gihub repository contains codes for the automatic cranial implant design methods described in 'A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge'.
### Data
The training and testing set can be found at the AutoImplant challenge website (https://autoimplant.grand-challenge.org/). 
### Codes
The codes run through Python '3.6.8' with tensorflow '1.4.0' on win10 with GTX Nvidia 1070 GPU. Other python and tensorflow versions have not been tested.

in the **main.py:**

* **load n1 model:** from n1_model import auto_encoder   
* **load n2 model:** from n2_model import auto_encoder
* **load skull shape completion model:** from skull_completion_model import auto_encoder
* **to train model:**  model.train()
* **to test model:**   model.test()
* **to run the model (in training or testing mode):** python main.py
if no GPU available, set os.environ['CUDA_VISIBLE_DEVICES'] = '-1



### License
The codes are licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://github.com/Jianningli/autoimplant/blob/master/LICENSE)

### Contact
Questions regarding the paper can be communicated to Jianning Li at jianning.li(AT)icg.tugraz.at
### Citation (bibtex)



