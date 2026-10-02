<div align="center">
  <h1>A Baseline Approach for the MICCAI 2020 Cranial Implant Design Challenge</h1>
  
  <p>
    <a href="http://jianningli.me/C2F"><img src="https://img.shields.io/badge/Project_Page-0078D4?style=flat-square&logo=google-chrome&logoColor=white" alt="Project Page" /></a>
    <a href="https://arxiv.org/abs/2006.12449"><img src="https://img.shields.io/badge/Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="Paper" /></a>
    <a href="https://autoimplant.grand-challenge.org/"><img src="https://img.shields.io/badge/Challenge_Page-FF8C00?style=flat-square&logo=challenge&logoColor=white" alt="Challenge" /></a>
  </p>
</div>

---

## 🚀 Coarse-to-Fine Implant Prediction

<div align="center">
  <img src="https://github.com/Jianningli/autoimplant/blob/master/images/overview.png" alt="overview" width="100%">
</div>

This GitHub repository contains codes for the automatic cranial implant design methods described in:
> Jianning Li, Antonio Pepe, Christina Gsaxner, Gord von Campe and Jan Egger. **A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge.** [arxiv:2006.12449 (2020)](https://arxiv.org/abs/2006.12449).

---

## 🧠 Direct Implant Generation & Volumetric Shape Completion

The `n1` model is trained for both direct implant prediction and skull shape completion (in a down-sampled mode). The pros and cons of both formulations are described below:

### Direct Implant Prediction
* ❌ Cannot generalize well to varied defects (e.g., defect shape, position, size).
* ✅ Can produce clean/high-quality implants directly.

<img src="https://github.com/Jianningli/autoimplant/blob/master/images/directimplantgeneration.png" alt="direct implant generation" width="80%">

### Skull Shape Completion
* ✅ Can generalize well to varied defects (defect shape, position), even if trained only on defects of a fixed pattern.
* ❌ The subtraction of the defective skull from the completed skull USUALLY won't yield the desired implant without further post-processing.

<img src="https://github.com/Jianningli/autoimplant/blob/master/images/skullshapecompletion.png" alt="skull shape completion" width="80%">

---

## 📊 Quantitative and Qualitative Shape Analysis

The predicted implant shape (or the completed skull) can be evaluated quantitatively using **Dice Similarity Score (DSC)** or **Hausdorff Distance (HD)**, or qualitatively by assessing how it matches with the ground truth. 

However, better and more specialized quantitative metrics (as well as the **loss function** for the shape learning network) can be devised for a more accurate evaluation of how two shapes match each other. Another aspect of qualitative evaluation is to visually inspect if the implant is consistent with the defective skull in terms of bone thickness, shape, as well as the boundary of the defected region. 

<div align="center">
  <img src="https://github.com/Jianningli/autoimplant/blob/master/images/match.png" alt="match" width="70%">
</div>

---

## 📁 Data

The training and testing set can be found at the [AutoImplant challenge website](https://autoimplant.grand-challenge.org/).
The challenge provides **100 data pairs for training** and **100 for testing**. An additional **10 test data** are provided for the evaluation of the algorithms' robustness.    

---

## 💻 Codes

> **Requirements:** Python `3.6.8` with TensorFlow `1.4.0` on Win10 with a GTX Nvidia 1070 GPU.

In `main.py` *(if no GPU available, set `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`)*:
```python
# Load n1 model
from n1_model import auto_encoder  

# Load n2 model
from n2_model import auto_encoder

# Load skull shape completion model
from skull_completion_model import auto_encoder

# Train model
model.train()

# Test model
model.test()
```

To run the model (in training or testing mode):
```bash
python main.py
```
---
To convert the output of `n2` to the original dimensions:
```bash
python pred_2_org.py
```
To remove the isolated noise automatically:
```bash
python pre_post_processing.py
```

> **Note:** More skull data processing codes can be found [HERE](https://github.com/Jianningli/autoimplant/tree/master/skull-processing).

---

## ⚖️ License
The codes are licensed under the MIT license. See [LICENSE](https://github.com/Jianningli/autoimplant/blob/master/LICENSE) for details.

If you find our codes useful or use our codes/methods in your research, please cite our paper:
```bibtex
@article{li2020baseline,
  title={A Baseline Approach for AutoImplant: the MICCAI 2020 Cranial Implant Design Challenge},
  author = {Jianning Li and Antonio Pepe and Christina Gsaxner and Gord von Campe and Jan Egger},
  journal={arXiv preprint arXiv:2006.12449},
  year={2020},
  month={06}
}
```

## 📬 Contact
**Jianning Li**  
Feel free to drop me an email if you have any questions regarding the paper/code: [jianning.li@icg.tugraz.at](mailto:jianning.li@icg.tugraz.at)
