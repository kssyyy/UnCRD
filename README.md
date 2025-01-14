# UnCRD
## Dataset
* The SEED dataset is publicly available on [here](https://bcmi.sjtu.edu.cn/home/seed/seed.html#).
* The SEED-IV dataset is publicly available on [here](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html).
* The SEED-V dataset is publicly available on [here](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).

## Requirements
* python==3.10.0
* torch==2.1.2
  
## Installation
```python
pip install -r requirements.txt
```
## Train train model
```python
python train.py --model "MLP" --save_name './ModelSave/EYE_within/' --model_name "MLP" --session "4"
```
## Train student model
```python
python distill.py --model "MLP" --loss 'unis_crd' --model_name './ModelSave/EYE_within/' --model_name "MLP" --session "4"
```
