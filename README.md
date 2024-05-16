# Selecting Effective Triplet Contrastive Loss for Domain Alignment (ICIC'24)


## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```ps1
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
```

This project is based on the code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.
