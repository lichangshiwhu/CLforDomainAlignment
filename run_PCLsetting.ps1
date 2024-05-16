#ablations on $\mathcal{L}_{ws}$

# officehome
python -u train_all.py pclset --dataset OfficeHome --deterministic --trial_seed 0 --data_dir E:\dataset\OfficeHomeDataset_10072016 --algorithm GCL --scale 1 --alpha 1 --beta 1  --steps 3000 --checkpoint_freq 300

# PACS
python -u train_all.py pclset --dataset PACS --deterministic --trial_seed 0 --data_dir E:\dataset\PACS --algorithm GCL --scale 1 --alpha 1 --beta 1 --checkpoint_freq 300 --steps 5000

# TerraIncognita
python -u train_all.py pclset --dataset TerraIncognita --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm GCL --scale 1 --alpha 1 --beta 1 --checkpoint_freq 1000 --steps 5000 
