conda-env list
conda activate py39torch11_6

# test OfficeHome of alpha and beta
python -u train_all.py scale_1_beta_1_alpha_1 --dataset OfficeHome --deterministic --trial_seed 0 --data_dir E:\dataset\OfficeHomeDataset_10072016 --algorithm GCL --scale 1 --alpha 1 --beta 1 --models regnety --batch_size 18
# test PACS of alpha and beta
python -u train_all.py scale_x_beta_0_alpha_0 --dataset PACS --deterministic --trial_seed 0 --data_dir E:\dataset\PACS --algorithm GCL --scale 1 --alpha 0 --beta 0 --models regnety --batch_size 18
python -u train_all.py scale_1_beta_1_alpha_1 --dataset PACS --deterministic --trial_seed 0 --data_dir E:\dataset\PACS --algorithm GCL --scale 1 --alpha 1 --beta 1 --models regnety --batch_size 18

# test TerraIncognita of alpha and beta
python -u train_all.py scale_1_beta_0p1_alpha_0p1 --dataset TerraIncognita --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm GCL --scale 1 --alpha 0.1 --beta 0.1 --models regnety --batch_size 18

# test VLCS of alpha and beta
python -u train_all.py scale_1_beta_0p5_alpha_0p5 --dataset VLCS --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm GCL --scale 1 --alpha 0.5 --beta 0.5 --models regnety --batch_size 18

# test DomainNet of alpha and beta
python -u train_all.py scale_10_beta_0_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm GCL --scale 10 --alpha 1 --beta 0 --models regnety --batch_size 18
