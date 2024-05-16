conda-env list
conda activate py39torch11_6

# test DomainNet of alpha, scale and beta
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_1_seed1 --dataset DomainNet --deterministic --trial_seed 1 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
# ablations on beta
python -u train_all.py scale_10_beta_0p0001_alpha_1_seed2 --dataset DomainNet --deterministic --trial_seed 2 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
python -u train_all.py scale_10_beta_0p001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.001
python -u train_all.py scale_10_beta_0p01_alpha_1_seed2 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.01 
python -u train_all.py scale_10_beta_0p1_alpha_1_seed2 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.1 
python -u train_all.py scale_10_beta_1_alpha_1_seed2 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 1 
python -u train_all.py scale_10_beta_10_alpha_1_seed2 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 10 
# ablations on alpha
python -u train_all.py scale_10_beta_0p0001_alpha_0p0001 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 0.0001 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_0p001 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 0.001 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_0p01 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 0.01 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_0p1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 0.1 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_10 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 10 --beta 0.0001 
# ablations on scale
python -u train_all.py scale_0p0001_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 0.0001 --alpha 1 --beta 0.0001 
python -u train_all.py scale_0p001_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 0.001 --alpha 1 --beta 0.0001 
python -u train_all.py scale_0p01_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 0.01 --alpha 1 --beta 0.0001 
python -u train_all.py scale_0p1_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 0.1 --alpha 1 --beta 0.0001 
python -u train_all.py scale_1_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 1 --alpha 1 --beta 0.0001 
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 
# ablations on batchsize
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 4
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 8
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 16
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 32
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 64
python -u train_all.py scale_10_beta_0p0001_alpha_1 --dataset DomainNet --deterministic --trial_seed 0 --data_dir E:\dataset --algorithm SCL --scale 10 --alpha 1 --beta 0.0001 --resnet18 True --batch_size 128

