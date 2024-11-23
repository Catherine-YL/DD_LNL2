For synthetic noise: `noise_type` and `noise_rate` can be changed .

For CIFAR-N: `noise_type` can be changed, `is_annot` and `is_human` is required.

## DATM

### Generate expert trajectories

```
cd buffer
```

#### CIFAR-10:

```
clean:
	python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --zca --buffer_path=../buffer_storage/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0.
symmetric、asymmetric(symmetric_0.2):
	python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --zca --buffer_path=../buffer_storage_symmetric_0.2/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0. --noise_type symmetric --noise_rate 0.2
annot(aggre):
	python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --zca --buffer_path=../buffer_storage_aggre/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0. --noise_type aggre --is_annot --is_human
```

#### CIFAR-100:

```
clean:
	python buffer_FTD.py --dataset=CIFAR100 --model=ConvNet --zca --buffer_path=../buffer_storage/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0.
symmetric、asymmetric(symmetric_0.2):
	python buffer_FTD.py --dataset=CIFAR100 --model=ConvNet --zca --buffer_path=../buffer_storage_symmetric_0.2/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0. --noise_type symmetric --noise_rate 0.2
annot(noisy100):
	python buffer_FTD.py --dataset=CIFAR100 --model=ConvNet --zca --buffer_path=../buffer_storage_noisy100/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0. --noise_type noisy100 --is_annot --is_human
```

#### Tiny-ImageNet:

```
clean:
	python buffer_FTD.py --dataset=Tiny --model=ConvNetD4 --zca --buffer_path=../buffer_storage/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0.
symmetric(symmetric_0.2):
	python buffer_FTD.py --dataset=Tiny --model=ConvNetD4 --zca --buffer_path=../buffer_storage_symmetric_0.2/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --mom=0. --noise_type symmetric --noise_rate 0.2
```

### Start distillation

All noise type configurations are in the 'configs' folder

```
cd distill
python DATM_tesla.py --cfg ../configs/xxxx.yaml
```

### Evaluation

```
cd distill
python evaluation.py --lr_dir=path_to_lr --data_dir=path_to_images --label_dir=path_to_labels --zca
```

## DANCE

### Generate pre-training buffers

#### CIFAR-10:

```
clean:
	python pretrain.py -d cifar10 --reproduce
symmetric、asymmetric(symmetric_0.2):
	python pretrain.py -d cifar10 --reproduce --noise_type symmetric --noise_rate 0.2
annot:
	python pretrain.py -d cifar10 --reproduce --noise_type aggre --is_annot --is_human
```

#### CIFAR-100:

```
clean:
	python pretrain.py -d cifar100 --reproduce
symmetric、asymmetric(symmetric_0.2):
	python pretrain.py -d cifar100 --reproduce --noise_type symmetric --noise_rate 0.2
annot:
	python pretrain.py -d cifar100 --reproduce --noise_type noisy100 --is_annot --is_human
```

#### Tiny-ImageNet:

```
clean:
	python pretrain.py -d tinyimagenet --reproduce
symmetric(symmetric_0.2):
	python pretrain.py -d tinyimagenet --reproduce --noise_type symmetric --noise_rate 0.2
```

### Start distillation

#### CIFAR-10:

```
clean:
  python DANCE.py -d cifar10 --ipc 1 --factor 2 --reproduce
symmetric、asymmetric(symmetric_0.2):
  python DANCE.py -d cifar10 --ipc 1 --factor 2 --reproduce --noise_type symmetric --noise_rate 0.2
annot(aggre):
  python DANCE.py -d cifar10 --ipc 1 --factor 2 --reproduce --noise_type aggre --is_annot --is_human
```

#### CIFAR-100:

```
clean:
  python DANCE.py -d cifar100 --ipc 1 --factor 2 --reproduce
symmetric、asymmetric(symmetric_0.2):
  python DANCE.py -d cifar100 --ipc 1 --factor 2 --reproduce --noise_type symmetric --noise_rate 0.2
annot(noisy100):
  python DANCE.py -d cifar100 --ipc 1 --factor 2 --reproduce --noise_type aggre --is_annot --is_human
```

#### Tiny-ImageNet:

```
clean:
  python DANCE.py -d tinyimagenet --ipc 1 --factor 2 --reproduce
symmetric(symmetric_0.2):
  python DANCE.py -d tinyimagenet --ipc 1 --factor 2 --reproduce --noise_type symmetric --noise_rate 0.2
```



## RCIG

- For MNIST, and Fashion-MNIST, we use the config `depth_3_no_flip.txt`
- For CIFAR-10, CIFAR-100, and CUB-200, we use config `depth_3.txt`
- For CIFAR-100 with 50 ipc we use config `cifar100_50.txt`
- For Tiny-ImageNet 1 ipc, we use config `depth_4_200.txt`
- For Tiny-ImageNet 10 ipc and resized Imagenet, we use config `depth_4_big.txt`
- For ImageNette and ImageWoof, we use config `depth_5.txt`

### Start distillation

#### CIFAR-10:

```
clean:
	python3 distill_dataset.py --dataset_name cifar10 --n_images 1 --output_dir ./output_dir/cifar10_ipc1_clean_0 --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0
symmetric、asymmetric(symmetric_0.2):
	python3 distill_dataset.py --dataset_name cifar10 --n_images 1 --output_dir ./output_dir/cifar10_ipc1_symmetric_0.2 --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0 --noise_type symmetric --noise_rate 0.2
annot:
	python3 distill_dataset.py --dataset_name cifar10 --n_images 1 --output_dir ./output_dir/cifar10_ipc1_aggre --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0 --noise_type aggre --is_annot
```

#### CIFAR-100:

```
clean:
	python3 distill_dataset.py --dataset_name cifar100 --n_images 1 --output_dir ./output_dir/cifar100_ipc1_clean_0 --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0
symmetric、asymmetric(symmetric_0.2):
	python3 distill_dataset.py --dataset_name cifar100 --n_images 1 --output_dir ./output_dir/cifar100_ipc1_symmetric_0.2 --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0 --noise_type symmetric --noise_rate 0.2
annot:
	python3 distill_dataset.py --dataset_name cifar100 --n_images 1 --output_dir ./output_dir/cifar100_ipc1_noisy100 --max_steps 10000 --config_path ./configs_final/depth_3.txt --random_seed 0 --noise_type noisy100 --is_annot
```



#### Tiny-ImageNet:

```
clean:
	python3 distill_dataset.py --dataset_name tiny_imagenet --n_images 1 --output_dir ./output_dir/tiny_ipc1_clean_0 --max_steps 10000 --config_path ./configs_final/depth_4_200.txt --random_seed 0
symmetric(symmetric_0.2):
	python3 distill_dataset.py --dataset_name tiny_imagenet --n_images 1 --output_dir ./output_dir/tiny_ipc1_symmetric_0.2 --max_steps 10000 --config_path ./configs_final/depth_4_200.txt --random_seed 0 --noise_type symmetric --noise_rate 0.2
```

### Evaluation

```
python3 eval.py --dataset_name cifar10 --checkpoint_path ./output_dir/cifar10_ipc1_clean_0/checkpoint_final/ --config_path ./configs_final/depth_3.txt --random_seed 0
```

