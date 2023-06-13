# piay
Official Implementation of "Perturbation Is All You Need"

## Dataset
[Link to HF Dataset.](https://huggingface.co/datasets/adhamelarabawy/islamic_art)

## Baselines
### Stable Diffusion 2.1
<img src='https://github.com/adham-elarabawy/piay/assets/9634713/5c17ee3d-9af0-4cf0-b927-f2808bcd8914' width='256'>

### DeepFloyd IF
<img src='https://github.com/adham-elarabawy/piay/assets/9634713/1332270c-d40c-4ec9-8a56-b5a5f468370d' width='256'>


## Instructions for Reproducability
### Create the Dataset
`python create_dataset.py --ds_request_size 10000 --prompt "beautiful islamic art, geometric, intricate, blue, shutterstock"`

## Accelerate Config
 - compute_environment: LOCAL_MACHINE
 - distributed_type: 'NO'
 - downcast_bf16: 'no'
 - gpu_ids: '0'
 - machine_rank: 0
 - main_training_function: main
 - mixed_precision: bf16
 - num_machines: 1
 - num_processes: 1
 - rdzv_backend: static
 - same_network: true
 - tpu_env: []
 - tpu_use_cluster: false
 - tpu_use_sudo: false
 - use_cpu: false
