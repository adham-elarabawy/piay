from diffusers import StableDiffusionPipeline
import torch
from jsonargparse import CLI
from typing import Union
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import create_repo, create_branch
from datasets import Dataset
import os

def main(
    base_model_id_or_path: str = "runwayml/stable-diffusion-v1-5",
    lora_model_id_or_path: str = "adhamelarabawy/piay",
    model_revision: str = "baseline",
    num_samples: int = 1000,
    cuda_device: int = 0,
    checkpoints: Union[str, int, list] = "all",
    prompt: str = "islamic art",
    seed: int = 2001,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id_or_path, torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(leave=False)

    # create pytorch generator for reproducibility
    generator = torch.Generator(device=f"cuda:{cuda_device}").manual_seed(seed)

    # dataset_id = create_repo(f"adhamelarabawy/piay", exist_ok=True, repo_type="dataset")
    # create_branch(dataset_id.repo_id, model_revision, exist_ok=False, repo_type="dataset")

    if checkpoints == "all":
        # TODO: Actually loop through the directories programmatically

        # every 500 checkpoints starting at 500 until 15000
        ckpt_nums = np.arange(500, 15500, 500)
    elif isinstance(checkpoints, int):
        ckpt_nums = [checkpoints]
    elif isinstance(checkpoints, str):
        ckpt_nums = [int(checkpoints)]
    elif isinstance(checkpoints, list):
        ckpt_nums = checkpoints
    else:
        raise ValueError(f"Invalid value for checkpoints: {checkpoints}")
    
    dir_names = [f"checkpoint-{ckpt_num}" for ckpt_num in ckpt_nums]

    # data_dict = {"image": [], "checkpoint": []}
    for dir in tqdm(dir_names, desc="Checkpoints"):
        

        pipe.unet.load_attn_procs(
            lora_model_id_or_path,
            weight_name="pytorch_model.bin",
            revision=model_revision,
            subfolder=dir,
        )
        pipe.to(f"cuda:{cuda_device}")

        for i in tqdm(range(num_samples), desc="Sampling Images", leave=False):
            out = pipe(prompt, generator=generator, ).images[0]
            save_path = f"../runs/{model_revision}/ckpt/{dir}/samples/{i}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out.save(save_path)
            # data_dict["image"].append(out)
            # data_dict["checkpoint"].append(dir)

    breakpoint()


if __name__ == "__main__":
    CLI()
