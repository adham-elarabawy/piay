import sys

sys.path.append("../pkg")
from fid_score import calculate_fid_given_paths
import torch
from datasets import load_dataset
import os
import glob
from tqdm import tqdm
from jsonargparse import CLI
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_theme()


def main(
    download_dataset: bool = False,
    cuda_device: int = 0,
    eval_dataset_path: str = "test_dataset",
    dataset_slice: str = "test",
    model_revision: str = "baseline",
    limits: list = [10, 50, 100],
    outfile: str = None,
):
    if not outfile:
        outfile = f"outputs/{model_revision}/fid_vs_checkpoint_sample_size_sweep_{dataset_slice}.json"
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    if download_dataset:
        # download huggingface dataset into a folder
        dataset = load_dataset("adhamelarabawy/islamic_art")
        for i, row in enumerate(tqdm(dataset[dataset_slice])):
            row["img"].save(os.path.join(eval_dataset_path, f"{i}.png"))

    path0 = eval_dataset_path
    extract_ckpt = lambda x: int(x.split("-")[-1].strip("/"))

    fids = dict()
    dirs = sorted(glob.glob(f"../runs/{model_revision}/ckpt/*/"), key=extract_ckpt)
    for i, dir in enumerate(tqdm(dirs, desc="Checkpoints")):

        for limit in tqdm(limits, leave=False, desc="Limits"):
            ckpt = extract_ckpt(dir)
            path1 = os.path.join(dir, "samples")
            # path1 = f"runs/{model_revision}/ckpt/checkpoint-500/samples"

            fid = calculate_fid_given_paths(
                [path0, path1],
                batch_size=50,
                device=device,
                dims=2048,
                num_workers=8,
                limits=(None, limit),
            )
            # populate fids dict using both limit and ckpt
            if limit not in fids:
                fids[limit] = dict()
            fids[limit][ckpt] = fid

        for limit, line_data in fids.items():
            x = list(line_data.keys())
            y = list(line_data.values())
            plt.plot(x, y, label=limit)


        # Adding labels and legend
        plt.xlabel("Checkpoint")
        plt.ylabel("FID")
        plt.title("FID vs Checkpoint")
        plt.legend()
        # plt.show()
        # Displaying the plot
        plt.savefig(f"outputs/{model_revision}/fid_vs_checkpoint_{dataset_slice}.png")
        plt.clf()

        with open(outfile, "w") as file:
            json.dump(fids, file)


if __name__ == "__main__":
    CLI()
