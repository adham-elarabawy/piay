from jsonargparse import CLI

from clip_retrieval.clip_client import ClipClient, Modality
from tqdm.auto import tqdm
import random
from utils import image_grid, download_image, filter_img_res, resize_and_center_crop
from datasets import Dataset


def create_dataset(ds_request_size: int, prompt: str, aesthetic_score: int = 9, aesthetic_weight: float = 0.9, min_res: int = 512, center_crop: bool = True):
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=ds_request_size, aesthetic_weight=0.9, aesthetic_score=9)

    results = client.query(text=prompt)

    for res in tqdm(results, desc="Downloading Images"):
        img = download_image(res["url"])
        res["img"] = img

    for res in tqdm(results, desc="Filtering by resolution"):
        res["img"] = filter_img_res(res["img"], min_res, min_res)

    results = [res for res in tqdm(results, desc="Filtering Images") if res["img"]]

    for res in tqdm(results, desc="Applying center crop"):
        res["img"] = resize_and_center_crop(res["img"], res=min_res)


    Dataset.from_list(results).train_test_split(0.2, shuffle=True).push_to_hub("adhamelarabawy/islamic_art")

if __name__ == "__main__":
    CLI(as_positional=False)


