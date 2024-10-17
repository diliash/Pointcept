import argparse
import glob
import os

import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path", type=str, required=True)
    parser.add_argument("--export_path", type=str, default="./minsu3d_predictions/inference/val/predictions/instance")
    args = parser.parse_args()

    os.makedirs(f"{args.export_path}/predicted_masks", exist_ok=True)

    model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.preds_path}/*.pth")]

    for model_id in tqdm(model_ids):
        with open(f"{args.export_path}/{model_id}.txt", "w+") as scene_file:
            scene_predictions = torch.load(f"{args.preds_path}/{model_id}.pth")
            scene_masks = scene_predictions["pred_masks"].numpy()
            semantic_predictions = scene_predictions["pred_classes"].numpy()
            scores = scene_predictions["pred_scores"].numpy()
            for instance, mask in enumerate(scene_masks):
                scene_file.write(f"predicted_masks/{model_id}_{instance:03}.txt {semantic_predictions[instance]} {scores[instance]}\n")
                with open(f"{args.export_path}/predicted_masks/{model_id}_{instance:03}.txt", "w+") as mask_file:
                    for flag in mask:
                        mask_file.write(f"{flag}\n")
