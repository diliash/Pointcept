import argparse
import os

import torch
from configs.partnetsim import inference_config
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models.point_group import PointGroup
from pointcept.models.swin3d import Swin3DUNet
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    save_path = args.save_path
    backbone_config = inference_config.model.pop("backbone")
    backbone_config.pop("type")
    inference_config.model.pop("type")
    backbone = Swin3DUNet(**backbone_config)
    model = PointGroup(backbone=backbone, build=False, **inference_config.model)

    val_data = build_dataset(inference_config.data["val"])

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            data = {k: v.cuda() for k, v in data.items()}
            pred = model(data)
            output_dict = {k: v.cpu() for k, v in pred.items()}

            os.makedirs(f"{save_path}/preds/inference", exist_ok=True)
            torch.save(output_dict, f"{save_path}/preds/inference/{val_loader.dataset.get_data_name(i)}.pth")
