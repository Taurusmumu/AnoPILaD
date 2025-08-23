import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import argparse
from pathlib import Path
import os
import csv
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from eval_sampler import DistributedEvalSampler
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from munch import munchify
from torchvision.utils import save_image
from PIL import Image
from ldm_solvers import get_solver
from utils import create_workdir, set_seed
import pandas as pd
import numpy as np
import lpips
import json
from accelerate import PartialState
from torchvision import transforms


class RecDataset(Dataset):
    def __init__(self, csv_path, image_root_path, transformer, out_dir):
        self.csv_path = csv_path
        self.image_root_path = image_root_path
        self.transformer = transformer
        csv_file = pd.read_csv(csv_path)
        print(f'Load for {len(list(csv_file["file_name"]))} samples')

        done_fn_list = []
        for path, dir, files in os.walk(os.path.join(out_dir, "result")):
            for fn in files:
                ext = os.path.splitext(fn)[-1]
                if ext.lower() not in ('.png', '.jpg'):
                    continue
                file_path = os.path.join(path.split('/')[-1], fn)
                done_fn_list.append(file_path)

        done_fn_list = [f"{fn.split('.')[0][:-4]}.png" for fn in done_fn_list]
        csv_file = csv_file.loc[~csv_file["file_name"].isin(done_fn_list)]
        self.file_names = list(csv_file["file_name"])
        self.texts = list(csv_file["text"])
        print(f'Again Load for {len(list(csv_file["file_name"]))} samples')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        fn = self.file_names[index]
        text = self.texts[index]
        image_path = os.path.join(self.image_root_path, fn)
        image = Image.open(image_path).convert("RGB")
        image = self.transformer(image)

        dir = fn.split('/')[0]
        fn = fn.split('/')[1]
        basename = fn.split('.')[0]
        return dir, fn, basename, image, text

def main():
    file = 'config.json'
    with open(file, 'r') as f:
        f_args = json.load(f)

    parser = argparse.ArgumentParser(description="AnoPILAD")
    parser.add_argument("--workdir", type=Path, default=f_args["workdir"])
    parser.add_argument("--dir_type", type=str, default=f_args["dir_type"])
    parser.add_argument("--pipeline_path", type=str, default=f_args["pipeline_path"])
    parser.add_argument("--data_path", type=str, default=f_args["data_path"])
    parser.add_argument("--NFE", type=int, default=f_args["NFE"])
    parser.add_argument("--start_lambda", type=int, default=f_args["start_lambda"])
    parser.add_argument("--img_resolution", type=Path, default=f_args["img_resolution"])
    parser.add_argument("--num_workers", type=str, default=f_args["num_workers"])
    parser.add_argument("--batch_size", type=str, default=f_args["batch_size"])
    parser.add_argument("--seed", type=int, default=88888888)
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default="ddim")

    args = parser.parse_args()
    set_seed(args.seed)
    distributed_state = PartialState()
    solver_config = munchify({'num_sampling': args.NFE})
    solver = get_solver(args.method,
                        pipeline_path=args.pipeline_path,
                        solver_config=solver_config,
                        device=distributed_state.device)

    csv_path = os.path.join(args.data_path, "metadata.csv")
    fieldnames = ["file_name", "mse_latent", "lpips_pixel"]
    output_loss_path = os.path.join(args.workdir, args.dir_type, str(args.start_lambda), 'loss.csv')
    output_rec_path = os.path.join(args.workdir, args.dir_type, str(args.start_lambda))
    create_workdir(output_rec_path, "image_result")

    if not os.path.isfile(output_loss_path):
        with open(output_loss_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Write the header
            writer.writeheader()

    img_transforms = transforms.Compose(
        [
            transforms.Resize(args.img_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    test_dataset = RecDataset(csv_path, args.data_path, img_transforms, output_rec_path)
    sampler = DistributedEvalSampler(test_dataset, rank=distributed_state.process_index, num_replicas=distributed_state.num_processes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False,
                                                  sampler=sampler
                                                  )
    loss_lpips = lpips.LPIPS(net='vgg').to(distributed_state.device)
    for i, (subfolder, fn, basename, src_img, prompt) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        src_img = src_img.to(distributed_state.device)
        with torch.no_grad():
            result, loss = solver.sample_forward_backward(gt=src_img,
                                                             prompt=[list(np.repeat("", len(subfolder))),
                                                                     list(prompt)],
                                                             cfg_guidance=args.cfg_guidance,
                                                             target_size=(256, 256),
                                                             start_lambda=args.start_lambda,
                                                             gamma=1e-1)
            lpips_pixel = loss_lpips(src_img, result).squeeze(1, 2, 3)
            result = result / 2 + 0.5
        write_blocks = []
        for j in range(len(subfolder)):
            os.makedirs(os.path.join(output_rec_path, 'image_result', subfolder[j]), exist_ok=True)
            save_image(result[j], f'{output_rec_path}/image_result/{subfolder[j]}/{basename[j]}_rec.png', normalize=True)
            write_blocks.append({
                "file_name": fn[j],
                "mse_latent": str(loss[j].item()),
                "lpips_pixel": str(lpips_pixel[j].item()),
            })
        with open(output_loss_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(write_blocks)


if __name__ == "__main__":

    main()
