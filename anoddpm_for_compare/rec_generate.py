import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
import os
import argparse
from pathlib import Path
import csv
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
from eval_sampler import DistributedEvalSampler
from munch import munchify
from torchvision.utils import save_image
from PIL import Image
from ldm_solvers import get_solver
from utils import create_workdir, set_seed
import pandas as pd
import numpy as np
import json
from accelerate import PartialState
from torchvision import transforms


class RecDataset(Dataset):
    def __init__(self, csv_path, image_root_path, transformer, out_dir):
        self.csv_path = csv_path
        self.image_root_path = image_root_path
        self.transformer = transformer
        csv_file = pd.read_csv(csv_path)
        self.file_names = list(csv_file["file_name"])
        print(f'Load for {len(list(csv_file["file_name"]))} samples')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        fn = self.file_names[index]
        image_path = os.path.join(self.image_root_path, fn)
        image = Image.open(image_path).convert("RGB")
        image = self.transformer(image)

        dir = fn.split('/')[0]
        fn = fn.split('/')[1]
        basename = fn.split('.')[0]
        return dir, fn, basename, image

def main():

    file = 'config_anoddpm.json'
    with open(f'configs/{file}', 'r') as f:
        f_args = json.load(f)

    parser = argparse.ArgumentParser(description="AnoDDPM")
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
    parser.add_argument("--method", type=str, default="ddim")

    args = parser.parse_args()
    set_seed(args.seed)
    distributed_state = PartialState()
    solver_config = munchify({'num_sampling': args.NFE})
    callback = None
    solver = get_solver(args.method,
                        pipeline_path=args.pipeline_path,
                        solver_config=solver_config,
                        device=distributed_state.device)

    csv_path = os.path.join(args.root_path, "metadata.csv")
    fieldnames = ["file_name", "mse_pixel", "lpips_pixel", "ssim_pixel"]
    output_loss_path = f"{args.workdir}/{args.sample_num}/{args.start_lambda}/loss.csv"  #
    output_rec_path = f"{args.workdir}/{args.sample_num}/{args.start_lambda}/"
    create_workdir(output_rec_path, "result")

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

    test_dataset = RecDataset(csv_path, args.root_path, img_transforms, output_rec_path)
    sampler = DistributedEvalSampler(test_dataset, rank=distributed_state.process_index, num_replicas=distributed_state.num_processes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=args.shuffle,
                                                  sampler=sampler
                                                  )

    loss_lpips = lpips.LPIPS(net='vgg').to(distributed_state.device)
    for i, (subfolder, fn, basename, src_img) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        src_img = src_img.to(distributed_state.device)
        with torch.no_grad():
            result = solver.sample_forward_backward(gt=src_img,
                                                     target_size=(256, 256),
                                                     callback_fn=callback,
                                                     start_lambda=args.start_lambda,
                                                     gamma=args.gamma
                                                    )
            lpips_pixel = loss_lpips(src_img, result).squeeze(1,2,3)
            image_np = (src_img / 2 + 0.5).clamp(0, 1).cpu().numpy()
            rec_np = (result / 2 + 0.5).clamp(0, 1).cpu().numpy()
            result = result / 2 + 0.5
        write_blocks = []
        for j in range(len(subfolder)):
            os.makedirs(os.path.join(output_rec_path, 'result', subfolder[j]), exist_ok=True)
            save_image(result[j],
                       args.workdir.joinpath(f'{output_rec_path}/result/{subfolder[j]}/{basename[j]}_rec.png'),
                       normalize=True)
            ssim_pixel = 1 - ssim(image_np[j], rec_np[j], channel_axis=0, data_range=1.0)
            mse_pixel = np.mean((image_np[j] - rec_np[j]) ** 2)

            write_blocks.append({
                "file_name": fn[j],
                "mse_pixel": str(mse_pixel),
                "lpips_pixel": str(lpips_pixel[j].item()),
                "ssim_pixel": str(ssim_pixel),
            })

        with open(output_loss_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(write_blocks)


    file.close()


if __name__ == "__main__":

    main()