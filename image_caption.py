import csv
import os
import numpy as np
import torch
import argparse
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from tqdm import tqdm

metastasis_text_prompts = [
    "sinus involvement",
    "large apocrine-like pleomorphic cells with pink, granular cytoplasm, large nuclei and prominent nucleoli",
    "cytoplasmic mucin",
    "Comedo, trabecular and papillary patterns",
    "balloon cell variant resembles histiocytes, although nuclei are atypical",
    "May expand sinuses",
    "Cuboidal cells with eosinophilic cytoplasm and central nucleus",
    "completely destroyed sinus architecture",
    "partially destroyed sinus architecture",
    "thickened capsule",
    "nodular growth pattern",
    "diffuse growth pattern",
    "Paracortical expansion",
    "Caseous necrosis",
    "Apoptosis",
    "Epithelioid cell clusters",
    "sinusoidal permeation",
    "discohesive cells",
    "cytokeratin",
    "complex or simple tubules with a compact glandular structure",
    "acini are lined by 2 or 3 layers of cells with basally oriented nuclei",
    "glandular arrangement",
    "glands acquire a haphazard arrangement with marked variation in size, shape, and outline",
    "glands are loosely and irregularly arranged",
    "glands with cells that have nuclear pseudostratification",
    "glands are lined by 3 or more layers",
    "central lumenal spaces of some small glands are filled by tumor cells producing small solid areas",
    "glandular structure is completely or almost completely lost",
    "cells grow predominantly in solid masses or cords",
    "large, highly irregular glands that frequently have outpouchings and microacinar forms",
    "small solid clusters or buds of tumor cells",
    "isolated or small clusters of malignant cells in the stroma",
    "cells are discontinuous from the more superficial malignant glands",
    "Microacinar structures",
    "small tubules that formed cribriform structures within medium or large gland",
    "small isolated round tubules within the stroma",
    "complex, irregular, cribriform glands and small solid areas",
    "tumor buds that emerge from medium-sized tubules",
    "glands are small, round, and microacinar",
    "Single and small clusters of undifferentiated cells are admixed",
    "microacinar architecture along its advancing edge",
    "Medium to small glands have an internal structure formed by microacini",
    "microacinar structures",
    "tumor budding",
    "undifferentiated cells",
    "advancing edge of the adenocarcinomas",
    "microacinar structures",
    "grade 3 adenocarcinoma",
    "irregularly folded, distorted, and small tubules"
]
normal_text_prompts = [
    'secondary lymphoid follicles',
    'histiocytes and high endothelial venules',
    'plasma cell-rich germinal center',
    'B-cell-rich non-germinal center',
    'large and small cleaved follicular center cells scant cytoplasm and inconspicuous nucleoli',
    'large B lymphocytes',
    'small unchallenged B cells',
    'plasmablasts',
    'abundant cytoplasm with medium to large nuclei with vesicular chromatin',
    'centroblasts',
    'tingible body macrophages',
    'tingible body macrophages have clear cytoplasm and contain apoptotic bodies',
    'mature T cells',
    'small dormant lymphocytes',
    'frequent mitotic figures',
    'trabeculae',
    'B immunoblasts',
    'sclerosis in an inguinal lymph node',
    'lymphoid nodules',
    'plasma cells',
    'germinal center',
    'recirculating cells',
    'thin connective tissue capsule',
    'open chromatin',
    'intranodal vessels',
    'mast cells',
    'smooth muscle proliferation in lymph node hilum',
    'macrophages',
    'histiocytes',
    'primary follicles',
    'medullary sinuses',
    'large noncleaved follicular center cells',
    'dense connective tissue',
    'centrocytes',
    'cells are elongated and resemble fibroblasts',
    'smooth muscle',
    'blood vessel',
    'endothelial cell',
    'cortex',
    'subcapsular sinuses',
    'small B and T lymphocytes',
    'interdigitating dendritic cells',
    'capsule',
    'lymphatic artery',
    'discontinuous endothelium',
    'lymphocytes',
    'afferent lymph vessels',
    'large B cells scattered throughout the paracortex',
    'distinct cytoplasmic boundaries',
    'faintly granular cytoplasm, large pale nuclei',
    'distinct group of non T, non B lymphocytes',
    'lymphatic vessels',
    'follicular dendritic cells',
    'peripheral nucleoli',
    'squamous endothelium',
    'mantle zone',
    'efferent vessels',
    'cortical sinuses',
    'coarse network of reticulin fibers',
    'postfollicular memory B cells',
    'medullary cords, sinuses and vessel',
    'tightly packed anastomosing networks',
    'straight branches',
    'prominent single nucleolus',
    'trabecular sinuses',
    'B cells',
    'erythrocytes',
    'arterioles',
    'plasmacytoid lymphocytes',
    'littoral cells',
    'plasmacytoid dendritic cells',
    'basophilic cytoplasm',
    'medulla',
    'memory B cells']
templates = [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E."
]

from PIL import Image
from torch.utils.data import Dataset

def number2mark(weight, base=1):
    def closest_bound(number, left_bound, right_bound):
        return min([left_bound, right_bound], key=lambda x: abs(number - x))

    return_str = ''
    if base == 1.1:
        while closest_bound(weight, 1, 1.1) == base:
            return_str += '+'
            weight /= base
    elif base == 0.9:
        while closest_bound(weight, 0.9, 1) == base:
            return_str += '-'
            weight /= base
    return return_str


class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir, preprocess):
        self.samples = []
        self.preprocess = preprocess

        for path, dir, files in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                file_path = os.path.join(path, filename)
                self.samples.append(file_path)

        print("Loaded {} samples".format(len(self.samples)))

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = self.preprocess(Image.open(image_path))
        image_path = '/'.join([image_path.split('/')[-2], image_path.split('/')[-1]])
        return image_path, image

    def __len__(self):
        return len(self.samples)


def detect_large_value_conch(sim_scores, k=5):
    sorted_index = np.argsort(sim_scores)[::-1]
    top_k_index = sorted_index[:k]
    top_k_scores = sim_scores[top_k_index]
    prompt_weights = top_k_scores / top_k_scores[k // 2]
    top_prompts = np.array(normal_text_prompts)[top_k_index]
    return list(zip(top_prompts, prompt_weights))


def get_text_feature(model, tokenizer):
    text_normal_embeds = []
    for template in templates:
        cur_prompts = []
        for prompt in normal_text_prompts:
            cur_prompts.append(template.replace("CLASSNAME", prompt))
        text_normal_tokens = tokenize(texts=cur_prompts, tokenizer=tokenizer).to('cuda')  # tokenize the text
        text_normal_embeds.append(model.encode_text(text_normal_tokens))
    text_normal_embeds = torch.stack(text_normal_embeds, dim=0)
    averaged_text_normal_embeds = torch.mean(text_normal_embeds, dim=0).to('cuda')
    text_features = averaged_text_normal_embeds
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def run(model, tokenizer, dataloader, output_path, tag=""):
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "text"])
        # Write the header
        writer.writeheader()

    with torch.inference_mode():
        text_features = get_text_feature(model, tokenizer).T

        for i, (image_paths, inputs) in tqdm(enumerate(dataloader), desc='Inference {}'.format(tag), total=len(dataloader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            image_embs = model.encode_image(inputs, proj_contrast=True, normalize=True)
            image_embs /= image_embs.norm(dim=-1, keepdim=True)
            sim_scores = torch.matmul(image_embs, text_features)
            sim_scores = sim_scores.cpu().numpy()

            for image_path, sim_score in zip(image_paths, sim_scores):
                promp_text = 'a histopathological photograph of '
                largest_index_list = detect_large_value_conch(sim_score)
                for i, (prompt, weight) in enumerate(largest_index_list):
                    if i == 0 or i == 1:
                        base = 1.1
                    elif i == 2:
                        base = 1
                    else:
                        base = 0.9
                    promp_text += f'({prompt}){number2mark(weight, base)} and '
                with open(output_path, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=["file_name", "text"])
                    # Write the header
                    writer.writerow({
                        "file_name": image_path,
                        "text": promp_text[:-5]
                    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create weighted prompts for lymphnode images")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
    model.to('cuda')
    model.eval()
    tokenizer = get_tokenizer()

    for folder in ["train", "test_in", "test_out"]:
        data_folder_path = os.path.join(args.data_root_path, folder)
        if not os.path.isdir(data_folder_path):
            continue
        output_path = os.path.join(data_folder_path, "metadata.csv")
        inference_dataset = ImageCaptionDataset(data_folder_path, preprocess)
        inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=args.batch_size, num_workers=32,
                                                       pin_memory=False, shuffle=False)

        largest_index_list = run(model, tokenizer, inference_loader, output_path=output_path, tag=folder)
