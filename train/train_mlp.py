import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import alpha_clip
from model.model import IM2TEXT
from sav_dataset.test_dataloader2 import SAVMaskletDataset
from torch.utils.data import DataLoader
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def preprocess_images(images, device):
    images = images.permute(0, 3, 1, 2).float() / 255.0
    images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    return images.to(device)

def preprocess_masks(masks, device):
    masks = masks.unsqueeze(1).float() / 255.0
    masks = torch.nn.functional.interpolate(masks, size=(224, 224), mode='nearest')
    mean = torch.tensor([0.5], device=device).view(1, 1, 1, 1)
    std = torch.tensor([0.26], device=device).view(1, 1, 1, 1)
    masks = (masks - mean) / std
    return masks.to(device)

def get_text_features(model, token_features, device):
    batch_size = token_features.size(0)
    # Tokenize the template "a photo of"
    text_tokens = clip.tokenize(["a photo of"] * batch_size).to(device)
    eot_token_id = 49407
    collect_ind = text_tokens == eot_token_id  # 获取结束标记的索引
    collect_ind = collect_ind.nonzero()[:, 1]  # 提取结束标记的位置

    # 获取 token 的嵌入
    x = model.token_embedding(text_tokens).type(model.dtype)  # [batch_size, n_ctx, d_model]

    # 插入伪词 token 到嵌入中
    token_features = token_features.view(batch_size, 1, -1)  # [batch_size, 1, embed_dim]
    x = torch.cat([x[:, :collect_ind[0]], token_features, x[:, collect_ind[0]:-1]], dim=1)  # 插入到collect_ind位置

    # 加上位置嵌入
    x = x + model.positional_embedding[:x.size(1), :].type(model.dtype)

    # 通过 transformer 进行编码
    x = x.permute(1, 0, 2)  # [sequence_length, batch_size, embed_dim]
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # [batch_size, sequence_length, embed_dim]

    # Layer Normalization 和类型转换
    x = model.ln_final(x).type(model.dtype)

    # 提取伪词位置的文本特征 (结束标记后的位置)
    text_features = x[torch.arange(x.size(0)), collect_ind + 1] @ model.text_projection

    return text_features

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 alpha_clip 模型
    alpha_model, _ = alpha_clip.load(
        "ViT-L/14",
        alpha_vision_ckpt_pth="/ailab/user/mahaoxuan/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth",  # 替换为您的 alpha_clip 模型路径
        device=device
    )
    alpha_model.eval()

    # 加载 CLIP 模型
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    # 加载训练好的 img2text 模型
    img2text = IM2TEXT(
        embed_dim=768,
        middle_dim=1024,
        output_dim=clip_model.token_embedding.weight.shape[1],
        n_layer=4
    ).to(device)

    # 加载训练好的权重
    checkpoint_path = '/ailab/user/mahaoxuan/AlphaCLIP/result/result_512_middle_1024_split_60/img2text_epoch_26.pth'  # 替换为您的模型路径
    img2text.load_state_dict(torch.load(checkpoint_path, map_location=device))
    img2text.eval()

    # 选择一个 epoch 的 JSON 文件
    epoch = 26  # 您可以选择 26 到 59 之间的任意一个 epoch
    json_file = f"/ailab/user/mahaoxuan/data/SAM/test_pre_quick/prepared_data_004.json"

    # 加载 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 随机选择一个样本
    sample = random.choice(data)

    # 获取图像和掩码的路径
    image_path = sample['img_path']  # 假设 JSON 中的图像路径键名为 'img_path'
    mask_path = sample['mask_path']  # 假设 JSON 中的掩码路径键名为 'mask_path'

    # 由于可能存在相对路径或其他路径问题，请确保路径正确
    # 如果需要，可以修改为完整路径
    # image_path = os.path.join('/your/image/root/path', sample['img_path'])
    # mask_path = os.path.join('/your/mask/root/path', sample['mask_path'])

    # 加载并预处理图像和掩码
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # 转换为 numpy 数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 将图像和掩码转换为张量并添加批次维度
    images = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W, 3]
    masks = torch.from_numpy(mask_np).unsqueeze(0)    # [1, H, W]

    # 预处理
    images = preprocess_images(images, device)
    masks = preprocess_masks(masks, device)

    # 获取图像特征和 MLP 输出
    with torch.no_grad():
        image_features = alpha_model.visual(images, masks)
        token_features = img2text(image_features)

        # 可选步骤：获取文本特征
        text_features = get_text_features(clip_model, token_features, device)

    # 打印或处理 MLP 的输出
    print("Token features (MLP output):", token_features.cpu().numpy())
    print("Text features:", text_features.cpu().numpy())

    # 如果需要，可以计算与一些预定义文本的相似度
    texts = ["a cat", "a dog", "a car", "a tree"]
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_embeds = clip_model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * text_features_normalized @ text_embeds.T).softmax(dim=-1)
        print("Similarities with predefined texts:", similarities.cpu().numpy())

if __name__ == "__main__":
    main()
