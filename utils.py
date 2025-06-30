import torch
from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import math
from typing import Union, Tuple

# 修改后的 test_on_mff_dataset 函数
def test_on_mff_dataset(model, dataset_path, save_path, device, force_size: Union[Tuple[int, int], None] = None):
    """
    对单个数据集进行测试。

    Args:
        model: 用于融合的PyTorch模型。
        dataset_path: 包含两个子目录（如A和B）的数据集路径，每个子目录包含成对的图像。
        save_path: 保存融合结果的路径。将创建 'y'、'result' 和 'decision_map' 子目录。
        device: 运行模型的设备 (例如 'cuda' 或 'cpu')。
        force_size: 可选参数。如果提供 (例如 (512, 512))，则强制将输入图像调整为此尺寸进行处理。
                    如果为 None，则将图像尺寸向上调整到最接近的32的倍数。
    
    Returns:
        float: 整个数据集的平均SSIM值
    """
    transform = transforms.Compose([transforms.ToTensor()])  # 假设只包含ToTensor

    # 获取数据集中的图像
    try:
        sub_dirs = sorted(os.listdir(dataset_path))
        if len(sub_dirs) < 2:
             raise ValueError(f"数据集路径 '{dataset_path}' 应至少包含两个子目录。")
        dir_A = os.path.join(dataset_path, sub_dirs[0])
        dir_B = os.path.join(dataset_path, sub_dirs[1])
        if not os.path.isdir(dir_A) or not os.path.isdir(dir_B):
             raise ValueError(f"路径 '{dir_A}' 或 '{dir_B}' 不是有效的目录。")
    except FileNotFoundError:
        print(f"错误：找不到数据集路径 '{dataset_path}'")
        return 0.0
    except Exception as e:
        print(f"错误：处理数据集路径时出错: {e}")
        return 0.0

    # 按数字顺序排序图像 (假设文件名末尾是数字且扩展名为 .png/.jpg)
    try:
        img_names = sorted([f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f))],
                           key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        print(f"警告：目录 {dir_A} 中的文件名可能不全是数字结尾，尝试按字符串排序。")
        img_names = sorted([f for f in os.listdir(dir_A) if os.path.isfile(os.path.join(dir_A, f))])
    except FileNotFoundError:
         print(f"错误：找不到目录 '{dir_A}' 或无法列出文件。")
         return 0.0

    # 创建保存路径
    y_save_dir = os.path.join(save_path, 'y')
    result_save_dir = os.path.join(save_path, 'result')
    os.makedirs(y_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)

    model.eval()
    total_ssim = 0.0
    processed_images = 0

    with torch.no_grad():
        for img_name in tqdm(img_names, desc='处理图像'):
            img0_path = os.path.join(dir_A, img_name)
            img1_path = os.path.join(dir_B, img_name)

            # 检查 B 目录中是否存在对应图像
            if not os.path.exists(img1_path):
                print(f"警告：在目录 {dir_B} 中找不到对应的图像 {img_name}，跳过此对。")
                continue

            try:
                # 使用 PIL 打开图像
                img0_pil = Image.open(img0_path).convert('YCbCr')
                img1_pil = Image.open(img1_path).convert('YCbCr')

                # 提取 Y 通道
                img0_Y = img0_pil.split()[0]
                img1_Y = img1_pil.split()[0]

                # 记录原始尺寸 (宽, 高)
                original_w, original_h = img0_Y.size

                # --- 核心修改：根据 force_size 决定目标尺寸 ---
                if force_size:
                    # 如果提供了 force_size，直接使用它
                    if not (isinstance(force_size, tuple) and len(force_size) == 2 and
                            isinstance(force_size[0], int) and isinstance(force_size[1], int)):
                         print(f"错误：force_size 参数必须是 (宽度, 高度) 的整数元组，例如 (512, 512)。当前值：{force_size}")
                         return 0.0
                    target_w, target_h = force_size
                    print(f"信息：强制调整图像 {img_name} 尺寸为 {target_w}x{target_h}")
                else:
                    # 否则，计算调整后的目标尺寸 (32的倍数，向上取整)
                    target_w = math.ceil(original_w / 32) * 32
                    target_h = math.ceil(original_h / 32) * 32

                # 将 Y 通道图像调整到目标尺寸
                img0_Y_resized = img0_Y.resize((target_w, target_h), Image.Resampling.LANCZOS)
                img1_Y_resized = img1_Y.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # 转换为 tensor 并归一化到 [0,1]
                img0_Y_tensor = transform(img0_Y_resized).unsqueeze(0).to(device)
                img1_Y_tensor = transform(img1_Y_resized).unsqueeze(0).to(device)

                # 进行融合 (输入是调整后的尺寸)
                fused_output = model(img0_Y_tensor, img1_Y_tensor)  # 现在模型只返回一个输出

                # 处理融合输出
                fused_np = fused_output.squeeze().cpu().numpy()
                min_val, max_val = np.min(fused_np), np.max(fused_np)
                if max_val > min_val:
                    fused_np_normalized = (fused_np - min_val) / (max_val - min_val)
                else:
                    fused_np_normalized = np.zeros_like(fused_np)
                fused_np_scaled = np.clip(fused_np_normalized * 255, 0, 255).astype(np.uint8)
                fused_pil = Image.fromarray(fused_np_scaled, mode='L')
                fused_pil_resized = fused_pil.resize((original_w, original_h), Image.Resampling.LANCZOS)

                # 确保保存的文件名有正确的扩展名
                base_name, _ = os.path.splitext(img_name)
                save_img_name = base_name + '.png'

                # 保存 Y 通道结果图像
                y_save_path = os.path.join(y_save_dir, save_img_name)
                fused_pil_resized.save(y_save_path)

                # 进行颜色通道融合
                result_path = os.path.join(result_save_dir, save_img_name)
                try:
                    fuse_RGB_channels(img0_path, img1_path, y_save_path, result_path)
                except Exception as e:
                    print(f"错误: 调用 fuse_RGB_channels 处理图像 {img_name} 时发生异常: {e}")

            except FileNotFoundError as e:
                 print(f"错误：找不到文件 {e.filename}，跳过此图像对。")
            except Exception as e:
                print(f"处理图像 {img_name} 时发生未知错误: {e}")

    return 0.0


def fuse_RGB_channels(img_A_path, img_B_path, fused_Y_path, save_path):
    """融合RGB通道，只使用CV2"""
    # 读取源图像
    img_A = cv2.imread(img_A_path)  # BGR格式
    img_B = cv2.imread(img_B_path)  # BGR格式

    # 记录原始尺寸
    original_h, original_w = img_A.shape[:2]

    # 转换到YCrCb空间
    ycrcb_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2YCrCb)
    ycrcb_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2YCrCb)

    # 分离通道
    _, Cr1, Cb1 = cv2.split(ycrcb_A)  # 注意CV2的YCrCb顺序是Y, Cr, Cb
    _, Cr2, Cb2 = cv2.split(ycrcb_B)

    # 确保所有通道尺寸一致
    # 从路径读取融合后的Y通道 (假设它是一个灰度图像文件)
    fused_Y = cv2.imread(fused_Y_path, cv2.IMREAD_GRAYSCALE)
    if fused_Y is None:
        raise ValueError(f"无法从路径 '{fused_Y_path}' 读取融合后的Y通道图像")
    if fused_Y.shape[:2] != (original_h, original_w):
        fused_Y = cv2.resize(fused_Y, (original_w, original_h))

    # 融合Cr和Cb通道（使用权重融合）
    weights_Cr1 = np.abs(Cr1.astype(np.float32) - 128)
    weights_Cr2 = np.abs(Cr2.astype(np.float32) - 128)
    weights_sum_Cr = weights_Cr1 + weights_Cr2

    weights_Cb1 = np.abs(Cb1.astype(np.float32) - 128)
    weights_Cb2 = np.abs(Cb2.astype(np.float32) - 128)
    weights_sum_Cb = weights_Cb1 + weights_Cb2

    # 避免除零
    Cr = np.where(weights_sum_Cr == 0, 128,
                  (Cr1.astype(np.float32) * weights_Cr1 + Cr2.astype(np.float32) * weights_Cr2) / weights_sum_Cr)
    Cb = np.where(weights_sum_Cb == 0, 128,
                  (Cb1.astype(np.float32) * weights_Cb1 + Cb2.astype(np.float32) * weights_Cb2) / weights_sum_Cb)

    # 合并通道
    fused_ycrcb = cv2.merge([fused_Y, np.uint8(Cr), np.uint8(Cb)])

    # 转换回BGR空间
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

    # 保存结果
    cv2.imwrite(save_path, fused_bgr)