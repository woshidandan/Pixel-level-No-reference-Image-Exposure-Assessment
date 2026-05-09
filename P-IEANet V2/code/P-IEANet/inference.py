import torch
from torchvision import transforms
import cv2
import numpy as np
from wavelet_network import Wavelet_Net
import option
import os

def load_image(image_path):
    # 读取图像并转换为Tensor
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]  # 保存原始尺寸
    img = img / 255.0  # 归一化
    
    # 与make_heatmap.py保持一致的变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # HWC -> CHW, [0,1]
        transforms.Resize((256, 256)),  # 调整图像大小为256x256
        transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])  # 添加normalize
    ])
    img_tensor = transform(img).float().unsqueeze(0)  # 增加batch维
    return img_tensor, (original_height, original_width)

def save_tensor_as_image(tensor, filename, original_size=None):
    # 将网络输出Tensor保存为图片
    tensor = tensor.squeeze(0).detach().cpu()
    
    # 如果提供了原始尺寸，先恢复到原始尺寸
    if original_size is not None:
        original_height, original_width = original_size
        resize_transform = transforms.Resize((original_height, original_width))
        tensor = resize_transform(tensor)
    
    # 检查tensor的通道数
    print(f"Tensor形状: {tensor.shape}")
    print(f"通道数: {tensor.shape[0]}")
    
    if tensor.shape[0] == 3:  # RGB图像
        numpy_image = tensor.permute(1, 2, 0).numpy()
        numpy_image = (numpy_image / np.max(numpy_image) * 255).astype(np.uint8)
        print(f"RGB图像numpy形状: {numpy_image.shape}")
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, numpy_image)
    elif tensor.shape[0] == 1:  # 灰度图像
        numpy_image = tensor.squeeze(0).numpy()
        numpy_image = (numpy_image / np.max(numpy_image) * 255).astype(np.uint8)
        cv2.imwrite(filename, numpy_image)
    else:  # 多通道特征图，取前3个通道作为RGB
        print(f"多通道特征图，通道数: {tensor.shape[0]}")
        # 如果通道数大于3，取前3个通道
        if tensor.shape[0] >= 3:
            tensor = tensor[:3]  # 取前3个通道
        else:
            # 如果通道数小于3，复制通道到3个通道
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] == 2:
                # 对于2通道，复制第一个通道作为第三个通道
                tensor = torch.cat([tensor, tensor[0:1]], dim=0)
        
        numpy_image = tensor.permute(1, 2, 0).numpy()
        numpy_image = (numpy_image / np.max(numpy_image) * 255).astype(np.uint8)
        print(f"处理后numpy形状: {numpy_image.shape}")
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, numpy_image)

def save_individual_channels(tensor, base_filename, original_size=None, num_channels=3):
    """分别保存前N个通道的可视化图片"""
    tensor = tensor.squeeze(0).detach().cpu()
    
    # 如果提供了原始尺寸，先恢复到原始尺寸
    if original_size is not None:
        original_height, original_width = original_size
        resize_transform = transforms.Resize((original_height, original_width))
        tensor = resize_transform(tensor)
    
    print(f"分别保存前{num_channels}个通道，tensor形状: {tensor.shape}")
    
    for i in range(min(num_channels, tensor.shape[0])):
        # 获取单个通道
        channel = tensor[i]
        
        # 归一化到0-255
        channel_np = channel.numpy()
        channel_np = (channel_np / np.max(channel_np) * 255).astype(np.uint8)
        
        # 生成文件名
        filename = f"{base_filename}_channel_{i}.png"
        
        # 保存为灰度图像
        cv2.imwrite(filename, channel_np)
        print(f"保存通道{i}到: {filename}")
        
        # 同时保存为热力图风格的彩色图像
        colormap_filename = f"{base_filename}_channel_{i}_colormap.png"
        colored = cv2.applyColorMap(channel_np, cv2.COLORMAP_JET)
        cv2.imwrite(colormap_filename, colored)
        print(f"保存通道{i}热力图到: {colormap_filename}")

def save_heatmap_style_output(tensor, base_filename, original_size=None):
    """模拟make_heatmap.py的处理方式"""
    tensor = tensor.squeeze(0).detach().cpu()
    
    # 如果提供了原始尺寸，先恢复到原始尺寸
    if original_size is not None:
        original_height, original_width = original_size
        resize_transform = transforms.Resize((original_height, original_width))
        tensor = resize_transform(tensor)
    
    # 取第一个通道（对应make_heatmap.py中的exposure[0]）
    exposure = tensor[0]
    
    # 符号翻转（对应make_heatmap.py中的exposure*-1）
    exposure = exposure * -1
    
    # 值域重新映射（对应tensor_to_heatmap中的处理）
    exposure = (exposure + 1) / 2
    exposure = exposure.clamp(0, 1)
    
    # 保存为与make_heatmap.py风格一致的图像
    exposure_np = exposure.numpy()
    exposure_np = (exposure_np * 255).astype(np.uint8)
    
    # 保存灰度版本
    filename = f"{base_filename}_heatmap_style.png"
    cv2.imwrite(filename, exposure_np)
    print(f"保存热力图风格输出到: {filename}")
    
    # 保存彩色热力图版本
    colormap_filename = f"{base_filename}_heatmap_style_colormap.png"
    colored = cv2.applyColorMap(exposure_np, cv2.COLORMAP_JET)
    cv2.imwrite(colormap_filename, colored)
    print(f"保存彩色热力图到: {colormap_filename}")

def inference_with_double_input(model, img_tensor, device):
    """使用双输入进行推理，模拟make_heatmap.py的方式"""
    # 复制输入图像（对应make_heatmap.py中的torch.cat([image, image], dim=0)）
    double_input = torch.cat([img_tensor, img_tensor], dim=0)
    
    with torch.no_grad():
        output_tensor = model(double_input)
    
    # 取第一个样本的输出（对应make_heatmap.py中的exposure[0]）
    return output_tensor[0:1]  # 保持batch维度

if __name__ == "__main__":
    # ===== 配置参数 =====
    input_image_path = "heat_map_input/portrait_1.jpg"        # 输入图像路径
    output_image_path = "heat_map_output/portrait_1_output.png"      # 输出保存路径
    model_weight_path = "wavelet_epoch_2_val_loss0.03022034629540784.pth"  # 模型权重路径

    # ===== 加载配置 =====
    opt = option.init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===== 加载模型 =====
    model = Wavelet_Net()
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 加载输入图像 =====
    img_tensor, original_size = load_image(input_image_path)
    img_tensor = img_tensor.to(device)

    # ===== 推理 =====
    print("=== 单输入推理 ===")
    with torch.no_grad():
        output_tensor_single = model(img_tensor)
    print(f"单输入输出tensor形状: {output_tensor_single.shape}")
    
    print("\n=== 双输入推理（模拟make_heatmap.py）===")
    output_tensor_double = inference_with_double_input(model, img_tensor, device)
    print(f"双输入输出tensor形状: {output_tensor_double.shape}")

    # ===== 调试输出形状 =====
    print(f"原始图像尺寸: {original_size}")
    
    # ===== 保存输出图像 =====
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # 保存单输入结果
    print("\n=== 保存单输入结果 ===")
    save_tensor_as_image(output_tensor_single, output_image_path.replace('.png', '_single.png'), original_size)
    base_filename_single = output_image_path.replace('.png', '_single')
    save_individual_channels(output_tensor_single, base_filename_single, original_size, num_channels=3)
    
    # 保存双输入结果（模拟make_heatmap.py风格）
    print("\n=== 保存双输入结果（make_heatmap.py风格）===")
    save_tensor_as_image(output_tensor_double, output_image_path.replace('.png', '_double.png'), original_size)
    base_filename_double = output_image_path.replace('.png', '_double')
    save_individual_channels(output_tensor_double, base_filename_double, original_size, num_channels=3)
    save_heatmap_style_output(output_tensor_double, base_filename_double, original_size)

    print(f"推理完成，单输入输出已保存到：{output_image_path.replace('.png', '_single.png')}")
    print(f"推理完成，双输入输出已保存到：{output_image_path.replace('.png', '_double.png')}")