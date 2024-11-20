import os
from tqdm import tqdm
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
from wavelet_network import Wavelet_Net
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from matplotlib.colors import LinearSegmentedColormap

def get_img_name_from_dir(dir):
    folder_path = dir

    jpg_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

    return jpg_files

def read_image(image_path):
    image = Image.open(image_path).convert('RGB')

    image_tensor = to_tensor(image).unsqueeze(0)

    return image_tensor

def tensor_to_heatmap(tensor, output_path):

    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    tensor_np = tensor.cpu().numpy()

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  #
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)

    plt.figure(figsize=(6, 6))
    plt.imshow(tensor_np, cmap=cmap)
    plt.axis("off")

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


device = torch.device("cuda:0")

model=Wavelet_Net()
model_path = "wavelet_epoch_2_val_loss0.03022034629540784.pth"
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model = model.to(device)
model.eval()


IMAGE_NET_MEAN = [0., 0., 0.]
IMAGE_NET_STD = [1., 1., 1.]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            normalize])

# Place the image to be predicted in the img_pred folder
# the predicted heat map result will be saved as **_exposure.png
img_path = "img_to_pred/"


img_list = get_img_name_from_dir(img_path)
for img in tqdm(img_list):
    image = read_image(os.path.join(img_path, img))
    height, width = image.shape[2], image.shape[3]
    origin_trans = transforms.Resize((height, width))
    image = transform(image)
    image = torch.cat([image, image], dim=0)
    image = image.to(device)
    exposure = model(image)
    exposure = origin_trans(exposure[0])[0]
    exposure = exposure*-1
    tensor_to_heatmap(exposure.detach().cpu(), os.path.join(img_path, img[:-4]+"_exposure.png"))