import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import IEA_Dataset
from util import EDMLoss,AverageMeter
import option
from torch.autograd import Variable
import cv2
from pytorch_msssim import ssim
from wavelet_network import Wavelet_Net


npy_file_path='F:/IEA/label/exposure_condition'

opt = option.init()
opt.device = torch.device("cuda:0")


def save_tensor_as_image(tensor, filename):
    numpy_image = tensor.permute(1, 2, 0).detach().cpu().numpy()

    cv2.imwrite(filename, np.uint8(numpy_image/np.max(numpy_image)*255))


def adjust_learning_rate(params, optimizer, epoch):
    lr = params.init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_data_part(opt):
    train_csv_path =  'train_256.csv'
    val_csv_path = 'valid_256_2ev.csv'
    test_csv_path = 'valid_256_2ev.csv'

    train_ds = IEA_Dataset(train_csv_path, opt.path_to_images, if_train = True,npy_file_path=npy_file_path)
    val_ds = IEA_Dataset(val_csv_path, opt.path_to_images, if_train = False,npy_file_path=npy_file_path)
    test_ds = IEA_Dataset(test_csv_path, opt.path_to_images, if_train=False,npy_file_path=npy_file_path)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader,test_loader


def train(opt,model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()

    for idx, (x, y,npy_map,target_img) in enumerate(tqdm(loader)):
        x =Variable(x, requires_grad=True).to(opt.device)
        y = y.to(opt.device)
        npy_map = torch.unsqueeze(npy_map, dim=1)
        npy_map=npy_map.to(opt.device)
        target_img=target_img.to(opt.device)

        npy_map_pred=model(x)

        loss=criterion(npy_map,npy_map_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))
    print("train loss:", train_losses.avg)

    return train_losses.avg


def validate(opt,model, loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    validate_losses2 = AverageMeter()
    true_score = []
    pred_score = []
    Ssim = 0
    flag = 1

    grey_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
    ])

    for idx, (x, y,npy_map,target_img) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        npy_map = torch.unsqueeze(npy_map, dim=1)
        npy_map=npy_map.to(opt.device)
        y = y.to(opt.device)
        target_img = target_img.to(opt.device)

        npy_map_pred = model(x)

        loss = criterion(npy_map, npy_map_pred)
        #loss2=criterion(npy_map, npy_map_pred)

        validate_losses.update(loss.item(), x.size(0))
        Ssim = Ssim + torch.mean(
            ssim(npy_map.detach().to("cpu"), npy_map_pred.detach().to("cpu"), data_range=1, size_average=False))
        flag = flag + 1


    print("val loss(target-pred):",validate_losses.avg)
    Ssim = Ssim / flag
    print("ssim:", Ssim.item())

    return validate_losses.avg,0,0,0





def start_train(opt):
    grey_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1)
        # transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = create_data_part(opt)

    model=Wavelet_Net()
    model_path = "wavelet_epoch_2_val_loss0.03022034629540784.pth"
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))


    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    criterion = torch.nn.L1Loss()
    model = model.to(opt.device)
    criterion.to(opt.device)

    writer = None

    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader) * e,
                          name=f"{opt.experiment_dir_name}_by_batch")
        val_loss,vacc,vsrcc, vlcc = validate(opt,model=model, loader=val_loader, criterion=criterion,
                            writer=writer, global_step=len(val_loader) * e,
                            name=f"{opt.experiment_dir_name}_by_batch")

        model_name = f"wavelet_epoch_{e}_val_loss{val_loss}.pth"
        torch.save(model.state_dict(), os.path.join(opt.experiment_dir_name, model_name))





if __name__ =="__main__":
    import warnings

    warnings.filterwarnings("ignore")

    start_train(opt)