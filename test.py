import os
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import math
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

from models.Omni_freq_deblur_arch import Omni_freq_deblur_arch

DATAROOT_LQ = '/home/zqk/data/GS-Blur-Split/test/input_noise'
DATAROOT_GT = '/home/zqk/data/GS-Blur-Split/test/target'
TEST_MODEL = '/home/zqk/Omni_freq_deblur_arch/GSblur.pth'
NUM_WORKERS = 4
MODEL_WIDTH = 48


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.C1 = (0.01 * 1.0) ** 2
        self.C2 = (0.03 * 1.0) ** 2
        
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, sigma)
        _2D_window = torch.outer(_1D_window, _1D_window)
        _3D_window = _2D_window.unsqueeze(0) * _1D_window.view(window_size, 1, 1)
        
        self.kernel = _3D_window.float().unsqueeze(0).unsqueeze(0)
        
        self.conv = nn.Conv3d(1, 1, (window_size, window_size, window_size), 
                                stride=1, padding=(window_size//2, window_size//2, window_size//2), 
                                bias=False, padding_mode='replicate')
        
        self.conv.weight.data = self.kernel
        self.conv.weight.requires_grad = False

    def forward(self, img1, img2):
        img1 = img1.permute(0, 2, 3, 1).unsqueeze(1)
        img2 = img2.permute(0, 2, 3, 1).unsqueeze(1)

        mu1 = self.conv(img1)
        mu2 = self.conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv(img1 ** 2) - mu1_sq
        sigma2_sq = self.conv(img2 ** 2) - mu2_sq
        sigma12 = self.conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return ssim_map.mean()

def psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class HardcodedDeblurDataset(Dataset):
    def __init__(self, lq_dir, gt_dir, transform=None):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        
        if not os.path.exists(self.lq_dir) or not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Specified paths not found. Please check: \nLQ: {self.lq_dir}\nGT: {self.gt_dir}")
            
        self.image_list = sorted(os.listdir(self.lq_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            input_noise_path = os.path.join(self.lq_dir, self.image_list[idx])
            target_path = os.path.join(self.gt_dir, self.image_list[idx])
            
            image = Image.open(input_noise_path).convert('RGB')
            label = Image.open(target_path).convert('RGB')
            
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            name = self.image_list[idx]
            return image, label, name
        except Exception as e:
            print(f"Error loading {self.image_list[idx]}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256), "error"

def main():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please launch this script using 'torchrun' to enable DDP mode.")
        
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    try:
        model = Omni_freq_deblur_arch(width=MODEL_WIDTH)
    except TypeError:
        if local_rank == 0: print("Warning: Omni_freq_deblur_arch does not accept 'width'. Using default.")
        model = Omni_freq_deblur_arch()

    model.to(device)

    if local_rank == 0:
        print(f"Loading weights from: {TEST_MODEL}")
        
    checkpoint = torch.load(TEST_MODEL, map_location=device)
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    dataset = HardcodedDeblurDataset(DATAROOT_LQ, DATAROOT_GT)
    sampler = DistributedSampler(dataset, shuffle=False)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=sampler, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    ssim_calc = SSIM(window_size=11).to(device)

    local_psnr = 0.0
    local_ssim = 0.0
    local_count = 0

    if local_rank == 0:
        iterator = tqdm(dataloader, desc="DDP Testing", unit="img")
    else:
        iterator = dataloader

    with torch.no_grad():
        for input_img, label_img, name in iterator:
            if name[0] == "error": continue 

            input_img = input_img.to(device)
            label_img = label_img.to(device)

            _, _, h, w = input_img.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                input_img = F_torch.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')

            pred = model(input_img)

            if pad_h > 0 or pad_w > 0:
                pred = pred[:, :, :h, :w]

            pred = torch.clamp(pred, 0, 1)
            
            pred = pred + 0.5 / 255.0
            pred = torch.clamp(pred, 0, 1)

            local_psnr += psnr_torch(pred, label_img).item()
            local_ssim += ssim_calc(pred, label_img).item()
            local_count += 1

    metrics_tensor = torch.tensor([local_psnr, local_ssim, local_count], dtype=torch.float64, device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        global_psnr_sum = metrics_tensor[0].item()
        global_ssim_sum = metrics_tensor[1].item()
        global_total_count = int(metrics_tensor[2].item())

        if global_total_count > 0:
            avg_psnr = global_psnr_sum / global_total_count
            avg_ssim = global_ssim_sum / global_total_count
            
            print("\n" + "="*40)
            print(f" DDP Multi-GPU Testing Completed")
            print(f" Test Path (LQ): {DATAROOT_LQ}")
            print(f" Test Path (GT): {DATAROOT_GT}")
            print(f" Total Images: {global_total_count}")
            print(f" Average PSNR: {avg_psnr:.4f} dB")
            print(f" Average SSIM: {avg_ssim:.4f}")
            print("="*40 + "\n")
        else:
            print("No valid images detected.")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
