import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import MCAFNetPP_b
from dataset import DehazeDataset
from utils import psnr, save_samples
from losses import SobelLoss, FFTLoss


device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)


train_dataset = DehazeDataset(
    "datasets/train/hazy",
    "datasets/train/clear",
    patch_size=128
)

test_dataset = DehazeDataset(
    "datasets/test/hazy",
    "datasets/test/clear",
    patch_size=256
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4)


model = MCAFNetPP_b().to(device)

l1 = nn.L1Loss()
edge = SobelLoss()
fft = FFTLoss()

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)


for epoch in range(1,151):

    if epoch <= 40:
        train_dataset.set_patch(128)
    elif epoch <= 100:
        train_dataset.set_patch(192)
    else:
        train_dataset.set_patch(256)


    model.train()

    pbar = tqdm(train_loader)

    for hazy, clear in pbar:

        hazy = hazy.to(device)
        clear = clear.to(device)

        optimizer.zero_grad()

        out, *_ = model(hazy)

        loss = (
            l1(out, clear)
            + 0.1 * edge(out, clear)
            + 0.05 * fft(out, clear)
        )

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")


    scheduler.step()


    model.eval()

    total_psnr = 0

    with torch.no_grad():

        for i, (hazy, clear) in enumerate(test_loader):

            hazy = hazy.to(device)
            clear = clear.to(device)

            out, *_ = model(hazy)

            total_psnr += psnr(out, clear)

            if i == 0:
                save_samples(epoch, hazy, out, clear)


    avg_psnr = total_psnr / len(test_loader)

    print(f"\nEpoch {epoch} PSNR: {avg_psnr:.2f}")


    torch.save(
        model.state_dict(),
        f"checkpoints/mcafnetpp_epoch_{epoch}.pth"
    )