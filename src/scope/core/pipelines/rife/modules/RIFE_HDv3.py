# Modified from https://github.com/hzwer/Practical-RIFE
# The original repo is: https://github.com/hzwer/Practical-RIFE

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from .IFNet_HDv3 import IFNet
from .loss import EPE, SOBEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, local_rank=-1):
        super().__init__()
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self, mode: bool = True):
        self.flownet.train(mode)
        return super().train(mode)

    def eval(self):
        self.flownet.eval()
        return super().eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {k.replace("module.", ""): v for k, v in param.items()}
            else:
                return param

        if rank > 0:
            return

        raw = (
            torch.load(f"{path}/flownet.pkl")
            if torch.cuda.is_available()
            else torch.load(f"{path}/flownet.pkl", map_location="cpu")
        )
        state_dict = convert(raw)

        # Filter out unexpected keys (teacher.*, caltime.*, etc.) to avoid strict load errors
        model_state = self.flownet.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state}
        unexpected = [k for k in state_dict.keys() if k not in model_state]
        if unexpected:
            print(f"[RIFE] Ignoring {len(unexpected)} unexpected keys: {unexpected[:3]}...")

        self.flownet.load_state_dict(filtered, strict=False)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), f"{path}/flownet.pkl")

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep=timestep, scale_list=scale_list)
        return merged[3]

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(
            torch.cat((imgs, gt), 1), timestep=0.5, scale_list=scale, training=training
        )
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3] * 0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[3], {
            "mask": mask,
            "flow": flow[3][:, :2],
            "loss_l1": loss_l1,
            "loss_smooth": loss_smooth,
        }
