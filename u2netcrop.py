from u2net import crop as u2net
import torch
import os

model = u2net.U2NET(3,1)
model.load_state_dict(torch.load("u2net/u2net.pth", map_location=torch.device('cpu')))
model.eval()

fname = "sample-img"

u2net.crop_img(fname, model)
