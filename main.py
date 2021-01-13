from matting.core.deploy import inference_img_whole
from matting.core.model_loader import my_torch_load
from torch.utils.data import DataLoader
from SOD.data_loader import DatasetVal
from SOD.classnet import ClassNet
from SOD.config import Config
from SOD.train import Trainer
from matting.core import net
import skimage.io as io
import numpy as np
import argparse
import torch
import cv2
import os

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========================= get salient map ========================= #
    val_dataset = DatasetVal()
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8)

    sod_net = ClassNet().to(device)  # SOD model
    cfg = Config()

    trainer = Trainer('inference', sod_net, cfg, device)
    trainer.load_weights('model/sod_model.pth')  # load SOD model
    trainer.train(val_loader, val_dataset)  # predict the salient map

    # make the cache of gpu empty after getting salient map because of the independence of SOD and Matting
    torch.cuda.empty_cache()

    # ========================= salient=>trimap ========================= #
    for _, _, file_names in os.walk("data/salmap"):
        for file_name in file_names:
            salient_map = io.imread('data/salmap/' + file_name)
            kernel_size = 5  # kernel size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # dilate operation
            dilate = cv2.dilate(salient_map, kernel, iterations=15)
            dilate[dilate < 128] = 0
            dilate[dilate >= 128] = 128

            # erode operation
            erode = cv2.erode(salient_map, kernel, iterations=15)
            erode[erode < 128] = 0
            erode[erode >= 128] = 127

            # get trimap and save it
            trimap = dilate + erode
            cv2.imwrite("data/trimap/" + file_name, trimap)
    torch.cuda.empty_cache()

    # ========================= image matting ========================= #
    # parameters setting
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cuda = True
    args.resume = "model/matting_model.pth"
    args.stage = 1
    args.crop_or_resize = "whole"
    args.max_size = 1600

    model = net.VGG16(args)
    ckpt = my_torch_load(args.resume)

    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.cuda()
    image_names = os.listdir('data/image')
    trimap_names = os.listdir('data/trimap')
    print(image_names)
    for image_name, trimap_name in zip(image_names, trimap_names):
        image_path = 'data/image/' + image_name
        trimap_path = 'data/trimap/' + trimap_name

        image = cv2.imread(image_path)
        print(image_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_mattes = inference_img_whole(args, model, image, trimap)

        pred_mattes = (pred_mattes * 255).astype(np.uint8)
        pred_mattes[trimap == 255] = 255
        pred_mattes[trimap == 0] = 0

        cv2.imwrite('data/output/' + trimap_name, pred_mattes)