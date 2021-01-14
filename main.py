from matting.scripts.demo import inference
from torch.utils.data import DataLoader
from SOD.data_loader import DatasetVal
from SOD.classnet import ClassNet
from SOD.config import Config
from SOD.train import Trainer
import skimage.io as io
from tqdm import tqdm
import numpy as np
import torch
import time
import cv2
import os

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========================= get salient map ========================= #
    val_dataset = DatasetVal()
    # print(len(val_dataset))
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
        with tqdm(total=len(file_names)) as pbar:
            for file_name in file_names:
                salient_map = io.imread('data/salmap/' + file_name)
                kernel_size = 3  # kernel size
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
                pbar.set_postfix(step='Transfrom Salmap to Trimap', image_name=file_name)
                pbar.update(1)
    torch.cuda.empty_cache()

    # ========================= image matting ========================= #
    # print("Step3.  image matting")
    image_names = os.listdir('data/image')
    trimap_names = os.listdir('data/trimap')
    i = 0
    with tqdm(total=len(image_names)) as pbar:
        for image_name, trimap_name in zip(image_names, trimap_names):
            i = i + 1
            image_path = 'data/image/' + image_name
            trimap_path = 'data/trimap/' + trimap_name

            pred = inference(image_path, trimap_path)
            cv2.imwrite('data/output/' + trimap_name, pred.astype(np.uint8))
            # print(f'  {i}/{len(image_names)}-----{image_name} is done!')
            pbar.set_postfix(step='Image Matting', image_name=image_name)
            pbar.update(1)
    torch.cuda.empty_cache()
