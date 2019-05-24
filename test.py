# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import TestDataset
from model import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
from graphModule import GraphConv

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, args):
    (img, info) = data

    # prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)


def test(segmentation_module, loader, args):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        
        with torch.no_grad():
            scores = torch.zeros(1, 150, segSize[0], segSize[1])
            # scores = async_copy_to(scores, args.gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                print(img)
                del feed_dict['img_ori']
                del feed_dict['info']
                # feed_dict = async_copy_to(feed_dict, args.gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict)
                scores = scores + pred_tmp / len(args.imgSize)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred, args)

        pbar.update(1)


def main(args):
    # torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    
    enc_out = torch.randn(([1,2048,64,64]))
    net_encoder = builder.build_encoder(
        weights="baseline-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth")
    gcu = GraphConv(X=enc_out)#, V=2), GCU(X=enc_out, V=4), GCU(X=enc_out, V=8),GCU(X=enc_out, V=32)]

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, gcu, crit, tr=False)

    # print("Prinitng Params", gcu[1].parameters())
    for m in gcu.parameters():
        print("Hello",m.shape,m.name,m)
    print("dddddddddddddddd", len(list(gcu.parameters())))
    for m in gcu.modules():
        print("Prining", m.parameters())
    # Dataset and Loader
    if len(args.test_imgs) == 1 and os.path.isdir(args.test_imgs[0]):
        test_imgs = find_recursive(args.test_imgs[0])

    else:
        test_imgs = args.test_imgs


    list_test = [{'fpath_img': x} for x in test_imgs]
    
    dataset_test = TestDataset(
        list_test, args, max_sample=-1)


    loader_test = torchdata.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)


    # Main loop
    test(segmentation_module, loader_test, args)

    print('Inference done!')




if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--test_imgs', required=True, nargs='+', type=str,
                        help='a list of image paths, or a directory name')
    parser.add_argument('--imgSize', default=[505],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    args = parser.parse_args()

    main(args)

"""
->work around for doubt 2 that I mailed put z =1 and both denominator and numerator are 0

->In middle of writing code for training 
    files modifies model.py and test.py
->need to check how to send in batches

"""