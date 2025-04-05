import argparse

from ITFNet import ITFNet
from dataset import FlameSet
from torch.utils.data import DataLoader
from dataset import FlameSet,DataPrefetcher
import torch
from torch.autograd import Variable
import numpy as np
import os
import cv2
from PIL import Image


parser=argparse.ArgumentParser(description='')
parser.add_argument('--dataset_test', type=str, default='../datasets/Maryland/test')
parser.add_argument('--checkpoint', type=str,default='model_best_epoch.pth')
parser.add_argument("--net", type=str, default='ITFNet')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--gpu_id', default=3, type=int,help='GPU ID to use')
opt = parser.parse_args()

test_set = FlameSet(opt.dataset_test)
test_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)


bestmodel = torch.load('log/Maryland_2/bestmodel_dir/%s'%opt.checkpoint)

image_path = 'log/Maryland_2/test_image'

def test(test_data_loader, model):
    with torch.no_grad():
        model.eval()
        for i,(pan,ms,gt,filename) in enumerate(test_data_loader):
            if opt.cuda:
                input_pan = pan.cuda(opt.gpu_id)
                input_ms = ms.cuda(opt.gpu_id)
                output = model(input_pan, input_ms)
                imgf = output.squeeze().cpu().numpy()  # (WV-2_1,256,256)
                imgf = np.clip(imgf,-1,1)
                #compare with gt image
                img_gt = gt.squeeze().cpu().numpy()  # (WV-2_1,256,256)
                imgf_gt = np.clip(img_gt, -1, 1)
                two_img = np.concatenate((imgf, imgf_gt), axis=2)
                two_img = ((two_img * 0.5 + 0.5) * 255).astype('uint8')
                #finish compare
                only_test_image = ((imgf * 0.5 + 0.5) * 255).astype('uint8')
                image = cv2.merge(only_test_image)
                print(filename)
                cv2.imwrite(os.path.join(image_path, '%s' % (filename)), image)  # 读取ms文件名

if not os.path.exists(image_path):
    os.makedirs(image_path)

test(test_data_loader, bestmodel['model'])

#模型的保存与加载
#1.通过torch.save()来保存模型的state_dict模型参数，并通过load_state_dict()来加载并恢复模型参数
# PATH = 'xxxx.pth'
# torch.save(model.state_dict(), PATH)
#
# model = ModelName()  # 首先通过代码获取模型结构
# model.load_state_dict(torch.load(PATH))  # 然后加载模型的state_dict
# model.eval()
#
# 2.保存和加载整个模型
# torch.save(model, PATH)
# model = torch.load(PATH)
# model.eval()


# def test(test_data_loader, model):
#     with torch.no_grad():
#         model.eval()
#         for i,(pan,ms,gt,filename) in enumerate(test_data_loader):
#             if opt.cuda:
#                 filename = filename[0].split('\\')[-1]
#                 input_pan = pan.cuda()
#                 input_ms = ms.cuda()
#                 output = model(input_pan, input_ms)
#                 imgf = output.squeeze().cpu().numpy()  # (3,256,256)
#                 imgf = np.clip(imgf,-1,1)
#                 #compare with gt image
#                 img_gt = gt.squeeze().cpu().numpy()  # (3,256,256)
#                 imgf_gt = np.clip(img_gt, -1, 1)
#                 two_img = np.concatenate((imgf, imgf_gt), axis=2)
#                 two_img = ((two_img * 0.5 + 0.5) * 255).astype('uint8')
#                 #finish compare
#                 only_test_image = ((imgf * 0.5 + 0.5) * 255).astype('uint8')
#                 image = cv2.merge(only_test_image)
#                 print(filename)
#                 cv2.imwrite(os.path.join(image_path, '%s' % (filename)), image)  # 读取ms文件名
#
# if not os.path.exists(image_path):
#     os.makedirs(image_path)
# model_test = ITFNet(num_channels = 4)
# model_test.cuda()
# PATH = 'log/Maryland_2/backup_dir/model_epoch_7200.pth'
# checkpoint = torch.load(PATH)
# model_test.load_state_dict(checkpoint['model_state_dict'],strict=False)
# test(test_data_loader, model_test)
