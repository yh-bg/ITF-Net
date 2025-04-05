import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from dataset import FlameSet,DataPrefetcher
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import cv2
from ITFNet import ITFNet


def main():
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
        #生成随机数中包括1和10000 在需要生成随机数据的实验中，每次实验都需要生成数据。   随机初始化参数
        # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    opt.seed = random.randint(1, 10000)  #
    torch.manual_seed(opt.seed)  # 为CPU设置种子数，生成随机数
    # if cuda:
    #     torch.cuda.manual_seed(opt.seed)  # 为GPU设置种子数，生成随机数
    #
    # cudnn.benchmark = True
    #
    # print('======>Load datasets')
    # train_set = FlameSet(opt.dataset_train)
    # val_set = FlameSet(opt.dataset_val)
    # train_data_loader=DataLoader(dataset=train_set,batch_size=opt.batchSize, shuffle=True)  #shuffle 打乱数据集
    # val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize, shuffle=True)
    # count=0
    # for idx, (img, label,target,filename) in enumerate(train_data_loader):
    #     count+=1
    # print(count)
    #
    # print("===> Building model")
    # model = MFUnet(num_channels = 4)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        # 设ice to "0" or "1" or ... accordingly
        device = torch.device(opt.gpu_id)  # Or whichever device ID you want to use
        with torch.cuda.device(device):  # Within this context all tensors produced will reside on the chosen device
            cudnn.benchmark = True

            print('======>Load datasets')
            train_set = FlameSet(opt.dataset_train)
            val_set = FlameSet(opt.dataset_val)
            train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
            val_data_loader = DataLoader(dataset=val_set, batch_size=opt.batchSize, shuffle=True)
            count = 0
            for idx, (img, label, target, filename) in enumerate(train_data_loader):
                # Ensure that tensors get loaded on the chosen GPU by adding .to(device)
                img = img.to(device)
                label = label.to(device)
                target = target.to(device)
                count += 1
            print(count)
            print("===> Building model")
            model = ITFNet(num_channels=4)
            # criterion = nn.MSELoss()
            criterion = nn.L1Loss()
            min_loss = 100.
            # model = model.cuda(opt.gpu_id)
            # criterion = criterion.cuda(opt.gpu_id)
            # print("===> Setting GPU Cuda: {}".format(opt.gpu_id))
            model = model.to(device)
            criterion = criterion.to(device)
            print("===> Setting GPU Cuda: {}".format(opt.gpu_id))
            print("===> Setting Optimizer and lr_scheduler")
            optimizer = optim.Adam(model.parameters(), lr=opt.lr)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1000],gamma=0.1)
            print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            # optionally resume from a checkpoint
            if opt.resume:
                if os.path.isfile(opt.resume):
                    print("===> loading checkpoint '{}'".format(opt.resume))
                    checkpoint = torch.load(opt.resume)
                    opt.start_epoch = checkpoint["epoch"] + 1
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict((checkpoint["optimizer_state_dict"]))
                else:
                    print("===> no checkpoint found at '{}'".format(opt.resume))

            # optionally copy weights from a checkpoint
            if opt.pretrained:
                if os.path.isfile(opt.pretrained):
                    print("===> loading model '{}'".format(opt.pretrained))
                    weights = torch.load(opt.pretrained)
                    model.load_state_dict(weights['model'].state_dict())
                else:
                    print("===> no model found at '{}'".format(opt.pretrained))

            print("===> Training")
            t = time.strftime("%Y%m%d%H%M")  # 年月日小时分钟
            if not os.path.exists('log'):
                os.mkdir('log')
            global log
            log = 'log/' + t + '_batchSize%d' % (opt.batchSize) + '_lr%f' % (opt.lr) + '_SSdatasetz_c'
            if not os.path.exists(log):
                os.mkdir(log)
            backup_model = os.path.join(log, opt.backup)
            if not os.path.exists(backup_model):
                os.mkdir(backup_model)  # 备份pkl文件
            best_model = os.path.join(log, opt.best_model)
            if not os.path.exists(best_model):
                os.mkdir(best_model)  # 存储最好结果pkl
            train_image = os.path.join(log, opt.train_path)
            if not os.path.exists(train_image):
                os.mkdir(train_image)  # 存储训练的输出图像
            val_image = os.path.join(log, opt.val_path)
            if not os.path.exists(val_image):
                os.mkdir(val_image)  # 存储训练的输出图像
            original_image = os.path.join(log, opt.ori_path)
            if not os.path.exists(original_image):
                os.mkdir(original_image)  # 存储训练的输出图像

            train(train_data_loader, val_data_loader, optimizer, model, criterion, t, backup_model, best_model,
                  min_loss, train_image, val_image, original_image)





def train(train_data_loader,val_data_loader, optimizer, model, criterion,t,backup_model,best_model,min_loss,train_image,val_image,original_image):
    print('===> Begin Training!')
    model.train()
    steps_per_epoch = len(train_data_loader)
    if opt.train_log:
        if os.path.isfile(opt.train_log):
            train_log = open(opt.train_log,"a")  # log/Networkname_202111022144_train.log
        else:
            print("=> no train_log found at '{}'".format(opt.train_log))
    else:
            train_log = open(os.path.join(log, "%s_%s_train.log") % (opt.net, t),"w")  # log/Networkname_202111022144_train.log

    if opt.epoch_time_log:
        if os.path.isfile(opt.epoch_time_log):
            epoch_time_log = open(opt.epoch_time_log,"a")
        else:
            print("==> no epoch_time_log found at '{}'".format(opt.epoch_time_log))
    else:
        epoch_time_log = open(os.path.join(log, "%s_%s_epoch_time.log") % (opt.net, t), "w")
    time_sum = 0

    global val_log
    if opt.val_log:
        if os.path.isfile(opt.val_log):
            val_log = open(opt.val_log, "a")
        else:
            print("===> no val_log found at '{}'".format(opt.val_log))
    else:
        val_log = open(os.path.join(log, "%s_%s_val.log") % (opt.net, t), "w")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # 计时,每个epoch的时间
        start = time.time()

        # 加快读取数据集
        prefetcher = DataPrefetcher(train_data_loader)
        data = prefetcher.next()

        for i in range(opt.batchSize):
            input_pan, input_ms, target = data[0][i:i + 1, :, :, :], data[1][i:i + 1, :, :, :], data[2][i:i + 1, :, :, :]
            if opt.cuda:
                input_pan = input_pan.cuda(opt.gpu_id)
                input_ms = input_ms.cuda(opt.gpu_id)
                target = target.cuda(opt.gpu_id)
            output = model(input_pan, input_ms)
            # save_ms_pan(input_ms,input_pan,epoch,original_image)
            if epoch % 100 == 0:#训练过程特征图输出
                mid_feature = output[0, :, :, :]
                train_gt = target[0, :, :, :]
                save_mid_feature(train_image,mid_feature,train_gt,epoch)
            train_loss = criterion(output, target)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # i = 0
        # while data is not None:
        #     # run_step()
        #     i += 1
        #     if i >= steps_per_epoch:
        #         break  # 循环次数从0开始计数还是从1开始计数的问题
        #     input_pan, input_ms, target = data[0], data[1], data[2]
        #     if opt.cuda:
        #         input_pan = input_pan.cuda()
        #         input_ms = input_ms.cuda()
        #         target = target.cuda()
        #     output = model(input_pan, input_ms)
        #     if epoch % 100 == 0:  # 训练过程特征图输出
        #         mid_feature = output[0, :, :, :]
        #         train_gt = target[0, :, :, :]
        #         save_mid_feature(train_image, mid_feature, train_gt, epoch)
        #     train_loss = criterion(output, target)
        #     optimizer.zero_grad()
        #     train_loss.backward()
        #     optimizer.step()

        print("===> Epoch[{}/{}]: Lr:{} Train_Loss: {:.10f}".format(epoch, opt.nEpochs, optimizer.param_groups[0]["lr"],
                                                                    train_loss.item()))
        train_log.write("Epoch[{}/{}]: Train_Loss: {:.15f}\n".format(epoch, opt.nEpochs, train_loss.item()))

        # backup a model every ** epochs and validate
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, optimizer, backup_model)
            checkpoint = torch.load(os.path.join(backup_model, "model_epoch_{}.pth".format(epoch)))
            model.load_state_dict(checkpoint['model_state_dict'])
            # model.load_state_dict(checkpoint['model'])
            print('===> Validating the model after training {} epochs'.format(epoch))
            val_retloss = val(val_data_loader, model, criterion, epoch, val_image)
            val_log.write("{} {:.10f}\n".format((epoch), val_retloss))
            # Save the best weight
            if min_loss > val_retloss:
                save_bestmodel(model, epoch, best_model)
                min_loss = val_retloss

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch
        epoch_time_log.write("No:{} epoch training costs {:.4f}min\n".format(epoch, time_epoch / 60))

    train_log.close()
    val_log.close()
    epoch_time_log.close()

def val(val_data_loader, model, criterion, epoch,val_image):
    model.eval()
    avg_l1 = 0
    val_loss=[]

    with torch.no_grad():
        for k, data in enumerate(val_data_loader):
            input_pan, input_ms, target = data[0], data[1],data[2]
            if opt.cuda:
                input_pan = input_pan.cuda(opt.gpu_id)
                input_ms = input_ms.cuda(opt.gpu_id)
                target = target.cuda(opt.gpu_id)

            output = model(input_pan, input_ms)
            val_imgshow = output[0, :, :, :]
            val_gt = target[0, :, :, :]
            save_mid_feature(val_image, val_imgshow,val_gt, epoch)
            loss = criterion(output, target)
            avg_l1 += loss.item()

    del (input_pan, input_ms,target, output)
    print("===> Epoch{} Avg. Val Loss: {:.10f}".format(epoch, avg_l1 / len(val_data_loader)))#ssim_value
    val_loss.append(avg_l1)
    return avg_l1 / len(val_data_loader)


def save_checkpoint(model, epoch,optimizer,backup_model):
    model_out_path = os.path.join(backup_model,"model_epoch_{}.pth".format(epoch))
    state = {"epoch": epoch, "model_state_dict": model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}
    torch.save(state, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def save_bestmodel(model, epoch,backupbest_model):
    model_out_path = os.path.join(backupbest_model,"model_best_epoch.pth")
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)
    print("Save best model at {} epoch!!!".format(epoch))

def save_mid_feature(train_image,mid_feature,gt,epoch):
    imgf = np.array(mid_feature.squeeze().detach().cpu())  # (WV-2,256,256)
    imgf = np.clip(imgf,-1,1)
    img_gt = np.array(gt.squeeze().detach().cpu())  # (WV-2,256,256)
    imgf_gt = np.clip(img_gt, -1, 1)
    two_img = np.concatenate((imgf,imgf_gt),axis=2)
    image = ((two_img * 0.5 + 0.5) * 255).astype('uint8')
    image = cv2.merge(image)
    cv2.imwrite(os.path.join(train_image, '%d_output.png' % (epoch)), image)  # 读取ms文件名


# def save_ms_pan(input_ms, input_pan, epoch,original_image):
#     # 假设你想将多光谱图像和全色图像保存为文件或进行其他处理
#     # 这里只是一个示例，具体实现取决于你的需求
#
#     # 将多光谱图像和全色图像转换成可保存的格式（例如，numpy数组、PIL图像等）
#     ms_array = input_ms.squeeze().detach().cpu().numpy()
#     pan_array = input_pan.squeeze().detach().cpu().numpy()
#
#     # 进行剪裁和归一化操作
#     ms_array = np.clip(ms_array, -1, 1)
#     pan_array = np.clip(pan_array, -1, 1)
#
#     # Scale and convert the images to uint8
#     scaled_ms = ((ms_array * 0.5 + 0.5) * 255).astype('uint8')
#     scaled_pan = ((pan_array * 0.5 + 0.5) * 255).astype('uint8')
#     scaled_ms = np.transpose(scaled_ms, (1, 2, 0))
#     cv2.imwrite(os.path.join(original_image, '%d_ms.png' % (epoch)), scaled_ms)
#     cv2.imwrite(os.path.join(original_image, '%d_pan.png' % (epoch)), scaled_pan)



if __name__ == "__main__":
    main()
    os.system("shutdown")



