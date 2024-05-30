import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from util.parser import get_parser_with_args
from util.helpers import (get_criterion,
                           initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
import json
import pandas as pd
from util.AverageMeter import AverageMeter, RunningMetrics
from tqdm import tqdm
import random
import numpy as np
import ml_collections
from torch.utils.data import DataLoader, Subset
from models.block.Drop import dropblock_step
from util.common import check_dirs, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
import argparse
from models.seg_model import Seg_Detection
from util.transforms import train_transforms,test_transforms , val_transforms
from glob import glob
from collections import OrderedDict

from sklearn.model_selection import KFold
import gc

# torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser('Seg Detection train')
parser.add_argument("--backbone", type=str, default="swinv2_128")
parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
parser.add_argument("--head", type=str, default="fcn")
parser.add_argument("--loss_function", type=str, default="hybrid")
parser.add_argument("--pretrain", type=str,
                    default='')  # 预训练权重路径
parser.add_argument("--input_size", type=int, default=256)

parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=int, default=0.0035)
parser.add_argument("--epochs", type=int, default=1000)

opt = parser.parse_args(args=[])
_, metadata = get_parser_with_args()



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=123)

from dataset import CrackData
print('===> Loading datasets')
train_path = "/mnt/Disk1/liyu/Work1/dataset/train"
val_path = "/mnt/Disk1/liyu/Work1/dataset/valid"
test_path = "/mnt/Disk1/liyu/Work1/dataset/test"


### 数据集
# train_path = "/mnt/Disk1/liyu/Work1/dataset/train"
# val_path = "/mnt/Disk1/liyu/Work1/dataset/valid"
# test_path = "/mnt/Disk1/liyu/Work1/dataset/test"

train_data = pd.DataFrame({'images': sorted(glob(os.path.join(train_path, "img") + "/*/*.bmp")),
              'masks': sorted(glob(os.path.join(train_path, "gt") + "/*/*.bmp"))})                                                                                                                                                                                                                                                                                        

val_data = pd.DataFrame({'images': sorted(glob(os.path.join(val_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(val_path, "gt") + "/*.bmp"))})

test_data = pd.DataFrame({'images': sorted(glob(os.path.join(test_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(test_path, "gt") + "/*.bmp"))})


combined_data = pd.concat([train_data, test_data]).reset_index(drop=True)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 释放模型、优化器和调度器
def clear_memory():
    if 'model' in globals():
        del model
    if 'optimizer' in globals():
        del optimizer
    if 'scheduler' in globals():
        del scheduler
    torch.cuda.empty_cache()
    gc.collect()



for fold, (train_ids, test_ids) in enumerate(kfold.split(combined_data)):
    print(f"Training Fold {fold}")
    
    # train_dataset = CrackData(df = train_data,transforms=train_transforms)
    # val_dataset = CrackData(df = val_data,transforms=val_transforms)
    # test_dataset = CrackData(df = test_data,transforms=test_transforms)

    # 创建交叉验证的训练和测试子集
    train_subset = combined_data.iloc[train_ids]
    test_subset = combined_data.iloc[test_ids]

    train_dataset = CrackData(df=train_subset, transforms=train_transforms)
    test_dataset = CrackData(df=test_subset, transforms=test_transforms)
    val_dataset = CrackData(df = val_data,transforms=val_transforms)
    print(len(train_dataset), len(val_dataset), len(test_dataset))


    train_loader = DataLoader(dataset=train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, num_workers=opt.num_workers, batch_size=1, shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, num_workers=opt.num_workers, batch_size=1, shuffle=False)

    print(train_loader)


    print('===> Building model')
    save_path = check_dirs(ori_dirs="./runs/BUSI_train")

    # vars() 函数返回对象object的属性和属性值的字典对象。
    with open(save_path+'/1_opt.txt', 'a') as f:
        for arg in vars(opt):
            f.write('{0}: {1} \n'.format(arg, getattr(opt, arg)))  # getattr() 函数是获取args中arg的属性值
        f.close()

    """
    Load Model then define other aspects of the model
    """
    logging.info('LOADING Model')
    model=Seg_Detection(opt).to(device)
    


    print("load weight~~~~~~~~~~~~~~~~~~~~~~~~~")
    criterion = get_criterion(opt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate ,weight_decay=0.001) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    """
    Set starting values
    """

    best_metrics = {'precision_1': -1, 'recall_1': -1, 'F1_1': -1, 'Overall_Acc': -1,'Mean_IoU': -1}

    logging.info('STARTING training')
    total_step = -1


    print('---------- Networks initialized -------------')
    scale = ScaleInOutput(opt.input_size)

    for epoch in range(opt.epochs):
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()

        """
        Begin Training
        """
        model.train()
        logging.info('SET model mode to train!')
        batch_iter = 0
        tbar = tqdm(train_loader, position=0, ncols=100)
        i=1
        train_running_metrics =  RunningMetrics(2)
        for batch_img, labels in tbar:
            tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
            batch_iter = batch_iter+opt.batch_size
            total_step += 1
            # Set variables for training
            batch_img= batch_img.float().to(device)
            # print(batch_img.size())


            labels = labels.long().to(device)


            # Zero the gradient
            optimizer.zero_grad()

            # Get model preditions, calculate loss, backprop
            batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))   # 指定某个尺度进行训练

            cd_preds= model(batch_img)

            cd_preds= scale.scale_output(cd_preds)
        

            cd_loss = criterion(cd_preds, labels,device)


            loss = cd_loss
            loss.backward()
            optimizer.step()

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                        (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                        (labels.size()[0] * (opt.input_size**2)))
            train_running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())


            # clear batch variables from memory
            del batch_img, labels


        scheduler.step()
        dropblock_step(model)
        mean_train_metrics = train_running_metrics.get_scores()
        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """
        Begin Validation
        """
        model.eval()
        val_running_metrics =  RunningMetrics(2)
        with torch.no_grad():
            
            vbar = tqdm(val_loader, position=0, ncols=100)
            for batch_img ,labels in vbar:
                # Set variables for training
                vbar.set_description("epoch {} info val test".format(epoch))
                batch_img = batch_img.float().to(device)
                labels = labels.long().to(device)

                batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))   # 指定某个尺度进行训练

                cd_preds = model(batch_img)
                cd_preds = scale.scale_output(cd_preds)


                cd_loss = criterion(cd_preds, labels,device)


                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                # Calculate and log other batch metrics
                cd_corrects = (100 *
                            (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                            (labels.size()[0] * (opt.input_size**2)))

                val_running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())

                # clear batch variables from memory
                del batch_img, labels
                torch.cuda.empty_cache()


            mean_val_metrics = val_running_metrics.get_scores()
            logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

            # if ((mean_val_metrics['precision_1'] > best_metrics['precision_1'])
            #      or
            #      (mean_val_metrics['recall_1'] > best_metrics['recall_1'])
            #      or
            #      (mean_val_metrics['F1_1'] > best_metrics['F1_1'])):
            if mean_val_metrics['F1_1'] > best_metrics['F1_1']:



                # Insert training and epoch information to metadata dictionary
                logging.info('update the model')
                metadata['validation_metrics'] = mean_val_metrics

                # Save model and log
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                with open(save_path+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(metadata, fout)

                torch.save(model, save_path+'/checkpoint_epoch_'+str(epoch)+'.pt')

                best_metrics = mean_val_metrics


            print('An epoch finished.')
            gc.collect()
            torch.cuda.empty_cache()
            
    # 交叉验证折训练和验证结束后
    clear_memory()
    
    print('Done!')