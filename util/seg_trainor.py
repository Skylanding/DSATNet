"""jinjin 定义一个训练代码插件"""
import os
from tqdm import tqdm
import torch
import torch.utils.data

from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob
# from thop import profile
import warnings
warnings.filterwarnings("ignore")
import logging
from abc import ABC, abstractmethod
import cv2, json
from .seg_testor import test_Testor, train_Testor
from pprint import pprint


class train_trainor():
    """jin   仅仅用于训练时候训练器"""
    def __init__(self, model, train_loader, valid_loader, criterion, train_running_metrics, device, save_dir = "train_results"):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.train_running_metrics = train_running_metrics
        self.valid_loader = valid_loader
        self.epoch_loss = pd.DataFrame(columns=["Epoch", "Loss"])

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)


    def train_1epoch(self, optimizer, scheduler, epoch):
        self.train_running_metrics.reset()
        
        total_loss = 0.0
        self.model.train()
        self.model.to(self.device)
        trainbar = tqdm(self.train_loader, desc=f"Trian_ {self.model.__class__.__name__}")
        for i, results in enumerate(trainbar):

            batch_img, labels, *_   = results  #jinjin 不确定返回几个值，但是确定前两个是我要的
            batch_img = batch_img.float().to(self.device)
            labels = labels.long().to(self.device)
            
            # if i ==3:
            #     break
            
            # print(batch_img.size(), labels.size())

            optimizer.zero_grad()
            
            cd_preds= self.model(batch_img)
            
            from torch.nn.modules.loss import _Loss


            if isinstance(self.criterion, _Loss):
                # 如果是 torch.nn.modules.loss._Loss 的子类，使用 one-hot 值计算
                # 具体实现根据你的需求进行调整
                
                true_1_hot = torch.eye(2).to(self.device)[labels.squeeze(1)]
                true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
                cd_loss = self.criterion(cd_preds, true_1_hot)
                # print("Using one-hot values for calculation.")
            else:
                # 如果不是 torch.nn.modules.loss._Loss 的子类，使用原值计算
                # 具体实现根据你的需求进行调整
                # print("Using original values for calculation.")
                cd_loss = self.criterion(cd_preds, labels, self.device)

            
            loss = cd_loss
            
            loss.backward()
            
            optimizer.step()
    
            _, cd_preds = torch.max(cd_preds, 1)
            total_loss += loss.item()
            # Calculate and log other batch metrics
            self.train_running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())
            
            del batch_img, labels
        average_loss = total_loss/(len(self.train_loader))
        self.epoch_loss = self.epoch_loss.append({"Epoch": epoch, "Loss": average_loss}, ignore_index=True)
            
    def train_epoches(self, optimizer, scheduler, valid_running_metrics, epoches):
        
        
        best_metrics = {'precision_1': -1, 'recall_1': -1, 'F1_1': -1, 'Overall_Acc': -1,'Mean_IoU': -1}
        
        validor = train_Testor(self.model, self.valid_loader, valid_running_metrics, self.device)

        for epoch in range(epoches):
            metadata = {}
            self.train_1epoch(optimizer, scheduler, epoch)
            
            validor.test_segmentation()
            scheduler.step()
            
            train_score = self.train_running_metrics.get_scores()
            valid_score = validor.RunningMetrics.get_scores()
            
            
            print("训练结果")
            pprint(train_score, indent=4, width=1)
            print("验证结果")
            pprint(valid_score, indent=4, width=1)
            print(f"完成训练 epoch: {epoch+1}/{epoches}")
            
            metadata['validation_metrics'] = valid_score
            metadata['train_metrics'] = train_score
            

    
            if valid_score['F1_1'] > best_metrics['F1_1']:
                best_metrics['F1_1'] =  valid_score['F1_1']
                os.makedirs(self.save_dir, exist_ok=True)
                
                def convert_numpy_to_list(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
        
                with open(self.save_dir+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                    json.dump(metadata, fout, default=convert_numpy_to_list)
                torch.save(self.model, self.save_dir+'/checkpoint_epoch_'+str(epoch)+'.pt')
        
        self.epoch_loss.to_csv(os.path.join(self.save_dir, f"{self.model.__class__.__name__}train_epoch_loss.csv"), 
                               index=False)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.epoch_loss['Epoch'], self.epoch_loss['Loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{self.model.__class__.__name__}_train_epoch_loss_curve.png"))
        plt.show()
        
                    

