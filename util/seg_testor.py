#%%

"""jinjin 定义一个测试代码插件"""
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
from thop import profile
import warnings
warnings.filterwarnings("ignore")
# import logging
from abc import ABC, abstractmethod
import cv2



class train_Testor():
    """jin   仅仅用于训练时候"""
    def __init__(self, model, data_loader, RunningMetrics, device, save_dir = False):
        self.model = model
        self.data_loader = data_loader
        self.RunningMetrics = RunningMetrics
        self.device = device
        
        self.save_dir = save_dir
        if save_dir is not False:
            #jin, 默认是不保存信息的，
            os.makedirs(self.save_dir, exist_ok=True)

    def test_segmentation(self):

        self.model.eval()
        self.model.to(self.device)
        tbar = tqdm(self.data_loader, desc=f"Test_ {self.model.__class__.__name__}")
        for i, results in enumerate(tbar):
            batch_img, labels, *_   = results  #jinjin 不确定返回几个值，但是确定前两个是我要的
           
            batch_img = batch_img.float().to(self.device)
            labels = labels.long().to(self.device)
            # print(batch_img.size(), labels.size())

            cd_preds= self.model(batch_img)
            num_classes = cd_preds.shape[1]
            _, cd_preds = torch.max(cd_preds, 1)
            
            self.RunningMetrics.update(cd_preds.data.cpu().numpy(), labels.data.cpu().numpy())

            
        flops, params = profile(self.model, inputs=(batch_img[0].unsqueeze(0),))
        
        print("model_name",self.model.__class__.__name__)
        print("类别：", num_classes)
        print('flops :%.3f'%(flops/1024**3),'G')				# 打印计算量
        print('params:%.3f'%(params/1024**2), 'MB')				# 打印参数量
        score = self.RunningMetrics.get_scores()
        from  pprint import pprint
        # pprint(score)
        return flops, params, score
    
    
    




def pred_gt_result_image(gt_mask, pred_mask):
    """jin gt_mask, pred_mask 图像上只有0和255两个值"""

    overlap = cv2.bitwise_and(gt_mask, pred_mask)
    false_positive = cv2.subtract(pred_mask, overlap)
    false_negative = cv2.subtract(gt_mask, overlap)
    # print(gt_mask.shape)
    result_image = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Set overlapping regions to white
    result_image[overlap > 0] = [255, 255, 255]  # White
    result_image[false_positive > 0] = [0, 0, 255]  # Red
    result_image[false_negative > 0] = [0, 255, 0]  # Blue
    return result_image

def pred_gt_contour(gt_mask, pred_mask):
    """jin gt_mask, pred_mask 图像上只有0和255两个值"""

    
    contour_preds, _ = cv2.findContours( pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_gts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contour_preds)==1:
        # mask1 对应的是pred，这里有个筛选
        contour_preds = contour_preds
    if len(contour_preds)>=2:
        """去除轮廓面小于200的  jinjin, 可能需要，也可能不需要"""
        contour_preds = [contour for contour in contour_preds if (cv2.contourArea(contour) != 0)&(cv2.contourArea(contour) >= 200)]

    contour_gts = [contour for contour in contour_gts if (cv2.contourArea(contour) != 0)&(cv2.contourArea(contour) >= 200)]
    return contour_preds,  contour_gts

def generate_gt_pred_image(gt_mask, pred_mask, image, pred_color=(0, 255, 0), gt_color=(255, 0, 0), line_c=10):
    result_image = pred_gt_result_image(gt_mask, pred_mask)
    contour_preds,  contour_gts = pred_gt_contour(gt_mask, pred_mask)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制预测轮廓
    cv2.drawContours(image_bgr, contour_preds, -1, pred_color, line_c)
    # 绘制真实轮廓
    cv2.drawContours(image_bgr, contour_gts, -1, gt_color, line_c)
    
    return result_image, image_bgr


from util.grad_cam import GradCAM



class test_Testor():
    """jin   仅仅用于测试的时候， 不仅生成各种结果, 还能保存图片"""
    def __init__(self, model, data_loader, RunningMetrics, contourSimilarityCalculator, 
                 device, dataset_images_df, save_dir = False, save_ori_shape= True,
                 target_layer=None):
        self.model = model
        self.data_loader = data_loader
        self.RunningMetrics = RunningMetrics
        self.contourSimilarityCalculator = contourSimilarityCalculator
        self.device = device
        self.dataset_images_df = dataset_images_df
        self.save_ori_shape = save_ori_shape
        
        self.target_layer = target_layer
        self.save_dir = save_dir
        if self.save_dir is not False:
            #jin, 默认是不保存信息的，
            os.makedirs(self.save_dir, exist_ok=True)

    def test_segmentation(self):

        self.model.eval()
        self.model.to(self.device)
        tbar = tqdm(self.data_loader, desc=f"Test_ {self.model.__class__.__name__}")
        for i, results in enumerate(tbar):
            batch_img, labels, ws, hs, *_   = results  #jinjin 不确定返回几个值，但是确定前两个是我要的
            batch_img = batch_img.float().to(self.device)
            
            labels = labels.long().to(self.device)
            # print(batch_img.size(), labels.size())

            cd_preds= self.model(batch_img)
            num_classes = cd_preds.shape[1]
            _, cd_preds = torch.max(cd_preds, 1)
            
            self.RunningMetrics.update(cd_preds.data.cpu().numpy(), labels.data.cpu().numpy()) #(1, 256, 256), (1, 256, 256) b, h,w
            self.contourSimilarityCalculator.update(cd_preds.data.cpu().numpy(), labels.data.cpu().numpy(), img_sizes=(ws, hs))   #jin计算轮廓值，支持batchsize计算
            if self.target_layer is not None:
                for class_i in range(1, num_classes):
                    cam = GradCAM(self.model, self.target_layer, target_class=class_i)
                    heatmap_colors, heat_images = cam(batch_img)
                    if self.save_dir is not False:
                        _, _, _, _, image_paths, gt_paths = results
                        for image_path, heatmap_color, heat_image in zip(image_paths, heatmap_colors, heat_images):
                            
                            image_name = os.path.basename(image_path)
                            save_dir_tmp = os.path.join(self.save_dir,image_name.split(".")[0], str(class_i))
                            os.makedirs(save_dir_tmp, exist_ok=True)
                            
                            heatmap_path =  os.path.join(save_dir_tmp, image_name.split(".")[0]+"_heatmap.bmp")
                            heat_image_path = os.path.join(save_dir_tmp, image_name.split(".")[0]+"_heat_image.bmp")
                            
                            cv2.imwrite(heatmap_path, heatmap_color)
                            cv2.imwrite(heat_image_path, heat_image)
                            
                        
                    
                    
                    
            for pred, image, gt, w, h, image_path, gt_path in zip(cd_preds, *results):
                # print( image.shape, gt.shape, pred.shape, w, h, image_path, gt_path)
                pred, image, gt, w, h = pred.data.cpu().numpy(), image.data.cpu().numpy(), gt.data.cpu().numpy(), int(w), int(h)
                image= np.array(image).astype(np.uint8).transpose(( 1, 2, 0))
                for class_i in range(1, num_classes):
                    pred_i_binary, gt_i_binary = (pred==class_i), (gt==class_i)
                    pred_i_255, gt_i_255 = np.uint8( pred_i_binary*255), np.uint8(gt_i_binary*255)
                    
                    result_image, image_contour_bgr = generate_gt_pred_image(gt_i_255, pred_i_255, image, line_c=5)
                    
                    image_name = os.path.basename(image_path)

                    if self.save_dir is not False:
                        save_dir_tmp = os.path.join(self.save_dir,image_name.split(".")[0], str(class_i))
                        os.makedirs(save_dir_tmp, exist_ok=True)
                        gt_pred_image_path =  os.path.join(save_dir_tmp, image_name.split(".")[0]+"_gt_pred.bmp")
                        image_gt_contour_path = os.path.join(save_dir_tmp, image_name.split(".")[0]+"_gt_pred_contour.bmp")
                        gt_path = os.path.join(save_dir_tmp, image_name.split(".")[0]+"_gt.bmp")
                        pred_path = os.path.join(save_dir_tmp, image_name.split(".")[0]+"_pred.bmp")
                        image_path = os.path.join(save_dir_tmp, image_name)
                        if self.save_ori_shape is True:
                            cv2.imwrite(gt_path, cv2.resize(gt_i_255, (w,h)))
                            cv2.imwrite(pred_path, cv2.resize(pred_i_255, (w,h)))
                            cv2.imwrite(gt_pred_image_path, cv2.resize(result_image, (w,h)))
                            cv2.imwrite(image_path, cv2.resize(image, (w,h)))
                            cv2.imwrite(image_gt_contour_path, cv2.resize(image_contour_bgr, (w,h)))
                        if self.save_ori_shape is False:
                            cv2.imwrite(gt_path, gt_i_255)
                            cv2.imwrite(pred_path, pred_i_255)
                            cv2.imwrite(gt_pred_image_path, result_image)
                            cv2.imwrite(image_path, image)
                            cv2.imwrite(image_gt_contour_path, image_contour_bgr)
        
        #jin 计算相应的轮廓相似性矩阵            
        test_df_imagenames = pd.DataFrame()
        test_df_imagenames["images_names"] = self.dataset_images_df["images"].apply(lambda x: os.path.basename(x))
        
        all_info_df = pd.concat([test_df_imagenames, self.contourSimilarityCalculator.all_df.reset_index()], axis=1)
        
        #jin 计算mean iou, dice
        all_info_df["mean_iou"] = all_info_df.apply(lambda x: np.array([x[str(i)+"_iou"] for i in range(num_classes)]).mean(), axis=1)

        self.all_info_df = all_info_df
        flops, params = profile(self.model, inputs=(batch_img[0].unsqueeze(0),))
        import time
        start_time = time.time()
        cd_preds= self.model(batch_img[0].unsqueeze(0))
        densenet_time = time.time() - start_time
        
        print("model_name",self.model.__class__.__name__)
        print('flops :%.3f'%(flops/1024**3),'G')				# 打印计算量
        print('params:%.3f'%(params/1024**2), 'MB')				# 打印参数量
        print("per-image need:%.6f seconeds"%(densenet_time))
        score = self.RunningMetrics.get_scores()
        from  pprint import pprint
        pprint(score)
        return flops, params, score, all_info_df

        
        
        
        
        
# %%
