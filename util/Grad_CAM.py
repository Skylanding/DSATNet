#%%
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    def __init__(self, model, target_layer, target_class):
        self.model = model
        self.gradients = None
        self.features = None
        self.target_layer = target_layer
        self.target_class = target_class

        self.hook_forward = self.target_layer.register_forward_hook(self.forward_a)
        self.hook_backward = self.target_layer.register_full_backward_hook(self.backward_b)
    
    def forward_a(self, module, input, output):
        self.features = output   #这个地方容易错，一定要小心， 还有就是关掉应用的时候with torch.no_grad():
    
    def backward_b(self,module, grad_input, grad_output):
        self.gradients = grad_output[0]   #这个地方容易错，一定要小心， 还有就是，

    def get_activation_map_weights(self, gradients):
        return torch.mean(gradients, dim=[2, 3], keepdim=True)
    
    def remove_hook(self):
        self.hook_forward.remove()
        self.hook_backward.remove()

    def generate_heatmap(self, x):
        self.model.eval()
        self.model.zero_grad()
        
        pred = self.model(x)

        target_one_hot = torch.zeros_like(pred)
        target_one_hot[:, self.target_class]=1
            
        pred = pred.requires_grad_(True)

        pred.backward(target_one_hot)

        activation_weights = self.get_activation_map_weights(self.gradients)
        weighted_activation = F.relu(activation_weights * self.features)
        heatmap = torch.mean(weighted_activation, dim=1, keepdim=True)

        return heatmap

    def resize_heatmap(self, heatmap, new_height, new_width):
        return F.interpolate(heatmap, size=(new_height, new_width), mode='bilinear', align_corners=False)

    def prepare_heatmap_for_overlay(self, heatmap):
        heatmap = (heatmap - heatmap.max()) / (heatmap.max() - heatmap.min())  #应该减去最大值
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)

        return heatmap

    def prepare_image_for_overlay(self, image):
        image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        image = (image - image.max()) / (image.max() - image.min())  #应该减去最大值
        image = (image * 255).astype("uint8")

        return image

    def overlay_image_and_heatmap(self, image, heatmap, alpha=0.4, color_model= '2'):
        if color_model == '1':  #保存蓝色区域
            return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        
        if color_model == '2':  #保存只显示红色区域
            blue_mask = (heatmap[:, :, 0] >= 80) & (heatmap[:, :, 0] <= 200)#& (heatmap[:, :, 2] >= 0)
            heatmap[blue_mask,0] = 0   ###竟然还能这么写

            return cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        

    def one_input(self, x):#[1, c, n, w]
        
        heatmap = self.generate_heatmap(x)
        
        heatmap = self.resize_heatmap(heatmap, new_height=x.shape[2], new_width=x.shape[3])

        heatmap = self.prepare_heatmap_for_overlay(heatmap)

        heatmap_color = heatmap #jin 会倍overlay_image_and_heatmap影像
        image = self.prepare_image_for_overlay(x)
        output = self.overlay_image_and_heatmap(image, heatmap)
        heat_image = output
        return heatmap_color, heat_image
    
    def __call__(self, inputs):
        
        stacked_heatmaps = []
        stacked_heat_images = []
        for input_i in inputs:
            heatmap_i, heat_image_i = self.one_input(input_i.squeeze(0))
            stacked_heatmaps.append(heatmap_i)      
            stacked_heat_images.append(heat_image_i)  
        
        stacked_heatmaps = np.stack(stacked_heatmaps, axis=0)
        stacked_heat_images = np.stack(stacked_heat_images, axis=0)
        return stacked_heatmaps, stacked_heat_images
    
     

# import argparse
# from models.seg_model import Seg_Detection
# from util.transforms import train_transforms,test_transforms , val_transforms
# from glob import glob
# from collections import OrderedDict
# import os
# torch.cuda.device_count()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parser = argparse.ArgumentParser('Seg Detection train')
# parser.add_argument("--backbone", type=str, default="swinv2_128")
# parser.add_argument("--neck", type=str, default="fpn+fuse+drop")
# parser.add_argument("--head", type=str, default="fcn")
# parser.add_argument("--loss_function", type=str, default="bce+dice")
# parser.add_argument("--pretrain", type=str,
#                     default='')  # 预训练权重路径
# parser.add_argument("--input_size", type=int, default=256)

# parser.add_argument("--num_workers", type=int, default=1)
# parser.add_argument("--batch_size", type=int, default=2)
# parser.add_argument("--learning_rate", type=int, default=0.0035)
# parser.add_argument("--epochs", type=int, default=1000)

# opt = parser.parse_args(args=[])


# input_image = torch.randn(1, 3, 256, 256)  # 假设输入图像为224x224的RGB图像
# target_class = 1  # 假设目标类别为第5类

# model=Seg_Detection(opt)
# # print(model)
# target_layer = model.head.classify # 假设要可视化的目标层是模型的features部分
# #%%
# grad_cam = GradCAM(model, target_layer, target_class=1 )
# output = grad_cam(input_image)
# cv2.imwrite("jin.png", output)


