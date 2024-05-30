import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_JS(SR,GT):
    # JS : Jaccard similarity
    intersection = GT * SR
    JS = (float(intersection.sum())) / float((GT.sum() + SR.sum()-intersection.sum())+0.00001)
   
    return JS

def mutil_IOU(SR,GT): 
    
    num,_,_,num_classes=GT.size()
    
    mean_IoU=0
    for n in range(num):
        tmp_total_IOU=0
        for i in range(num_classes):
            tmp_total_IOU+=get_JS(SR[n,:,:,i],GT[n,:,:,i])
        tmp_mean_IOU=tmp_total_IOU/num_classes 
        mean_IoU+=tmp_mean_IOU
    return mean_IoU/num
    
    
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def dice_loss(logits, true, device,eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(device)[true.squeeze(1).to(device)].to(device)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    
    true_1_hot=true_1_hot.to(device)
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = ( 2*intersection / ((cardinality ) + eps)).mean()
    return ( 1 - dice_loss )


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-7, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.eps = eps

    def forward(self, logits, true,device):
        """Computes the Tversky loss [1].
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
            
        true_1_hot=true_1_hot.to(device)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()
        return (1 - tversky_loss)


def active_contour_loss( y_pred,y_true, weight=1):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    # length term
    device = y_pred.device
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term
    c_in = torch.ones_like(y_pred).to(device)
    c_out = torch.zeros_like(y_pred).to(device)
    if (y_true.size()[1]==1) & (y_true.ndim ==4):
        y_true = torch.eye(y_pred.size()[1])[y_true.squeeze(1)]
        y_true = y_true.permute(0, 3, 1, 2).float().to(device)
    if  y_true.ndim ==3:
        y_true = torch.eye(y_pred.size()[1])[y_true]
        y_true = y_true.permute(0, 3, 1, 2).float().to(device)

    region_in = torch.abs(torch.mean(y_pred * ((y_true - c_in) ** 2)))  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.abs(torch.mean((1 - y_pred) * ((y_true - c_out) ** 2)))

    region = region_in + region_out

    loss = weight * lenth + region

    return loss