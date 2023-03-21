# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import logging


def save_model(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def load_model(model, path):
    state_dict = torch.load(path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict, strict=False)       #

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum().item() / len(correct)
    return acc

def binary_acc_prompt(preds, labels):      #preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.argmax(preds, dim=1), labels.flatten()).float()      #eq里面的两个参数的shape=torch.Size([16])    
    acc = correct.sum().item() / len(correct)
    return acc


class kd_loss_fn(nn.Module):
    def __init__(self, num_classes, args):
        super(kd_loss_fn, self).__init__()
        self.num_classes = num_classes
        self.alpha = args.alpha
        self.T = args.temperature
        
    def forward(self, output_batch, labels_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # labels_batch  -> B, LongTensor
        # teacher_outputs -> B X num_classes
        
        # torch.save(output_batch, './output_batch')
        # torch.save(labels_batch,'./labels_batch')
        # torch.save(teacher_outputs,'./teacher_outputs')
    
        # zero-mean, and small value
        # teacher_outputs = (teacher_outputs - torch.mean(teacher_outputs, dim=1).view(-1,1))/100.0
        # output_batch = (output_batch - torch.mean(output_batch, dim=1).view(-1,1))/100.0
    
        teacher_outputs=F.softmax(teacher_outputs/self.T,dim=1)
        output_batch=F.log_softmax(output_batch/self.T,dim=1)    
    
        #CE_teacher = -torch.sum(torch.sum(torch.mul(teacher_outputs,output_batch)))/teacher_outputs.size(0)
        #CE_teacher.requires_grad_(True)
        KL_teacher = nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) * self.T
        CE_true = nn.CrossEntropyLoss()(output_batch, labels_batch) 
        loss = KL_teacher * self.alpha + CE_true * (1 - self.alpha)    
        return loss
 
class Att_Loss(nn.Module):
    def __init__(self, temperature = 1, loss = 'CE'):
        super(Att_Loss, self).__init__()        
        self.T = temperature
        self.loss = loss
    def forward(self, output_batch, labels_batch, attention):
        # output_batch  -> B X num_classes X num_student
        # attention     -> B X num_student X num_student
        # teacher_outputs -> B X num_classes
        
        batch_size, num_classes, num_student = output_batch.size()
        labels_batch = labels_batch.view(-1,1).repeat(1, num_student)  # B X num_student
        loss_true = nn.CrossEntropyLoss()(output_batch, labels_batch) * num_student
        # teacher_outputs = teacher_outputs.repeat(args.num_student, 1, 1).view(-1, num_classes, args.num_student) # B X num_classes X num_student    
        
        attention_label = torch.bmm(output_batch, attention.permute(0,2,1))     # B X num_classes X num_student
        
        if self.loss == 'CE':
            output_batch = F.log_softmax(output_batch/self.T, dim=1)
            attention_outputs = F.softmax(attention_label/self.T, dim=1)            # B X num_classes X num_student
            loss_att = -torch.sum(torch.mul(output_batch, attention_outputs))/batch_size
        elif self.loss == 'MSE':
            # calculate the average distance between attention and identity
            output_batch = F.softmax(output_batch, dim=1)    
            attention_outputs = F.softmax(attention_label, dim=1)            # B X num_classes X num_student
            loss_att = torch.sum((output_batch - attention_outputs) ** 2) / batch_size 
        # calculate the log angle 
        identity = torch.eye(num_student).reshape(1, num_student, num_student).repeat(batch_size, 1, 1).cuda()
        # calculate the average distance between attention and identity
        scale = torch.Tensor([batch_size * num_student]).sqrt().cuda()
        dist_att = torch.norm(attention - identity, p='fro')/scale
        # dist_p = torch.norm(output_batch, p='fro')
        # angle = torch.log(loss_att) - torch.log(dist) - torch.log(dist_p)
        # angle = loss_att/(dist * dist_p)
        return loss_true, loss_att, dist_att
        
class KL_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes            
        # teacher_outputs -> B X num_classes
        
        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)
        
        output_batch = F.log_softmax(output_batch/self.T, dim = 1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T, dim = 1) + 10**(-7)
        #teacher_outputs = teacher_outputs/self.T + 10**(-7)                    #投票后本身就已经满足softmax

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) 
        
        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss
        
class CE_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(CE_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T, dim=1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T, dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, teacher_outputs))/teacher_outputs.size(0)
        return loss
        
class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        batch_size = output_batch.size(0)
        output_batch = F.softmax(output_batch, dim = 1)
        teacher_outputs = F.softmax(teacher_outputs, dim = 1)
        # Same result MSE-loss implementation torch.sum -> sum of all element
        loss = torch.sum((output_batch - teacher_outputs) ** 2) / batch_size 
        
        return loss

class E_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(E_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T,dim=1)    
        self_outputs = F.softmax(output_batch/self.T,dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, self_outputs))/output_batch.size(0)
        
        return loss
