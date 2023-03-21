import os
import numpy as np
import random
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import utils
import datasets
from data import FewshotDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import logging
from model import ElectraForPrompt

class Trainer:
    # set all parameters for train_and_eval
    def set_paras(self, config):
        # parameters about template, label description words, data format and metric
        self.task_name = config["task_name"]
        self.task_type = config.get('task_type', 'classfication')
        if self.task_type == 'regression':
            self.Vl, self.Vu = config["data_format"]['Vl_Vu']
        self.template = config["template"]

        self.class_index = config["template"]["class_index"]
        self.pred_class_index = sorted(self.class_index, key=lambda s: self.class_index[s])
        self.data_path = config["data_path"]
        self.data_seed = self.data_path.split('/')[-1]
        self.data_format = config["data_format"]
        self.metric = config['metric']

        # Loading from a file for preventing errors caused by network issues. 
        self.metric_fn = datasets.load_metric(f'metrics/{self.metric}')
    
        # general parameters
        self.seed = config["seed"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epsilon = config["epsilon"]
        self.num_epoch = config["num_epoch"]
        self.max_length = config["max_length"]
        self.batch_size = config["batch_size"]
        self.model_name = config["model_name"]

    # train and eval the model
    def train_and_eval(self, dev_as_test=False):
        # set seed 
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # create checkpoint and logger file
        checkpoint_path = os.path.join(os.path.join('checkpoint', self.task_name, self.data_seed))        
        os.makedirs(checkpoint_path, exist_ok=True)
        utils.set_logger(os.path.join(checkpoint_path, 'train.log'))

        # load dataset
        logging.info('create dataset')  
        regression_task = self.task_type == 'regression'
        train_dataset = FewshotDataset(data_path=self.data_path, 
                                        split='train', 
                                        template=self.template, 
                                        model_name=self.model_name, 
                                        max_seq_length=self.max_length,
                                        data_format=self.data_format,
                                        regression=regression_task)
        dev_dataset = FewshotDataset(data_path=self.data_path, 
                                        split='dev', 
                                        template=self.template, 
                                        model_name=self.model_name, 
                                        max_seq_length=self.max_length,
                                        data_format=self.data_format,
                                        regression=regression_task)
        if dev_as_test: # for grid search phase
            test_dataset = dev_dataset
        else:
            test_dataset = FewshotDataset(data_path=self.data_path, 
                                        split='test', 
                                        template=self.template, 
                                        model_name=self.model_name, 
                                        max_seq_length=self.max_length,
                                        data_format=self.data_format,
                                        regression=regression_task)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # create model
        logging.info('create model')
        model = ElectraForPrompt(self.model_name)
        model = model.cuda()

        # train
        num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
        logging.info('Total params: %.2fM' % num_params)
        params = {}
        for n, p in model.named_parameters():
            params[n] = p                       
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.learning_rate, eps = self.epsilon)
        total_steps = len(train_loader) * self.num_epoch # [number of batches] x [number of NUM_EPOCHS].
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        logging.info('start training')
        max_val_acc = 0
        for epoch in range(self.num_epoch):
            loss = self.train(model, optimizer, scheduler, train_loader)
            val_metric = self.evaluate(model, dev_loader)
            
            if epoch % 5 == 0: # print every 5 epoches
                logging.info(f'{epoch+1}/{self.num_epoch}, train loss:{loss:.3f}, eval {self.metric}:{val_metric:.3f}')
            if epoch > 0 and val_metric > max_val_acc:
                max_val_acc = val_metric
                utils.save_model(model, os.path.join(checkpoint_path, 'best.pt'))

        # evaluate
        utils.load_model(model, os.path.join(checkpoint_path, 'best.pt'))
        return self.evaluate(model, test_loader)
        
    # train the model
    def train(self, model, optimizer, scheduler, loader):  
        model.train()

        avg_loss = []
        # y_pred, y_true = [], []

        for batch in loader:
            loss, class_prob = model(batch)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 处理出预测类别 (batch, class)
            # pred_class = [self.pred_class_index[i.item()] for i in torch.argmax(class_prob, -1)] # 求得哪个类的orignal概率大
            # label_class = batch['sentence_label']
            # y_pred += pred_class
            # y_true += label_class
            avg_loss.append(loss.item())
        
        # self.metric_fn.compute(references=y_true, predictions=y_pred)
        return sum(avg_loss)/len(avg_loss)
    
    # evaluate the model
    def evaluate(self, model, loader):
        model.eval()        

        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in loader:
                _, class_prob = model(batch)
        
                if self.task_type == 'classfication':
                    # prediction result： which label description word is the most original
                    pred_class = [self.pred_class_index[i.item()] for i in torch.argmax(class_prob, -1)]
                    label_class = batch['sentence_label'] # real label, str
                    y_pred += pred_class
                    y_true += label_class
                elif self.task_type == 'regression':
                    # get the prediction probability of two category
                    yu_prob = class_prob[:, self.pred_class_index.index('Yu')]
                    yl_prob = class_prob[:, self.pred_class_index.index('Yl')]

                    # normalize
                    yl_prob = yl_prob / (yl_prob + yu_prob)
                    yu_prob = yu_prob / (yl_prob + yu_prob)

                    # compute the final prediction value
                    pred_value = yl_prob * self.Vl + yu_prob * self.Vu
                    y_pred += pred_value.tolist()

                    # real label
                    label_value = [float(i) for i in batch['sentence_label']]
                    y_true += label_value

            task_metric = self.compute_metric(y_pred, y_true)
        
        return task_metric[self.metric]
         
    # compute metric
    def compute_metric(self, y_pred, y_true):
        if self.task_type == 'classfication':
            y_true = [self.class_index[i] for i in y_true] # str -> id
            y_pred = [self.class_index[i] for i in y_pred]
            return self.metric_fn.compute(references=y_true, predictions=y_pred)

        elif self.task_type == 'regression':
            return self.metric_fn.compute(references=y_true, predictions=y_pred)
