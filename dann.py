
import os
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import timm
import cv2
import time
import random
random.seed(42)

BF_DIR = '/mnt/projects/bhatta70/train/bf_domain/'
T20X_DIR = '/mnt/projects/bhatta70/train/20x-fewshot/'
T20X5H_DIR = '/mnt/projects/bhatta70/train/20x-5h-fewshot/'
AGAR_DIR = '/mnt/projects/bhatta70/train/agar-fewshot/'
DEFOCUS_DIR = '/mnt/projects/bhatta70/train/defocus-fewshot/'
BF_SAMPLES = {
    'BC': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'bc' in file.lower()],
    'BS': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'bs' in file.lower()],
    'EC': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'ec' in file.lower()],
    'LI': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'li' in file.lower()],
    'SE': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'se' in file.lower()],
    'ST': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'st' in file.lower()],
}

TWENTYX_SAMPLES = {
    'BC': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'bc' in file.lower()],
    'BS': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'bs' in file.lower()],
    'EC': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'ec' in file.lower()],
    'LI': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'li' in file.lower()],
    'SE': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'se' in file.lower()],
    'ST': [os.path.join(T20X_DIR,file) for file in os.listdir(T20X_DIR) if 'st' in file.lower()]
}
TWENTYX5H_SAMPLES = {
    'BC': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'bc' in file.lower()],
    'BS': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'bs' in file.lower()],
    'EC': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'ec' in file.lower()],
    'LI': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'li' in file.lower()],
    'SE': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'se' in file.lower()],
    'ST': [os.path.join(T20X5H_DIR,file) for file in os.listdir(T20X5H_DIR) if 'st' in file.lower()]
}
AGAR_SAMPLES = {
    'BC': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'bc' in file.lower()],
    'BS': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'bs' in file.lower()],
    'EC': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'ec' in file.lower()],
    'LI': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'li' in file.lower()],
    'SE': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'se' in file.lower()],
    'ST': [os.path.join(AGAR_DIR,file) for file in os.listdir(AGAR_DIR) if 'st' in file.lower()]
}
DEFOCUS_SAMPLES = {
    'BC': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'bc' in file.lower()],
    'BS': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'bs' in file.lower()],
    'EC': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'ec' in file.lower()],
    'LI': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'li' in file.lower()],
    'SE': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'se' in file.lower()],
    'ST': [os.path.join(DEFOCUS_DIR,file) for file in os.listdir(DEFOCUS_DIR) if 'st' in file.lower()]
}


DOMAIN_TARGETS = {
    'bf': BF_SAMPLES,
    '20x': TWENTYX_SAMPLES,
    '20x5h': TWENTYX5H_SAMPLES,
    'agar': AGAR_SAMPLES,
    'defocus': DEFOCUS_SAMPLES
}

def mmd_loss(source_features, target_features):
    def gaussian_kernel(x, y, sigma=1.0):
        return torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0)).pow(2), dim=-1) / (2 * sigma ** 2))

    xx = gaussian_kernel(source_features, source_features)
    yy = gaussian_kernel(target_features, target_features)
    xy = gaussian_kernel(source_features, target_features)

    return torch.mean(xx + yy - 2 * xy)

# define the model class
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
class DANN(pl.LightningModule):
    def __init__(self, num_classes=6, target_domains=['bf', '20x'], num_target_samples=5, mmd_weight=0, align_weight=1, cotrain_only=False):
        super(DANN, self).__init__()

        for target_domain in target_domains:
            assert target_domain in DOMAIN_TARGETS.keys(), f"Target domain must be one of {DOMAIN_TARGETS.keys()}"
        assert num_target_samples <= 5, "Number of target samples must be less than or equal to 5"
        
        self.mmd_weight = mmd_weight
        self.align_weight = align_weight
        self.num_classes = num_classes
        if cotrain_only:
            self.align_weight = 0
            self.mmd_weight = 0

        self.target_domains = {domain: {} for domain in target_domains}
        self.num_target_samples = num_target_samples
        # model_name = 'efficientnetv2_rw_m' 
        model_name = 'resnet152' 
        self.feature_extractor = timm.create_model(model_name, num_classes = num_classes, pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])  # remove the last layer
        self.feature_extractor.classifier = nn.Identity()


        for target_domain in self.target_domains.keys():
            self.target_domains[target_domain] = DOMAIN_TARGETS[target_domain]
            for classname in self.target_domains[target_domain].keys():
                class_samples = random.sample(self.target_domains[target_domain][classname], num_target_samples)
                self.target_domains[target_domain][classname] = class_samples 
         
        # task classifier (label predictor)
        feats_dim = 2048 if model_name.startswith('resnet') else 2152
        self.task_classifier = nn.Linear(feats_dim, num_classes)
        # domain classifier
        # self.domain_classifier= nn.Sequential(
        #     nn.Linear(2152, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, len(self.target_domains.keys())+1),
        # )
        self.domain_classifier= nn.Sequential(
            nn.Linear(feats_dim, 1024),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(1024, len(self.target_domains.keys())+1),
        )
    def load_few_shot_samples(self,classnames, labels):
        target_samples = []
        class_labels = []
        domain_labels = []
        for domain_idx,target_domain in enumerate(self.target_domains.keys()):
            for i,classname in enumerate(classnames):
                label = labels[i]
                if classname not in self.target_domains[target_domain].keys():
                    continue
                img_file = np.random.choice(self.target_domains[target_domain][classname])
                img = cv2.imread(img_file)
                try:
                    transform = self.trainer.datamodule.transforms

                    img = np.array(img, dtype=np.uint8)
                    # print("attempting transform for ", img_file)
                    sample = transform(img=img.copy())
                except:
                    sample = torch.zeros(3, 224, 224)
                target_samples.append( sample)
                class_labels.append(label)
                domain_labels.append(domain_idx+1)
        target_samples= torch.stack(target_samples).to(self.device)
        return target_samples, class_labels, domain_labels
    def forward(self, x, alpha):
        features = self.feature_extractor(x)

        # task prediction
        class_output = self.task_classifier(features)

        # domain prediction
        reverse_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        # print("Class output: ", class_output.shape)
        # print("Domain output: ", domain_output.shape)
        # print("Features: ", features.shape)
        return features, class_output, domain_output
    def training_step(self, batch, batch_idx):
        input = batch[0]
        labels = batch[1]
        
        if self.num_target_samples == 0:
            features, class_output, domain_output = self.forward(input, 1)
            task_loss = F.cross_entropy(class_output.float(), labels)
            self.log('train_loss', task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            total_loss=  task_loss 
            return total_loss
        ## -- Load target samples for each class in the batch -- ##
        # get classname from trainer data module
        data_module = self.trainer.datamodule.train_ds
        # Convert target indices to class names
        class_to_idx = data_module.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        # get classnames of targets
        sample_classnames = [idx_to_class[i.item()].upper() for i in labels]
        target_samples, target_labels, domain_labels = self.load_few_shot_samples(sample_classnames, labels)
        target_samples = target_samples.to(self.device)
        target_labels = torch.tensor(target_labels).to(self.device)
       
        domain_labels = torch.cat((torch.zeros(input.size(0)), torch.tensor(domain_labels)), 0).to(self.device).long()
        input = torch.cat((input, target_samples), 0)
        labels = torch.cat((labels, target_labels), 0)
        
        p = self.current_epoch / self.trainer.max_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        features, class_output, domain_output = self.forward(input, alpha)

        # compute the task loss using cross entropy loss function
        task_loss = F.cross_entropy(class_output.float(), labels)
        # domain_loss = F.binary_cross_entropy_with_logits(domain_output.flatten(), domain_labels) 
        domain_loss = F.cross_entropy(domain_output, domain_labels)

        # log the training loss
        self.log('train_loss', task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('domain_loss', domain_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('mmd_loss', mmd, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        total_loss=  task_loss + self.align_weight*domain_loss #+ self.mmd_weight*mmd
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        input = batch[0]
        labels = batch[1]
        _, class_output, _ = self.forward(input, 1)
        val_loss = F.cross_entropy(class_output.float(), labels)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    def configure_optimizers(self):
        # create an instance of the AdamW optimizer
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2) # default: lr=1e-3
        # sch = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3) # every epoch by default
        # return ({'optimizer': opt, 'lr_scheduler':sch})
        
        # create a learning rate scheduler that decreases the learning rate every 10c epochs by a factor of 0.3
        sch = {'scheduler': lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)}
        return [opt], [sch]
    def test_step(self, batch, batch_idx):
        input = batch[0]
        labels = batch[1]
        class_output, _ = self.forward(input, 1)
        test_loss = F.cross_entropy(class_output.float(), labels)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss
    def visualize_tsne(self, features):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features.cpu().detach().numpy())
        return tsne_results


if __name__=='__main__':
    dann = DANN( num_target_samples=2)

    target_samples, target_labels, domain_labels = dann.load_few_shot_samples(['BC','BS','EC','LI','SE', 'ST'], [0,1,2,3,4,5])
    print("Target samples: ", target_samples.shape)
    print("Target labels: ", target_labels)
    print("Domain labels: ", domain_labels)
