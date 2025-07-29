import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import cv2
import time
import random
random.seed(42)

torch.autograd.set_detect_anomaly(True)

BF_DIR = '/mnt/projects/bhatta70/train/bf_domain/'
k=5
BF_SAMPLES = {
    'BC': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'Bc' in file],
    'BS': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'Bs' in file],
    'EC': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'Ec' in file],
    'LI': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'Li' in file],
    'SE': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'SE' in file],
    'ST': [os.path.join(BF_DIR,file) for file in os.listdir(BF_DIR) if 'ST' in file]
}

def mmd_loss(source_features, target_features):
    def gaussian_kernel(x, y, sigma=1.0):
        return torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0)).pow(2), dim=-1) / (2 * sigma ** 2))

    xx = gaussian_kernel(source_features, source_features)
    yy = gaussian_kernel(target_features, target_features)
    xy = gaussian_kernel(source_features, target_features)
    return torch.mean(xx + yy - 2 * xy)
class ProtoTypeDANN(pl.LightningModule):
    def __init__(self, num_classes=6, feature_dim=128, mmd_weight=0.1, align_weight=0.01, temperature=0.5, val_domain='source'):
        super(ProtoTypeDANN, self).__init__()
        self.feature_extractor = timm.create_model('efficientnetv2_rw_m', num_classes=feature_dim, pretrained=True)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.mmd_weight = mmd_weight
        self.align_weight = align_weight
        self.temperature = temperature
        self.val_domain = val_domain
        
        # Initialize source and target prototypes
        self.register_buffer('source_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('target_prototypes', torch.zeros(num_classes, feature_dim))
        self.register_buffer('source_counts', torch.zeros(num_classes))
        self.register_buffer('target_counts', torch.zeros(num_classes))

    def forward(self, x):
        return self.feature_extractor(x)

    def compute_prototypes(self, features, labels, prototypes, counts):
        for c in range(self.num_classes):
            class_features = features[labels == c]
            if len(class_features) > 0:
                class_prototype = class_features.mean(0)
                prototypes[c] = (prototypes[c] * counts[c] + class_prototype) / (counts[c] + 1)
                counts[c]  = counts[c] + 1
        return prototypes, counts

    def training_step(self, batch, batch_idx):
        source_inputs, source_labels = batch[0], batch[1]
        data_module = self.trainer.datamodule.train_ds
            # Convert target indices to class names
        class_to_idx = data_module.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        # get classnames of targets
        sample_classnames = [idx_to_class[i.item()].upper() for i in source_labels]
        target_inputs = self.load_brightfield_samples(sample_classnames)
        # copy target tensor
        target_labels = source_labels.clone()
        
        # Extract features
        source_features = self.forward(source_inputs)
        target_features = self.forward(target_inputs)
        
        # Compute and update prototypes
        with torch.no_grad():
            self.source_prototypes, self.source_counts = self.compute_prototypes(
                source_features, source_labels, self.source_prototypes, self.source_counts)
            self.target_prototypes, self.target_counts = self.compute_prototypes(
                target_features, target_labels, self.target_prototypes, self.target_counts)
        
        # Compute distances to prototypes
        source_dists = torch.cdist(source_features, self.source_prototypes)
        target_dists = torch.cdist(target_features, self.target_prototypes)
        
        source_logits = -1*source_dists / self.temperature
        target_logits = -1*target_dists / self.temperature
        
        # Compute losses
        source_loss = F.cross_entropy(source_logits, source_labels)
        target_loss = F.cross_entropy(target_logits, target_labels)
        task_loss = source_loss + target_loss
        
        mmd = mmd_loss(self.source_prototypes, self.target_prototypes)
        
        # Prototype alignment loss
        align_loss = F.mse_loss(self.source_prototypes, self.target_prototypes)
        total_loss = task_loss + self.mmd_weight * mmd + self.align_weight * align_loss
        self.log('train_source_loss', source_loss)
        self.log('train_target_loss', target_loss)
        self.log('train_mmd_loss', mmd)
        self.log('train_align_loss', align_loss)
        self.log('train_total_loss', total_loss)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        features = self.forward(inputs)
        if self.val_domain == 'source':
            dists = torch.cdist(features, self.source_prototypes)
        else:
            dists = torch.cdist(features, self.target_prototypes)  # Use target prototypes for validation
        logits = -dists / self.temperature
        loss = F.cross_entropy(logits, labels) 
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('val_acc', acc)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
        return [optimizer], [scheduler]
    def load_brightfield_samples(self,classnames):
        samples = []
        for classname in classnames:
            img_file = np.random.choice(BF_SAMPLES[classname])
            img = cv2.imread(img_file)
            #transform = self.trainer.datamodule.val_transforms
            transform = self.trainer.datamodule.transforms
            sample = transform(img=img)
            samples.append(sample)

        samples= torch.stack(samples).to(self.device)
        return samples