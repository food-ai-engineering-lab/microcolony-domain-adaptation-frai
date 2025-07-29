import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff
from PIL import Image
import skimage.exposure as exposure
from glob import glob
import pytorch_lightning as pl

# Input image dimensions (match this with your dataset)
max_px = 1002
min_px = 1002

# reference images for domain adapt
# ref_images = ['/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Ec-60x-3h-BF-20220818-R1-011.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Pf-60x-3h-BF-20220822-R1-003.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/SE-60x-3h-BF-20220821-R2-005.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bc-60x-3h-BF-20220823-R3-003.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bs-60x-3h-BF-20220822-R3-001.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Li-60x-3h-BF-20220818-R1-003.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Lm-60x-3h-BF-20220819-R1-006.jpg',
#               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/ST-60x-3h-BF-20220821-R2-002.jpg'
# ]

# class Custom_Transform(A.ImageOnlyTransform):
#     def __init__(self, reference_images, **kwargs):
#         super().__init__(**kwargs)
#         self.reference_images = reference_images
#     def apply(self, img, **params):
#         reference_images = self.reference_images
#         aug = A.Compose([A.FDA(reference_images, beta_limit=.45, p=1)])
#         result = aug(image=img)
#         return A.InvertImg(p=1)(image=result['image'])['image']
#     def get_transform_init_args_names(self):
#         return ()
# Define a class for data augmentation
class Transforms:
    # Training set transforms include several augmentation techniques
    def __init__(self, train=True):
        if train:
            self.tf = A.Compose([
                A.Resize(min_px, min_px),
                # A.FDA(reference_images = ref_images, beta_limit=.35, p=.6),
                A.ToFloat(max_value=255),
                # A.InvertImg(p=0.5),
                #Custom_Transform(reference_images = ref_images, p=1),
                # A.ToGray(p=1.0),
                ## Model v1
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2,
                                           brightness_by_max=True,
                                           p=0.5),
                
                ## Model v2 only
                A.Transpose(p=0.5),
                A.Blur(blur_limit=5, p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
                                border_mode=4, value=None, mask_value=None, 
                                 p=0.5),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
                                    interpolation=1, border_mode=4, value=None, 
                                    mask_value=None, p=0.5),
                A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
                         shear=None, interpolation=1, mask_interpolation=0, cval=0, 
                         cval_mask=0, mode=0, fit_output=False,  p=0.5),
                A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                              mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),

                # A.Defocus(radius=(3,10), alias_blur=(.1,.5), p=0.5),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
                # A.ZoomBlur(max_factor=1.31, step_factor=(0.001, 0.003), p=0.5),
                # A.GaussNoise(var_limit=(0.1, 2.0), mean=0, per_channel=True, p=0.1),
                # A.ElasticTransform(alpha=0.05, sigma=0.005, alpha_affine=0.005, p=0.1),
                # A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=2, p=0.1),
                # A.CoarseDropout(max_holes=1, max_height=1, max_width=1, min_holes=None, min_height=None, min_width=None, fill_value=0, p=0.1),

                ## Model v2a only
                # A.Transpose(p=0.6),
                # A.Blur(blur_limit=3, p=0.4),
                # # A.MedianBlur(blur_limit=3, p=0.4),
                # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.4),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
                #                 border_mode=4, value=None, mask_value=None, 
                #                 normalized=False, p=0.5),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
                #                     interpolation=1, border_mode=4, value=None, 
                #                     mask_value=None, p=0.4),
                # A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
                #          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
                #          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
                # A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
                #               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
                # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    
                ## Model v3 only
                # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                # A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.5),
                # A.ZoomBlur(max_factor=1.31, step_factor=(0.01, 0.03), p=0.5),
                # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                #                num_shadows_upper=2, shadow_dimension=5, p=0.5),
                # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
                # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
                # A.RandomScale(scale_limit=(-0.7,0), interpolation=1, p=0.5), #scaling factor range=(0.3,1) ~obj lens
                # A.PadIfNeeded(min_height=max_px, min_width=max_px, border_mode=0, value=(0,0,0)),

                ToTensorV2()]) # numpy HWC image -> pytorch CHW tensor 
        # Validation set transforms only include basic conversions and resizing
        else:
            self.tf = A.Compose([
                A.ToFloat(max_value=255),
                A.Resize(min_px, min_px),
                # A.FDA(reference_images = ref_images, beta_limit=.35, p=1),
                # Custom_Transform(reference_images = ref_images, p=1),
                # A.ToGray(p=1.0),
                ToTensorV2()])

    # Allow the class instance to be called as a function to transform images
    def __call__(self, img, *args, **kwargs):
        return self.tf(image=np.array(img))['image']

# Function to convert single-channel tif images to RGB
def tif1c_to_tif3c(path):
    """Converts single-channel tif images to RGB

    Args:
        path (string): A root folder containing original input images

    Returns:
        img_tif3c (numpy.ndarray): tif image converted to RGB
    """
    img_tif1c = tiff.imread(path)
    img_tif1c = np.array(img_tif1c)
    img_rgb = np.zeros((img_tif1c.shape[0],img_tif1c.shape[1],3),dtype=np.uint8) # blank array
    img_rgb[:,:,0] = img_tif1c # copy img 3 times to make the format of img.shape=[H, W, C]
    img_rgb[:,:,1] = img_tif1c
    img_rgb[:,:,2] = img_tif1c
    img_tif3c = img_rgb
    
    # normalize image to 8-bit range
    img_norm = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0,255)).astype(np.uint8)
    img_tif3c = img_norm
    
    print(img_tif3c.dtype)
    return img_tif3c

# Define a PyTorch Lightning data module for handling dataset
class McolonyDataModule(pl.LightningDataModule):
    def __init__(self, root: str, dl_workers: int = 0, batch_size=4, sampler: str = None):
        super().__init__()
        
        self.transforms = Transforms(train=True)
        # self.train_transforms = Transforms(train=True)
        self.val_transforms = Transforms(train=False)
        self.root = root
        self.workers = dl_workers
        self.batch_size = batch_size
        
        # Load sampler if it exists
        if sampler is not None:
            self.sampler = torch.load(sampler)
            self.sampler.generator = None
        else:
            self.sampler = None
            
    # Setup data for training/validation/testing
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            ds = datasets.ImageFolder(self.root, transform=self.transforms)
            # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
            # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
        if stage == "test" or stage is None:
            # ds = datasets.ImageFolder(self.root, transform=self.transforms)
            ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
        # Create train and validation splits
        train_size = int(np.floor(len(ds)*0.7))
        val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
        self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))
        self.train_ds = ds
    # Define methods to retrieve data loaders for each dataset
    def train_dataloader(self):
        if self.sampler is None:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=False)
        else:
            return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

# Define a class for handling test data
class McolonyTestData(object):
        def __init__(self, root):
            self.root = root
            self.tform = A.Compose([A.ToFloat(max_value=255), A.Resize(min_px, min_px), A.ToGray(p=1.0), ToTensorV2()])
            file_list = glob(root+'*.jpg')
            file_list.sort()
            self.img_idx = [os.path.basename(x) for x in file_list]

        def __getitem__(self, idx):            
            ## load images and labels
            fname = self.img_idx[idx]
            im = Image.open(self.root+fname)
            im = im.convert("RGB")
            im = self.tform(image=np.array(im))
            return im, fname

        def __len__(self):
            return len(self.img_idx)



# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# import tifffile as tiff
# from PIL import Image
# import skimage.exposure as exposure
# from glob import glob
# import pytorch_lightning as pl

# # Input image dimensions (match this with your dataset)
# max_px = 1002
# min_px = 1002

# # reference images for domain adapt
# # ref_images = ['/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Ec-60x-3h-BF-20220818-R1-011.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Pf-60x-3h-BF-20220822-R1-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/SE-60x-3h-BF-20220821-R2-005.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bc-60x-3h-BF-20220823-R3-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bs-60x-3h-BF-20220822-R3-001.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Li-60x-3h-BF-20220818-R1-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Lm-60x-3h-BF-20220819-R1-006.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/ST-60x-3h-BF-20220821-R2-002.jpg'
# # ]

# # class Custom_Transform(A.ImageOnlyTransform):
# #     def __init__(self, reference_images, **kwargs):
# #         super().__init__(**kwargs)
# #         self.reference_images = reference_images
# #     def apply(self, img, **params):
# #         reference_images = self.reference_images
# #         aug = A.Compose([A.FDA(reference_images, beta_limit=.45, p=1)])
# #         result = aug(image=img)
# #         return A.InvertImg(p=1)(image=result['image'])['image']
# #     def get_transform_init_args_names(self):
# #         return ()
# # Define a class for data augmentation
# class Transforms:
#     # Training set transforms include several augmentation techniques
#     def __init__(self, train=True):
#         if train:
#             self.tf = A.Compose([
#                 A.Resize(min_px, min_px),
#                 # A.FDA(reference_images = ref_images, beta_limit=.35, p=.6),
#                 A.ToFloat(max_value=255),
#                 # A.InvertImg(p=0.5),
#                 #Custom_Transform(reference_images = ref_images, p=1),
#                 # A.ToGray(p=1.0),
#                 ## Model v1
#                 A.Flip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 A.RandomBrightnessContrast(brightness_limit=0.2,
#                                            contrast_limit=0.2,
#                                            brightness_by_max=True,
#                                            p=0.5),
                
#                 ## Model v2 only
#                 A.Transpose(p=0.5),
#                 A.Blur(blur_limit=5, p=0.5),
#                 A.MedianBlur(blur_limit=5, p=0.5),
#                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.5),
#                 A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
#                                 border_mode=4, value=None, mask_value=None, 
#                                 normalized=False, p=0.5),
#                 A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
#                                     interpolation=1, border_mode=4, value=None, 
#                                     mask_value=None, p=0.5),
#                 A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
#                          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
#                          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
#                 A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
#                               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
#                 A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),

#                 A.Defocus(radius=(3,10), alias_blur=(.1,.5), p=0.5),
#                 A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
#                 # A.ZoomBlur(max_factor=1.31, step_factor=(0.001, 0.003), p=0.5),
#                 # A.GaussNoise(var_limit=(0.1, 2.0), mean=0, per_channel=True, p=0.1),
#                 # A.ElasticTransform(alpha=0.05, sigma=0.005, alpha_affine=0.005, p=0.1),
#                 # A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=2, p=0.1),
#                 # A.CoarseDropout(max_holes=1, max_height=1, max_width=1, min_holes=None, min_height=None, min_width=None, fill_value=0, p=0.1),

#                 ## Model v2a only
#                 # A.Transpose(p=0.6),
#                 # A.Blur(blur_limit=3, p=0.4),
#                 # # A.MedianBlur(blur_limit=3, p=0.4),
#                 # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.4),
#                 # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
#                 #                 border_mode=4, value=None, mask_value=None, 
#                 #                 normalized=False, p=0.5),
#                 # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
#                 #                     interpolation=1, border_mode=4, value=None, 
#                 #                     mask_value=None, p=0.4),
#                 # A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
#                 #          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
#                 #          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
#                 # A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
#                 #               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
#                 # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    
#                 ## Model v3 only
#                 # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
#                 # A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.5),
#                 # A.ZoomBlur(max_factor=1.31, step_factor=(0.01, 0.03), p=0.5),
#                 # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
#                 #                num_shadows_upper=2, shadow_dimension=5, p=0.5),
#                 # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
#                 # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
#                 # A.RandomScale(scale_limit=(-0.7,0), interpolation=1, p=0.5), #scaling factor range=(0.3,1) ~obj lens
#                 # A.PadIfNeeded(min_height=max_px, min_width=max_px, border_mode=0, value=(0,0,0)),

#                 ToTensorV2()]) # numpy HWC image -> pytorch CHW tensor 
#         # Validation set transforms only include basic conversions and resizing
#         else:
#             self.tf = A.Compose([
#                 A.ToFloat(max_value=255),
#                 A.Resize(min_px, min_px),
#                 # A.FDA(reference_images = ref_images, beta_limit=.35, p=1),
#                 # Custom_Transform(reference_images = ref_images, p=1),
#                 # A.ToGray(p=1.0),
#                 ToTensorV2()])

#     # Allow the class instance to be called as a function to transform images
#     def __call__(self, img, *args, **kwargs):
#         img_copy = np.ascontiguousarray(img).astype(np.float32)
#         try:
#             tf_img =  self.tf(image=img_copy)['image']
#         except Exception as e:
#             img_copy = np.flip(img_copy,axis=0).copy()
#             tf_img =  self.tf(image=img_copy)['image']
#         return tf_img

# # Function to convert single-channel tif images to RGB
# def tif1c_to_tif3c(path):
#     """Converts single-channel tif images to RGB

#     Args:
#         path (string): A root folder containing original input images

#     Returns:
#         img_tif3c (numpy.ndarray): tif image converted to RGB
#     """
#     img_tif1c = tiff.imread(path)
#     img_tif1c = np.array(img_tif1c)
#     img_rgb = np.zeros((img_tif1c.shape[0],img_tif1c.shape[1],3),dtype=np.uint8) # blank array
#     img_rgb[:,:,0] = img_tif1c # copy img 3 times to make the format of img.shape=[H, W, C]
#     img_rgb[:,:,1] = img_tif1c
#     img_rgb[:,:,2] = img_tif1c
#     img_tif3c = img_rgb
    
#     # normalize image to 8-bit range
#     img_norm = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0,255)).astype(np.uint8)
#     img_tif3c = img_norm
    
#     print(img_tif3c.dtype)
#     return img_tif3c

# # Define a PyTorch Lightning data module for handling dataset
# class McolonyDataModule(pl.LightningDataModule):
#     def __init__(self, root: str, dl_workers: int = 0, batch_size=4, sampler: str = None):
#         super().__init__()
        
#         self.transforms = Transforms(train=True)
#         # self.train_transforms = Transforms(train=True)
#         self.val_transforms = Transforms(train=False)
#         self.root = root
#         self.workers = dl_workers
#         self.batch_size = batch_size
        
#         # Load sampler if it exists
#         if sampler is not None:
#             self.sampler = torch.load(sampler)
#             self.sampler.generator = None
#         else:
#             self.sampler = None
            
#     # Setup data for training/validation/testing
#     def setup(self, stage: str = None):
#         if stage == "fit" or stage is None:
#             ds = datasets.ImageFolder(self.root, transform=self.transforms)
#             # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
#             # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
#         if stage == "test" or stage is None:
#             # ds = datasets.ImageFolder(self.root, transform=self.transforms)
#             ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
#         # Create train and validation splits
#         train_size = int(np.floor(len(ds)*0.7))
#         val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
#         self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))
#         self.train_ds = ds
#     # Define methods to retrieve data loaders for each dataset
#     def train_dataloader(self):
#         if self.sampler is None:
#             return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=False)
#         else:
#             return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.workers, pin_memory=False)

#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

# # Define a class for handling test data
# class McolonyTestData(object):
#         def __init__(self, root):
#             self.root = root
#             self.tform = A.Compose([A.ToFloat(max_value=255), A.Resize(min_px, min_px), A.ToGray(p=1.0), ToTensorV2()])
#             file_list = glob(root+'*.jpg')
#             file_list.sort()
#             self.img_idx = [os.path.basename(x) for x in file_list]

#         def __getitem__(self, idx):            
#             ## load images and labels
#             fname = self.img_idx[idx]
#             im = Image.open(self.root+fname)
#             im = im.convert("RGB")
#             im = self.tform(image=np.array(im))
#             return im, fname

#         def __len__(self):
#             return len(self.img_idx)
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# import tifffile as tiff
# from PIL import Image
# import skimage.exposure as exposure
# from glob import glob
# import pytorch_lightning as pl

# # Input image dimensions (match this with your dataset)
# max_px = 1002
# min_px = 1002

# # reference images for domain adapt
# # ref_images = ['/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Ec-60x-3h-BF-20220818-R1-011.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Pf-60x-3h-BF-20220822-R1-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/SE-60x-3h-BF-20220821-R2-005.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bc-60x-3h-BF-20220823-R3-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Bs-60x-3h-BF-20220822-R3-001.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Li-60x-3h-BF-20220818-R1-003.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/Lm-60x-3h-BF-20220819-R1-006.jpg',
# #               '/mnt/data/mcolony-classification-data/test/60x-3h-2-var/brightfield/ST-60x-3h-BF-20220821-R2-002.jpg'
# # ]

# # class Custom_Transform(A.ImageOnlyTransform):
# #     def __init__(self, reference_images, **kwargs):
# #         super().__init__(**kwargs)
# #         self.reference_images = reference_images
# #     def apply(self, img, **params):
# #         reference_images = self.reference_images
# #         aug = A.Compose([A.FDA(reference_images, beta_limit=.45, p=1)])
# #         result = aug(image=img)
# #         return A.InvertImg(p=1)(image=result['image'])['image']
# #     def get_transform_init_args_names(self):
# #         return ()
# # Define a class for data augmentation
# class Transforms:
#     # Training set transforms include several augmentation techniques
#     def __init__(self, train=True):
#         if train:
#             self.tf = A.Compose([
#                 A.Resize(min_px, min_px),
#                 # A.FDA(reference_images = ref_images, beta_limit=.35, p=.6),
#                 A.ToFloat(max_value=255),
#                 # A.InvertImg(p=0.5),
#                 #Custom_Transform(reference_images = ref_images, p=1),
#                 # A.ToGray(p=1.0),
#                 ## Model v1
#                 A.Flip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#                 A.RandomBrightnessContrast(brightness_limit=0.2,
#                                            contrast_limit=0.2,
#                                            brightness_by_max=True,
#                                            p=0.5),
                
#                 ## Model v2 only
#                 A.Transpose(p=0.5),
#                 A.Blur(blur_limit=5, p=0.5),
#                 A.MedianBlur(blur_limit=5, p=0.5),
#                 A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.5),
#                 A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
#                                 border_mode=4, value=None, mask_value=None, 
#                                 normalized=False, p=0.5),
#                 A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
#                                     interpolation=1, border_mode=4, value=None, 
#                                     mask_value=None, p=0.5),
#                 A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
#                          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
#                          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
#                 A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
#                               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
#                 A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),

#                 A.Defocus(radius=(3,10), alias_blur=(.1,.5), p=0.5),
#                 A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
#                 # A.ZoomBlur(max_factor=1.31, step_factor=(0.001, 0.003), p=0.5),
#                 # A.GaussNoise(var_limit=(0.1, 2.0), mean=0, per_channel=True, p=0.1),
#                 # A.ElasticTransform(alpha=0.05, sigma=0.005, alpha_affine=0.005, p=0.1),
#                 # A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=2, p=0.1),
#                 # A.CoarseDropout(max_holes=1, max_height=1, max_width=1, min_holes=None, min_height=None, min_width=None, fill_value=0, p=0.1),

#                 ## Model v2a only
#                 # A.Transpose(p=0.6),
#                 # A.Blur(blur_limit=3, p=0.4),
#                 # # A.MedianBlur(blur_limit=3, p=0.4),
#                 # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.4),
#                 # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, 
#                 #                 border_mode=4, value=None, mask_value=None, 
#                 #                 normalized=False, p=0.5),
#                 # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, 
#                 #                     interpolation=1, border_mode=4, value=None, 
#                 #                     mask_value=None, p=0.4),
#                 # A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, 
#                 #          shear=None, interpolation=1, mask_interpolation=0, cval=0, 
#                 #          cval_mask=0, mode=0, fit_output=False, keep_ratio=False, p=0.5),
#                 # A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, 
#                 #               mask_pad_val=0, fit_output=False, interpolation=1, p=0.5),
#                 # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    
#                 ## Model v3 only
#                 # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
#                 # A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.5),
#                 # A.ZoomBlur(max_factor=1.31, step_factor=(0.01, 0.03), p=0.5),
#                 # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
#                 #                num_shadows_upper=2, shadow_dimension=5, p=0.5),
#                 # A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
#                 # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
#                 # A.RandomScale(scale_limit=(-0.7,0), interpolation=1, p=0.5), #scaling factor range=(0.3,1) ~obj lens
#                 # A.PadIfNeeded(min_height=max_px, min_width=max_px, border_mode=0, value=(0,0,0)),

#                 ToTensorV2()]) # numpy HWC image -> pytorch CHW tensor 
#         # Validation set transforms only include basic conversions and resizing
#         else:
#             self.tf = A.Compose([
#                 A.ToFloat(max_value=255),
#                 A.Resize(min_px, min_px),
#                 # A.FDA(reference_images = ref_images, beta_limit=.35, p=1),
#                 # Custom_Transform(reference_images = ref_images, p=1),
#                 # A.ToGray(p=1.0),
#                 ToTensorV2()])

#     # Allow the class instance to be called as a function to transform images
#     def __call__(self, img, *args, **kwargs):
#         img_copy = np.ascontiguousarray(img).astype(np.float32)
#         try:
#             tf_img =  self.tf(image=img_copy)['image']
#         except Exception as e:
#             img_copy = np.flip(img_copy,axis=0).copy()
#             tf_img =  self.tf(image=img_copy)['image']
#         return tf_img

# # Function to convert single-channel tif images to RGB
# def tif1c_to_tif3c(path):
#     """Converts single-channel tif images to RGB

#     Args:
#         path (string): A root folder containing original input images

#     Returns:
#         img_tif3c (numpy.ndarray): tif image converted to RGB
#     """
#     img_tif1c = tiff.imread(path)
#     img_tif1c = np.array(img_tif1c)
#     img_rgb = np.zeros((img_tif1c.shape[0],img_tif1c.shape[1],3),dtype=np.uint8) # blank array
#     img_rgb[:,:,0] = img_tif1c # copy img 3 times to make the format of img.shape=[H, W, C]
#     img_rgb[:,:,1] = img_tif1c
#     img_rgb[:,:,2] = img_tif1c
#     img_tif3c = img_rgb
    
#     # normalize image to 8-bit range
#     img_norm = exposure.rescale_intensity(img_rgb, in_range='image', out_range=(0,255)).astype(np.uint8)
#     img_tif3c = img_norm
    
#     print(img_tif3c.dtype)
#     return img_tif3c

# # Define a PyTorch Lightning data module for handling dataset
# class McolonyDataModule(pl.LightningDataModule):
#     def __init__(self, root: str, dl_workers: int = 0, batch_size=4, sampler: str = None):
#         super().__init__()
        
#         self.transforms = Transforms(train=True)
#         # self.train_transforms = Transforms(train=True)
#         self.val_transforms = Transforms(train=False)
#         self.root = root
#         self.workers = dl_workers
#         self.batch_size = batch_size
        
#         # Load sampler if it exists
#         if sampler is not None:
#             self.sampler = torch.load(sampler)
#             self.sampler.generator = None
#         else:
#             self.sampler = None
            
#     # Setup data for training/validation/testing
#     def setup(self, stage: str = None):
#         if stage == "fit" or stage is None:
#             ds = datasets.ImageFolder(self.root, transform=self.transforms)
#             # ds = datasets.ImageFolder(self.root, transform=self.train_transforms)
#             # ds = datasets.ImageFolder(self.root, loader=tif1c_to_tif3c) # for imgs.tif
#         if stage == "test" or stage is None:
#             # ds = datasets.ImageFolder(self.root, transform=self.transforms)
#             ds = datasets.ImageFolder(self.root, transform=self.val_transforms)
            
#         # Create train and validation splits
#         train_size = int(np.floor(len(ds)*0.7))
#         val_size = int(len(ds)-int(np.floor(len(ds)*0.7)))
#         self.train, self.val = random_split(ds, [train_size, val_size], torch.Generator().manual_seed(111821))
#         self.train_ds = ds
#     # Define methods to retrieve data loaders for each dataset
#     def train_dataloader(self):
#         if self.sampler is None:
#             return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=False)
#         else:
#             return DataLoader(self.train, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.workers, pin_memory=False)

#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

#     def test_dataloader(self):
#         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False)

# # Define a class for handling test data
# class McolonyTestData(object):
#         def __init__(self, root):
#             self.root = root
#             self.tform = A.Compose([A.ToFloat(max_value=255), A.Resize(min_px, min_px), A.ToGray(p=1.0), ToTensorV2()])
#             file_list = glob(root+'*.jpg')
#             file_list.sort()
#             self.img_idx = [os.path.basename(x) for x in file_list]

#         def __getitem__(self, idx):            
#             ## load images and labels
#             fname = self.img_idx[idx]
#             im = Image.open(self.root+fname)
#             im = im.convert("RGB")
#             im = self.tform(image=np.array(im))
#             return im, fname

#         def __len__(self):
#             return len(self.img_idx)
