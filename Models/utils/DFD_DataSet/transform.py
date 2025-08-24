from torchvision import transforms


# import albumentations as A
# from albumentations.pytorch import ToTensorV2

def build_transforms(height, width):
    default_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])

    return default_transforms


xception_default_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

resnet_default_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

# def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
#                      norm_std=[0.229, 0.224, 0.225], **kwargs):
#     """Builds train and test transform functions.
#
#     Args:
#         height (int): target image height.
#         width (int): target image width.E
#         norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
#         norm_std (list or None, optional): normalization standard deviation values. Default is
#             ImageNet standard deviation values.
#         max_pixel_value (float): max pixel value
#     """
#
#     if norm_mean is None or norm_std is None:
#         norm_mean = [0.485, 0.456, 0.406] # imagenet mean
#         norm_std = [0.229, 0.224, 0.225] # imagenet std
#     normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
#
#     # GaussianBlur = A.GaussianBlur(p=0.5)
#     # GaussianBlur.sigma_limit = (.1, 2.)
#     train_transform = A.Compose([
#         # A.RandomResizedCrop(height=height, width=width, scale=(0.75, 1.0), p=1.0),
#         # A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.8),
#         # A.ToGray(p=0.2),
#         # A.GaussianBlur(p=0.5),
#         # A.HorizontalFlip(),
#         A.Resize(height, width),
#         A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
#         ToTensorV2(),
#     ])
#
#     test_transform = A.Compose([
#         A.Resize(height, width),
#         A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
#         ToTensorV2(),
#     ])
#
#     return train_transform, test_transform
