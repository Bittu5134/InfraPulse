from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.70, 1.0),
            ratio=(0.80, 1.25)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(
            distortion_scale=0.18,
            p=0.35
        ),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.05
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.10),
            ratio=(0.4, 2.5),
            value="random"
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tfms, eval_tfms

def make_loaders(
    data_dir,
    image_size=224,
    batch_size=16,
    num_workers=2
):
    train_tfms, eval_tfms = build_transforms(image_size)

    train_ds = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_tfms
    )

    val_ds = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=eval_tfms
    )

    test_ds = None
    try:
        test_ds = datasets.ImageFolder(
            root=f"{data_dir}/test",
            transform=eval_tfms
        )
    except Exception:
        pass

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx
