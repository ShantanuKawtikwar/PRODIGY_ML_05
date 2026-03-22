import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Food101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images. Example: 'data/food-101/food-101'
            split (string): 'train' or 'test'. Will read the corresponding txt file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Determine the meta file mapping path based on split
        meta_dir = os.path.join(root_dir, 'meta')
        if split == 'train':
            txt_path = os.path.join(meta_dir, 'train.txt')
        elif split == 'test':
            txt_path = os.path.join(meta_dir, 'test.txt')
        else:
            raise ValueError("Split must be 'train' or 'test'.")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Missing {txt_path}. Make sure dataset is downloaded properly.")

        # Read the file containing relative paths
        with open(txt_path, 'r') as f:
            self.image_paths = [line.strip() + ".jpg" for line in f.readlines()]
            
        # Read the classes file
        classes_path = os.path.join(meta_dir, 'classes.txt')
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_name = os.path.join(self.root_dir, 'images', rel_path)
        
        # The class name is the parent folder in the path, e.g. "apple_pie/1005649.jpg"
        class_name = rel_path.split('/')[0]
        label = self.class_to_idx[class_name]

        # Open image as RGB
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            # Handle rare corrupt images by returning a blank one or logging it
            print(f"Warning: Issue loading image {img_name}: {e}")
            image = Image.new('RGB', (224, 224)) # Dummy blank image
            
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(root_dir, batch_size=32, num_workers=4):
    """
    Returns train and test dataloaders with standard ResNet transforms.
    """
    # Standard transforms for ResNet family
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Instantiate datasets
    train_dataset = Food101Dataset(root_dir, split='train', transform=train_transform)
    test_dataset = Food101Dataset(root_dir, split='test', transform=test_transform)

    # Instantiate dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes
