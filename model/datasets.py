from __future__ import annotations
import csv, random, numpy as np, torch
from pathlib import Path
from typing import List, Tuple, cast, Sized
from setup_env import *
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import tempfile
import os
import matplotlib.pyplot as plt
from collections import Counter

# Default paths
DEFAULT_IMAGES_DIR = "data/_tmp_flyers/images"
DEFAULT_ANNOTATIONS_CSV = "data/_tmp_flyers/annotations.csv"

# Default configuration
IMG_SIZE = 512
IMG_CH = 3
BATCH_SIZE = 16
NUM_WORKERS = 0

def get_transforms(image_size: int = IMG_SIZE, train: bool = True):
    """Return torchvision transforms for train/validation.

    Use milder augmentations for flyers so text/layout are preserved.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class FlyerDataset(Dataset):
    """Dataset for flyers backed by a CSV annotations file.

    CSV must contain at least columns: filename, labels
    `labels` should be comma-separated tag names. Example row:
        image1.jpg, sale,holiday,free

    Returns:
        (image_tensor, label_vector, metadata) where metadata includes filename and raw labels.
    """

    def __init__(self, csv_path: Path | str, images_dir: Path | str, transform=None):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.data = []
        # accumulate unique class names in a set, then convert to a sorted list
        self._class_name_set = set()
        self.class_names: List[str] = []

        with self.csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
                self._class_name_set.update(row['labels'].split(','))

        self.class_names = sorted(list(self._class_name_set))  # Convert set to sorted list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image_path = self.images_dir / row['filename']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = row['labels'].split(',')
        metadata = {'filename': row['filename'], 'labels': row['labels']}

        # Convert labels to a single class index (select the first label)
        label_index = self.class_names.index(labels[0])  # Map the first label to index

        # Debugging logs
        print(f"Image shape: {image.size}, Labels: {row['labels']}, Metadata: {row}, Label Index: {label_index}")

        return image, label_index, metadata
class CustomDataset(Dataset):
    """Wrapper around FlyerDataset to forward items (keeps metadata intact)."""

    def __init__(self, dataset: FlyerDataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        # FlyerDataset already returns (image, label_vector, metadata)
        image, label_vector, metadata = self.dataset[index]
        return image, label_vector, metadata

    def __len__(self) -> int:
        return len(self.dataset)


def deterministic_split(dataset: Dataset, seed: int = SEED,
                        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[Subset, Subset, Subset]:
    """Split a dataset deterministically into train/val/test Subsets.

    Returns (train_subset, val_subset, test_subset)
    """
    assert sum(splits) > 0
    N = len(cast(Sized, dataset))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(splits[0] * N)
    n_val = int(splits[1] * N)
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:n_train + n_val].tolist()
    test_idx = perm[n_train + n_val:].tolist()
    # Guard: ensure every index used
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def worker_init_fn(worker_id: int, base_seed: int = SEED):
    """Seed numpy/random/torch inside DataLoader workers for reproducibility."""
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_datasets_and_loaders(
        images_dir: str | Path = DEFAULT_IMAGES_DIR, 
        annotations_csv: str | Path = DEFAULT_ANNOTATIONS_CSV,
        seed: int = SEED, 
        image_size: int = IMG_SIZE,
        batch_size: int = BATCH_SIZE, 
        num_workers: int = NUM_WORKERS,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """Convenience builder: returns train/val/test DataLoaders and class names.
    This function builds a FlyerDataset, performs a deterministic split, sets transforms
    on the underlying dataset and returns DataLoaders.
    """
    base_ds = FlyerDataset(annotations_csv, images_dir, transform=None)
    train_ds, val_ds, test_ds = deterministic_split(base_ds, seed=seed, splits=splits)
    # Attach transforms by updating the underlying dataset object (Subset.dataset)
    cast(FlyerDataset, train_ds.dataset).transform = get_transforms(image_size, train=True)
    cast(FlyerDataset, test_ds.dataset).transform = get_transforms(image_size, train=False)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers))

    return train_loader, val_loader, test_loader, base_ds.class_names


if __name__ == '__main__':
    # Small synthetic smoke-test: create a tiny dataset on disk and try loaders
    import shutil
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir()) / '_tmp_flyers'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    (tmp_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Debugging logs
    print(f"Temporary directory path: {tmp_dir}")
    print(f"Directory exists: {tmp_dir.exists()}")
    print(f"Directory writable: {os.access(tmp_dir, os.W_OK)}")

    # create 8 small synthetic images and a CSV with multi-labels
    class_pool = ['leasing', 'special', 'advertising', 'event', 'holiday', 'national day', 'celebration', 'apartment']
    csv_path = tmp_dir / 'annotations.csv'
    rows = []
    for i in range(8):
        img = Image.new('RGB', (640, 480), color=(int(30 + i * 20), 120, 160))
        fname = f'f{i}.jpg'
        img.save(tmp_dir / 'images' / fname)
        # assign 1-2 random tags
        tags = random.sample(class_pool, k=random.choice([1, 2]))
        rows.append((fname, ','.join(tags)))

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'labels'])
        for r in rows:
            writer.writerow(r)

    print('Created synthetic dataset at', tmp_dir)
    loaders = make_datasets_and_loaders(tmp_dir / 'images', csv_path, seed=123, image_size=128, batch_size=4, num_workers=0)
    train_loader, val_loader, test_loader, classes = loaders
    print('Classes:', classes)
    batch = next(iter(train_loader))
    imgs, labels, metadata = batch
    print('train batch images shape:', imgs.shape)
    print('train batch labels shape:', labels.shape)
    print('First label vector:', labels[0])
    print('First metadata (filename):', metadata['filename'])
    print('First metadata (labels):', metadata['labels'])

def create_train_loader(data_dir, image_size: int, batch_size: int = 32, splits=(0.8,0.1,0.1), seed=SEED, num_workers: int = 4):
    """
    Create and return a train DataLoader along with validation and test SubSets.
    data_dir: path to the dataset root (passed to FlyerDataset)
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / 'annotations.csv'
    dataset = FlyerDataset(csv_path, data_dir / 'images', transform=get_transforms(image_size, train=True))
    train_ds, val_ds, test_ds = deterministic_split(dataset, seed, splits)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed))
    return train_loader, val_ds, test_ds

def visualize_dataset(dataset: FlyerDataset, num_samples: int = 5):
    """
    Visualize the dataset: plot label distribution and show sample images with labels.
    
    Args:
        dataset: FlyerDataset instance (use transform=None for raw images)
        num_samples: Number of sample images to display
    """
    # Collect all labels
    all_labels = []
    for row in dataset.data:
        all_labels.extend(row['labels'].split(','))
    
    # Plot label distribution
    label_counts = Counter(all_labels)
    labels, counts = zip(*label_counts.items())
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
    # Show sample images in a separate figure
    num_to_show = min(num_samples, len(dataset))
    fig, axes = plt.subplots(1, num_to_show, figsize=(5 * num_to_show, 5))
    if num_to_show == 1:
        axes = [axes]
    
    for i in range(num_to_show):
        image, label_index, metadata = dataset[i]
        # Convert tensor back to PIL for display
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {dataset.class_names[label_index]}\nFile: {metadata['filename']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
     dataset = FlyerDataset(DEFAULT_ANNOTATIONS_CSV, DEFAULT_IMAGES_DIR, transform=None)  # No transform for raw images
     visualize_dataset(dataset, num_samples=3)
