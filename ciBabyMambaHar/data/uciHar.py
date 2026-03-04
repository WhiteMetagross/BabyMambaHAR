"""
UCI-HAR Dataset Loader

The "Hello World" of Human Activity Recognition.

Dataset Details:
- 6 Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- Sensors: Accelerometer + Gyroscope (waist-mounted smartphone)
- Window: 2.56 sec at 50Hz = 128 timesteps
- Channels: 9 (body_acc_xyz, body_gyro_xyz, total_acc_xyz)
- Samples: ~10k total (7352 train, 2947 test)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict


ACTIVITIES = [
    'WALKING',
    'WALKING_UPSTAIRS', 
    'WALKING_DOWNSTAIRS',
    'SITTING',
    'STANDING',
    'LAYING'
]

SIGNAL_NAMES = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]


class UciHarDataset(Dataset):
    """
    UCI Human Activity Recognition Dataset.
    
    Args:
        root: Path to 'UCI HAR Dataset' folder
        split: 'train' or 'test'
        useRaw: If True, use raw inertial signals; else use 561 features
        normalize: Whether to normalize data
        transform: Optional transform to apply
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        useRaw: bool = True,
        normalize: bool = True,
        transform = None
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.useRaw = useRaw
        self.normalize = normalize
        self.transform = transform
        
        # Load data
        if useRaw:
            self.data = self._loadRawSignals()
        else:
            self.data = self._loadFeatures()
        
        self.labels = self._loadLabels()
        
        # Normalize
        if normalize:
            self._normalizeData()
    
    def _loadRawSignals(self) -> np.ndarray:
        """Load raw inertial signals (128 timesteps x 9 channels)."""
        signals = []
        
        for name in SIGNAL_NAMES:
            filepath = os.path.join(
                self.root,
                self.split,
                'Inertial Signals',
                f'{name}_{self.split}.txt'
            )
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Signal file not found: {filepath}\n"
                    f"Please download UCI-HAR dataset and extract to: {self.root}"
                )
            
            signal = np.loadtxt(filepath)
            signals.append(signal)
        
        # Stack: (N, 128, 9) -> (samples, timesteps, channels)
        return np.stack(signals, axis=-1).astype(np.float32)
    
    def _loadFeatures(self) -> np.ndarray:
        """Load pre-computed 561 features."""
        filepath = os.path.join(self.root, self.split, f'X_{self.split}.txt')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature file not found: {filepath}")
        
        return np.loadtxt(filepath).astype(np.float32)
    
    def _loadLabels(self) -> np.ndarray:
        """Load activity labels (convert to 0-indexed)."""
        filepath = os.path.join(self.root, self.split, f'y_{self.split}.txt')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Label file not found: {filepath}")
        
        return np.loadtxt(filepath).astype(np.int64) - 1  # 0-indexed
    
    def _normalizeData(self):
        """Normalize data to zero mean and unit variance."""
        mean = self.data.mean()
        std = self.data.std() + 1e-8
        self.data = (self.data - mean) / std
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.data[idx])
        y = int(self.labels[idx])
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @property
    def numClasses(self) -> int:
        return len(ACTIVITIES)
    
    @property
    def inputShape(self) -> Tuple[int, ...]:
        if self.useRaw:
            return (128, 9)  # (seqLen, channels)
        else:
            return (561,)  # (features,)
    
    @property
    def classNames(self):
        return ACTIVITIES


def getUciHarLoaders(
    root: str = './datasets/UCI HAR Dataset',
    batchSize: int = 64,
    useRaw: bool = True,
    normalize: bool = True,
    numWorkers: int = 2,
    trainTransform = None,
    testTransform = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get UCI-HAR train and test data loaders.
    
    Args:
        root: Path to UCI HAR Dataset folder
        batchSize: Batch size
        useRaw: Use raw inertial signals (True) or features (False)
        normalize: Normalize data
        numWorkers: Number of data loading workers
        trainTransform: Transform for training data
        testTransform: Transform for test data
    
    Returns:
        (trainLoader, testLoader)
    """
    trainDataset = UciHarDataset(
        root=root,
        split='train',
        useRaw=useRaw,
        normalize=normalize,
        transform=trainTransform
    )
    
    testDataset = UciHarDataset(
        root=root,
        split='test',
        useRaw=useRaw,
        normalize=normalize,
        transform=testTransform
    )
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=True
    )
    
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True
    )
    
    return trainLoader, testLoader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing UCI-HAR Dataset...")
    
    try:
        dataset = UciHarDataset(
            root='./datasets/UCI HAR Dataset',
            split='train',
            useRaw=True
        )
        
        print(f"Train samples: {len(dataset)}")
        print(f"Input shape: {dataset.inputShape}")
        print(f"Num classes: {dataset.numClasses}")
        
        x, y = dataset[0]
        print(f"Sample shape: {x.shape}")
        print(f"Label: {y} ({ACTIVITIES[y]})")
        
    except FileNotFoundError as e:
        print(f"Dataset not found. Please run downloadBenchmarkDatasets.py first.\n{e}")
