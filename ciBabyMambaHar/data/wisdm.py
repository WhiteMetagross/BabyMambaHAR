"""
WISDM Dataset Loader

Large-scale Android accelerometer dataset for HAR.

Dataset Details:
- 6 Activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- Sensors: Accelerometer only (Android phone in pocket)
- Sample Rate: 20Hz
- Channels: 3 (acc_x, acc_y, acc_z)
- Samples: ~1M total (29 subjects)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
from pathlib import Path


ACTIVITIES = [
    'Walking',
    'Jogging',
    'Upstairs',
    'Downstairs',
    'Sitting',
    'Standing'
]

ACTIVITY_MAP = {act: idx for idx, act in enumerate(ACTIVITIES)}


class WisdmDataset(Dataset):
    """
    WISDM Activity Recognition Dataset.
    
    Args:
        root: Path to WISDM folder
        split: 'train' or 'test'
        windowSize: Window size for sliding window (default: 128)
        stride: Stride for sliding window (default: 64)
        normalize: Whether to normalize data
        transform: Optional transform to apply
        trainRatio: Ratio of data for training (default: 0.8)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 128,
        stride: int = 64,
        normalize: bool = True,
        transform = None,
        trainRatio: float = 0.8
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        self.stride = stride
        self.normalize = normalize
        self.transform = transform
        self.trainRatio = trainRatio
        
        # Load and process data
        self.data, self.labels = self._loadData()
        
        # Normalize
        if normalize:
            self._normalizeData()
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and window the WISDM data."""
        
        # Find the data file
        dataFile = self._findDataFile()
        
        if dataFile is None:
            raise FileNotFoundError(
                f"WISDM data file not found in {self.root}\n"
                "Please download from: https://www.cis.fordham.edu/wisdm/dataset.php"
            )
        
        # Parse the data file
        allData, allLabels = self._parseDataFile(dataFile)
        
        # Split into train/test
        splitIdx = int(len(allData) * self.trainRatio)
        
        if self.split == 'train':
            data = allData[:splitIdx]
            labels = allLabels[:splitIdx]
        else:
            data = allData[splitIdx:]
            labels = allLabels[splitIdx:]
        
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def _findDataFile(self) -> Optional[Path]:
        """Find the WISDM data file."""
        # Check common file names
        candidates = [
            'WISDM_ar_v1.1_raw.txt',
            'WISDM_ar_latest.txt',
            'WISDM_raw.txt',
            'raw.txt',
        ]
        
        for candidate in candidates:
            filePath = self.root / candidate
            if filePath.exists():
                return filePath
        
        # Search recursively
        for txtFile in self.root.rglob('*.txt'):
            if 'wisdm' in txtFile.name.lower() or 'raw' in txtFile.name.lower():
                return txtFile
        
        return None
    
    def _parseDataFile(self, filePath: Path) -> Tuple[List[np.ndarray], List[int]]:
        """Parse the WISDM raw data file."""
        allData = []
        allLabels = []
        
        # Read and parse file
        currentUser = None
        currentActivity = None
        currentWindow = []
        
        with open(filePath, 'r') as f:
            for line in f:
                # Clean line
                line = line.strip().rstrip(';')
                if not line:
                    continue
                
                # Parse line: user_id, activity, timestamp, x, y, z
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                
                try:
                    userId = int(parts[0])
                    activity = parts[1].strip()
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5].rstrip(';'))
                    
                    # Map activity to index
                    if activity not in ACTIVITY_MAP:
                        continue
                    activityIdx = ACTIVITY_MAP[activity]
                    
                    # Check if user/activity changed
                    if currentUser != userId or currentActivity != activityIdx:
                        # Save current window if long enough
                        if len(currentWindow) >= self.windowSize:
                            windows, labels = self._createWindows(
                                np.array(currentWindow),
                                currentActivity
                            )
                            allData.extend(windows)
                            allLabels.extend(labels)
                        
                        currentUser = userId
                        currentActivity = activityIdx
                        currentWindow = []
                    
                    currentWindow.append([x, y, z])
                    
                except (ValueError, IndexError):
                    continue
        
        # Handle last window
        if len(currentWindow) >= self.windowSize:
            windows, labels = self._createWindows(
                np.array(currentWindow),
                currentActivity
            )
            allData.extend(windows)
            allLabels.extend(labels)
        
        return allData, allLabels
    
    def _createWindows(
        self,
        data: np.ndarray,
        activity: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Create sliding windows from continuous data."""
        windows = []
        labels = []
        
        numWindows = (len(data) - self.windowSize) // self.stride + 1
        
        for i in range(numWindows):
            start = i * self.stride
            end = start + self.windowSize
            window = data[start:end]
            
            windows.append(window)
            labels.append(activity)
        
        return windows, labels
    
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
    def inputShape(self) -> Tuple[int, int]:
        return (self.windowSize, 3)
    
    @property
    def classNames(self):
        return ACTIVITIES


def getWisdmLoaders(
    root: str = './datasets/WISDM_ar_v1.1',
    batchSize: int = 64,
    windowSize: int = 128,
    stride: int = 64,
    normalize: bool = True,
    numWorkers: int = 2,
    trainTransform = None,
    testTransform = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get WISDM train and test data loaders.
    
    Args:
        root: Path to WISDM folder
        batchSize: Batch size
        windowSize: Window size for sliding window
        stride: Stride for sliding window
        normalize: Normalize data
        numWorkers: Number of data loading workers
        trainTransform: Transform for training data
        testTransform: Transform for test data
    
    Returns:
        (trainLoader, testLoader)
    """
    trainDataset = WisdmDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        normalize=normalize,
        transform=trainTransform
    )
    
    testDataset = WisdmDataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=stride,
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
    print("Testing WISDM Dataset...")
    
    try:
        dataset = WisdmDataset(
            root='./datasets/WISDM_ar_v1.1',
            split='train'
        )
        
        print(f"Train samples: {len(dataset)}")
        print(f"Input shape: {dataset.inputShape}")
        print(f"Num classes: {dataset.numClasses}")
        
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"Sample shape: {x.shape}")
            print(f"Label: {y} ({ACTIVITIES[y]})")
        
    except FileNotFoundError as e:
        print(f"Dataset not found. Please run downloadBenchmarkDatasets.py first.\n{e}")
