"""
MotionSense Dataset Loader

Real-world "In the Wild" HAR dataset collected from iPhone 6s.

Dataset Details:
- 6 Activities: Downstairs, Upstairs, Walking, Jogging, Sitting, Standing
- Sensors: Accelerometer + Gyroscope + Attitude (iPhone 6s)
- Window: Variable, typically 50Hz
- Channels: 12 (acc_xyz, gyro_xyz, attitude_roll/pitch/yaw, gravity_xyz)
- Samples: ~30k total (24 subjects)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
from pathlib import Path


ACTIVITIES = [
    'dws',  # Downstairs
    'ups',  # Upstairs
    'wlk',  # Walking
    'jog',  # Jogging
    'sit',  # Sitting
    'std'   # Standing
]

ACTIVITY_MAP = {act: idx for idx, act in enumerate(ACTIVITIES)}


class MotionSenseDataset(Dataset):
    """
    MotionSense Dataset for HAR.
    
    Args:
        root: Path to motion-sense-master folder
        split: 'train' or 'test'
        windowSize: Window size for sliding window (default: 128)
        stride: Stride for sliding window (default: 64)
        normalize: Whether to normalize data
        transform: Optional transform to apply
        trainRatio: Ratio of subjects for training (default: 0.8)
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
        """Load and window the MotionSense data."""
        allData = []
        allLabels = []
        
        # Find data directory  
        dataDir = self._findDataDir()
        
        if dataDir is None:
            raise FileNotFoundError(
                f"MotionSense data not found in {self.root}\n"
                "Please download from: https://github.com/mmalekzadeh/motion-sense"
            )
        
        # MotionSense structure: data/A_DeviceMotion_data/activity_X/sub_Y.csv
        # Activities: dws, jog, sit, std, ups, wlk (with suffix like _1, _11)
        
        # First, collect all unique subjects across all activities
        allSubjects = set()
        activityDirs = sorted([d for d in dataDir.iterdir() if d.is_dir()])
        
        for actDir in activityDirs:
            for csvFile in actDir.glob('sub_*.csv'):
                subjectId = csvFile.stem  # e.g. 'sub_1'
                allSubjects.add(subjectId)
        
        allSubjects = sorted(list(allSubjects))
        numSubjects = len(allSubjects)
        
        if numSubjects == 0:
            # Fallback to alternative loading
            return self._loadAlternative()
        
        # Split subjects (LOSO-style)
        trainSubjects = int(numSubjects * self.trainRatio)
        if self.split == 'train':
            selectedSubjects = set(allSubjects[:trainSubjects])
        else:
            selectedSubjects = set(allSubjects[trainSubjects:])
        
        # Load data from each activity folder
        for actDir in activityDirs:
            # Parse activity from folder name (e.g., 'dws_1' -> 'dws')
            activity = self._parseActivity(actDir.name)
            if activity is None:
                continue
            
            # Load CSV files for selected subjects
            for csvFile in actDir.glob('sub_*.csv'):
                subjectId = csvFile.stem
                if subjectId not in selectedSubjects:
                    continue
                
                try:
                    df = pd.read_csv(csvFile)
                    data = self._processDataframe(df)
                    
                    if data is not None and len(data) >= self.windowSize:
                        windows, labels = self._createWindows(data, activity)
                        allData.extend(windows)
                        allLabels.extend(labels)
                except Exception as e:
                    continue
        
        if len(allData) == 0:
            # Try alternative loading method
            allData, allLabels = self._loadAlternative()
        
        return np.array(allData, dtype=np.float32), np.array(allLabels, dtype=np.int64)
    
    def _findDataDir(self) -> Optional[Path]:
        """Find the data directory within the root."""
        # Check common locations - prioritize A_DeviceMotion_data which has the best data
        candidates = [
            self.root / 'data' / 'A_DeviceMotion_data',
            self.root / 'A_DeviceMotion_data',
            self.root / 'motion-sense-master' / 'data' / 'A_DeviceMotion_data',
            self.root / 'data',
            self.root / 'motion-sense-master' / 'data',
        ]
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Check if it has activity folders
                hasDirs = any(d.is_dir() for d in candidate.iterdir())
                if hasDirs:
                    return candidate
        
        return None
    
    def _parseActivity(self, name: str) -> Optional[int]:
        """Parse activity label from folder/file name."""
        nameLower = name.lower()
        
        for activity in ACTIVITIES:
            if activity in nameLower:
                return ACTIVITY_MAP[activity]
        
        return None
    
    def _processDataframe(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Process a dataframe to extract sensor channels."""
        # Expected columns (subset)
        accCols = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']
        gyroCols = ['rotationRate.x', 'rotationRate.y', 'rotationRate.z']
        
        # Check if columns exist
        availableCols = []
        for col in accCols + gyroCols:
            if col in df.columns:
                availableCols.append(col)
        
        if len(availableCols) == 0:
            return None
        
        return df[availableCols].values.astype(np.float32)
    
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
    
    def _loadAlternative(self) -> Tuple[List[np.ndarray], List[int]]:
        """Alternative loading method for different folder structures."""
        allData = []
        allLabels = []
        
        # Try loading from CSV files directly
        for csvFile in self.root.rglob('*.csv'):
            try:
                df = pd.read_csv(csvFile)
                
                # Try to infer activity from filename
                activity = self._parseActivity(csvFile.stem)
                if activity is None:
                    continue
                
                data = self._processDataframe(df)
                if data is not None and len(data) >= self.windowSize:
                    windows, labels = self._createWindows(data, activity)
                    allData.extend(windows)
                    allLabels.extend(labels)
            except:
                continue
        
        return allData, allLabels
    
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
        if len(self.data) > 0:
            return self.data[0].shape
        return (self.windowSize, 6)
    
    @property
    def classNames(self):
        return ACTIVITIES


def getMotionSenseLoaders(
    root: str = './datasets/motion-sense-master',
    batchSize: int = 64,
    windowSize: int = 128,
    stride: int = 64,
    normalize: bool = True,
    numWorkers: int = 2,
    trainTransform = None,
    testTransform = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MotionSense train and test data loaders.
    
    Args:
        root: Path to motion-sense-master folder
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
    trainDataset = MotionSenseDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        normalize=normalize,
        transform=trainTransform
    )
    
    testDataset = MotionSenseDataset(
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
    print("Testing MotionSense Dataset...")
    
    try:
        dataset = MotionSenseDataset(
            root='./datasets/motion-sense-master',
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
