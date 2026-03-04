"""
Opportunity Activity Recognition Dataset Loader

The OPPORTUNITY Activity Recognition Dataset contains data from 
on-body sensors during daily activities (12 subjects, 18 activities).

Download from: https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# Opportunity has 113 sensor channels after removing ambient sensors
# We use the 79 body-worn IMU channels (accelerometer + gyroscope)
OPPORTUNITY_CHANNELS = 79

# Labels for locomotion task (most common benchmark)
LOCOMOTION_LABELS = {
    0: 'Null',
    1: 'Stand',
    2: 'Walk', 
    3: 'Sit',
    4: 'Lie'
}

# Labels for gesture task (more challenging)
GESTURE_LABELS = {
    0: 'Null',
    1: 'Open Door 1',
    2: 'Open Door 2',
    3: 'Close Door 1',
    4: 'Close Door 2',
    5: 'Open Fridge',
    6: 'Close Fridge',
    7: 'Open Dishwasher',
    8: 'Close Dishwasher',
    9: 'Open Drawer 1',
    10: 'Close Drawer 1',
    11: 'Open Drawer 2',
    12: 'Close Drawer 2',
    13: 'Open Drawer 3',
    14: 'Close Drawer 3',
    15: 'Clean Table',
    16: 'Drink from Cup',
    17: 'Toggle Switch'
}


class OpportunityDataset(Dataset):
    """Opportunity Activity Recognition Dataset."""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 128,
        stride: int = 64,
        task: str = 'locomotion',  # 'locomotion' or 'gesture'
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        self.stride = stride
        self.task = task
        self.normalize = normalize
        
        # Load and process data
        self.data, self.labels = self._loadData()
        
        # Compute or use provided normalization stats
        if normalize:
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std
            else:
                # Data is [N, T, C], compute mean/std over samples and time
                self.mean = np.mean(self.data, axis=(0, 1), keepdims=True)
                self.std = np.std(self.data, axis=(0, 1), keepdims=True) + 1e-8
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Opportunity dataset files."""
        dataDir = self.root / 'OpportunityUCIDataset' / 'dataset'
        
        if not dataDir.exists():
            # Try alternate path
            dataDir = self.root / 'dataset'
            if not dataDir.exists():
                raise FileNotFoundError(
                    f"Opportunity dataset not found at {self.root}. "
                    f"Please download from UCI ML Repository."
                )
        
        # Subject-run mapping for train/test split
        # Standard split: S1-S3 for train, S4 for test (or use ADL sessions)
        if self.split == 'train':
            subjects = ['S1', 'S2', 'S3']
            sessions = ['ADL1', 'ADL2', 'ADL3', 'Drill']
        else:
            subjects = ['S2', 'S3']  # Use S2, S3 ADL4, ADL5 for test
            sessions = ['ADL4', 'ADL5']
        
        allData = []
        allLabels = []
        
        for subject in subjects:
            for session in sessions:
                filename = f"{subject}-{session}.dat"
                filepath = dataDir / filename
                
                if not filepath.exists():
                    continue
                
                # Load data file (space-separated)
                try:
                    rawData = np.loadtxt(filepath)
                except Exception:
                    continue
                
                # Extract sensor channels (columns 1-113) and labels
                # Column 0 is timestamp
                # Columns 1-113 are sensor data
                # Column 243 is locomotion label
                # Column 249 is gesture label (ML_Both_Arms)
                
                sensorData = rawData[:, 1:114]  # 113 channels
                
                # Select body-worn IMU channels only (remove object/ambient sensors)
                # Back, RUA, RLA, LUA, LLA sensors (each has acc+gyro = 6 channels)
                # Plus some additional body sensors
                imuIndices = list(range(0, 79))  # First 79 channels are body IMUs
                sensorData = sensorData[:, imuIndices]
                
                if self.task == 'locomotion':
                    labels = rawData[:, 243].astype(int)
                else:  # gesture
                    labels = rawData[:, 249].astype(int)
                
                # Handle NaN values
                sensorData = np.nan_to_num(sensorData, nan=0.0)
                
                # Create sliding windows
                for i in range(0, len(sensorData) - self.windowSize, self.stride):
                    window = sensorData[i:i + self.windowSize]
                    windowLabels = labels[i:i + self.windowSize]
                    
                    # Use majority label for the window
                    label = np.bincount(windowLabels[windowLabels >= 0]).argmax()
                    
                    # Skip null class for gesture task to reduce imbalance
                    if self.task == 'gesture' and label == 0:
                        continue
                    
                    allData.append(window)  # Shape: (windowSize, channels) = (T, C)
                    allLabels.append(label)
        
        if len(allData) == 0:
            raise ValueError(f"No data found for split '{self.split}'")
        
        data = np.array(allData, dtype=np.float32)
        labels = np.array(allLabels, dtype=np.int64)
        
        # Remap labels to be contiguous starting from 0
        uniqueLabels = np.unique(labels)
        labelMap = {old: new for new, old in enumerate(uniqueLabels)}
        labels = np.array([labelMap[l] for l in labels])
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


def getOpportunityLoaders(
    root: str = './datasets/Opportunity',
    batchSize: int = 64,
    windowSize: int = 128,
    stride: int = 64,
    task: str = 'locomotion',
    numWorkers: int = 4,
    returnWeights: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Opportunity train and test data loaders.
    
    Args:
        root: Path to Opportunity dataset
        batchSize: Batch size
        windowSize: Sliding window size (128, 256, or 512)
        stride: Stride for sliding window
        task: 'locomotion' (5 classes) or 'gesture' (18 classes)
        numWorkers: Number of data loading workers
        returnWeights: If True, also return class weights for imbalanced data
    
    Returns:
        trainLoader, testLoader, (optional: classWeights)
    """
    trainDataset = OpportunityDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        task=task,
        normalize=True,
    )
    
    # Use train stats for test normalization
    testDataset = OpportunityDataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=stride,
        task=task,
        normalize=True,
        mean=trainDataset.mean,
        std=trainDataset.std,
    )
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=True,
    )
    
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True,
    )
    
    if returnWeights:
        # Compute class weights for imbalanced dataset
        classCounts = np.bincount(trainDataset.labels)
        classWeights = 1.0 / (classCounts + 1e-6)
        classWeights = classWeights / classWeights.sum() * len(classCounts)
        classWeights = torch.FloatTensor(classWeights)
        return trainLoader, testLoader, classWeights
    
    return trainLoader, testLoader


def getOpportunityInfo(task: str = 'locomotion') -> dict:
    """Get dataset information."""
    if task == 'locomotion':
        return {
            'numClasses': 5,
            'inChannels': OPPORTUNITY_CHANNELS,
            'labels': LOCOMOTION_LABELS,
        }
    else:
        return {
            'numClasses': 18,
            'inChannels': OPPORTUNITY_CHANNELS,
            'labels': GESTURE_LABELS,
        }
