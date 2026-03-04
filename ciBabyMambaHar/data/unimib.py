"""
UniMiB SHAR Dataset Loader

University of Milano Bicocca Smartphone-based Human Activity Recognition Dataset.
Contains accelerometer data from 30 subjects performing 17 activities.

Download from: http://www.sal.disco.unimib.it/technologies/unimib-shar/
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# UniMiB SHAR has 3 accelerometer channels (x, y, z)
UNIMIB_CHANNELS = 3

# 17 activity classes (9 ADL + 8 falls)
ACTIVITY_LABELS = {
    0: 'StandingUpFS',      # Standing up from sitting
    1: 'StandingUpFL',      # Standing up from lying
    2: 'Walking',
    3: 'Running',
    4: 'GoingUpS',          # Going up stairs
    5: 'Jumping',
    6: 'GoingDownS',        # Going down stairs
    7: 'LyingDownFS',       # Lying down from standing
    8: 'SittingDown',
    # Falls
    9: 'FallingForw',       # Falling forward
    10: 'FallingRight',
    11: 'FallingBack',
    12: 'HittingObstacle',
    13: 'FallingWithPS',    # Falling with protection strategies
    14: 'FallingBackSC',    # Falling backward sitting on chair
    15: 'Syncope',          # Fainting
    16: 'FallingLeft',
}

# ADL only (9 classes) - more commonly used
ADL_LABELS = {
    0: 'StandingUpFS',
    1: 'StandingUpFL', 
    2: 'Walking',
    3: 'Running',
    4: 'GoingUpS',
    5: 'Jumping',
    6: 'GoingDownS',
    7: 'LyingDownFS',
    8: 'SittingDown',
}


class UniMiBSHARDataset(Dataset):
    """UniMiB SHAR Dataset."""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 128,
        task: str = 'adl',  # 'adl' (9 classes) or 'all' (17 classes)
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        testSubjects: list = None,
    ):
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        self.task = task
        self.normalize = normalize
        self.testSubjects = testSubjects or [1, 2, 3, 4, 5, 6]  # 6 subjects for test
        
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
        """Load UniMiB SHAR dataset from .mat files."""
        # Try to find the data directory (handle nested extraction)
        dataDir = None
        for possibleDir in [
            self.root / 'data',
            self.root / 'UniMiB-SHAR' / 'data',
            self.root,
        ]:
            if (possibleDir / 'adl_data.mat').exists():
                dataDir = possibleDir
                break
        
        if dataDir is None:
            raise FileNotFoundError(
                f"UniMiB SHAR dataset not found at {self.root}. "
                f"Please download from http://www.sal.disco.unimib.it/technologies/unimib-shar/"
            )
        
        # Load ADL data (most common task)
        if self.task == 'adl':
            dataFile = dataDir / 'adl_data.mat'
            labelsFile = dataDir / 'adl_labels.mat'
        else:
            # Load full data (ADL + Falls)
            dataFile = dataDir / 'full_data.mat'
            labelsFile = dataDir / 'full_labels.mat' if (dataDir / 'full_labels.mat').exists() else None
        
        # Load data: shape (N, 453) where 453 = 151 timesteps * 3 axes (flattened)
        matData = sio.loadmat(str(dataFile))
        keyName = 'adl_data' if self.task == 'adl' else 'full_data'
        rawData = matData[keyName]  # (N, 453)
        
        # Load labels: shape (N, 3) where columns are [activity, subject, trial]
        matLabels = sio.loadmat(str(labelsFile))
        labelKeyName = 'adl_labels' if self.task == 'adl' else 'full_labels'
        labelData = matLabels[labelKeyName]  # (N, 3)
        
        labels = labelData[:, 0].astype(np.int64)  # Activity label
        subjects = labelData[:, 1].astype(np.int64)  # Subject ID
        
        # Reshape flattened data: (N, 453) -> (N, 151, 3) = (N, T, C)
        nSamples = rawData.shape[0]
        rawData = rawData.reshape(nSamples, 151, 3)  # (N, T=151, C=3) - keep as [T, C]
        
        # Filter by split (subject-wise)
        if self.split == 'train':
            mask = ~np.isin(subjects, self.testSubjects)
        else:
            mask = np.isin(subjects, self.testSubjects)
        
        rawData = rawData[mask]
        labels = labels[mask]
        
        # Ensure labels are 0-indexed
        if labels.min() > 0:
            labels = labels - labels.min()
        
        # Pad or truncate to windowSize (data is [N, T, C])
        data = rawData.astype(np.float32)
        currentLen = data.shape[1]  # T=151
        if currentLen < self.windowSize:
            # Pad with zeros along time axis
            padWidth = ((0, 0), (0, self.windowSize - currentLen), (0, 0))
            data = np.pad(data, padWidth, mode='constant', constant_values=0)
        elif currentLen > self.windowSize:
            # Truncate along time axis
            data = data[:, :self.windowSize, :]
        
        return data, labels
    
    def _loadFromActivityFiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: load from individual activity .mat files."""
        dataDir = self.root / 'data'
        if not dataDir.exists():
            dataDir = self.root
        
        allData = []
        allLabels = []
        allSubjects = []
        
        for actIdx, actName in ACTIVITY_LABELS.items():
            actFile = dataDir / f'{actName}.mat'
            if not actFile.exists():
                continue
            
            matData = sio.loadmat(str(actFile))
            
            # Extract data - format varies
            for key in matData.keys():
                if not key.startswith('_'):
                    actData = matData[key]
                    break
            
            # Assume shape (N, T, 3) or (N, 3, T)
            if actData.ndim == 3:
                if actData.shape[2] == 3:
                    actData = np.transpose(actData, (0, 2, 1))
                
                nSamples = actData.shape[0]
                allData.append(actData)
                allLabels.extend([actIdx] * nSamples)
                # Assume uniform subject distribution
                allSubjects.extend(list(range(1, 31)) * (nSamples // 30 + 1)[:nSamples])
        
        if len(allData) == 0:
            raise FileNotFoundError(
                f"UniMiB SHAR dataset not found at {self.root}. "
                f"Please download from http://www.sal.disco.unimib.it/technologies/unimib-shar/"
            )
        
        data = np.concatenate(allData, axis=0).astype(np.float32)
        labels = np.array(allLabels, dtype=np.int64)
        subjects = np.array(allSubjects[:len(labels)], dtype=np.int32)
        
        # Filter by split
        if self.split == 'train':
            mask = ~np.isin(subjects, self.testSubjects)
        else:
            mask = np.isin(subjects, self.testSubjects)
        
        data = data[mask]
        labels = labels[mask]
        
        # Filter by task
        if self.task == 'adl':
            adlMask = labels < 9
            data = data[adlMask]
            labels = labels[adlMask]
        
        # Adjust window size
        currentLen = data.shape[2]
        if currentLen < self.windowSize:
            padWidth = ((0, 0), (0, 0), (0, self.windowSize - currentLen))
            data = np.pad(data, padWidth, mode='constant', constant_values=0)
        elif currentLen > self.windowSize:
            data = data[:, :, :self.windowSize]
        
        return data, labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


def getUniMiBLoaders(
    root: str = './datasets/UniMiB-SHAR',
    batchSize: int = 64,
    windowSize: int = 128,
    task: str = 'adl',
    numWorkers: int = 4,
    returnWeights: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get UniMiB SHAR train and test data loaders.
    
    Args:
        root: Path to UniMiB SHAR dataset
        batchSize: Batch size
        windowSize: Window size (128, 256, or 512)
        task: 'adl' (9 classes) or 'all' (17 classes including falls)
        numWorkers: Number of data loading workers
        returnWeights: If True, also return class weights
    
    Returns:
        trainLoader, testLoader, (optional: classWeights)
    """
    trainDataset = UniMiBSHARDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        task=task,
        normalize=True,
    )
    
    # Use train stats for test normalization
    testDataset = UniMiBSHARDataset(
        root=root,
        split='test',
        windowSize=windowSize,
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
        classCounts = np.bincount(trainDataset.labels)
        classWeights = 1.0 / (classCounts + 1e-6)
        classWeights = classWeights / classWeights.sum() * len(classCounts)
        classWeights = torch.FloatTensor(classWeights)
        return trainLoader, testLoader, classWeights
    
    return trainLoader, testLoader


def getUniMiBInfo(task: str = 'adl') -> dict:
    """Get dataset information."""
    if task == 'adl':
        return {
            'numClasses': 9,
            'inChannels': UNIMIB_CHANNELS,
            'labels': ADL_LABELS,
        }
    else:
        return {
            'numClasses': 17,
            'inChannels': UNIMIB_CHANNELS,
            'labels': ACTIVITY_LABELS,
        }
