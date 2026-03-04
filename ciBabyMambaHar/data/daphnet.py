"""
Daphnet Freezing of Gait Dataset Loader

Daphnet: Parkinson's Disease Freezing of Gait Detection
- 10 subjects with Parkinson's disease
- 3 IMU sensors (ankle, thigh, trunk)
- 9 channels (3 axes × 3 sensors)
- Sampling rate: 64Hz
- Binary classification: Walk/Stand vs Freeze
- Challenge: Extreme class imbalance (~10% Freeze, 90% Walk/Stand)

Training Strategy (Signal Rescue + Variance Stabilizer):

Phase 1: Signal Rescue (Preprocessing)
- Apply 4th order Butterworth low-pass filter (12Hz cutoff) BEFORE normalization
- Removes sensor jitter (20-30Hz) while preserving Freeze band (3-8Hz)
- Prevents aliasing from Stride-4 tokenizer (64Hz → 16Hz effective)
- Standardize AFTER filtering (not before)

Phase 2: Variance Stabilizer (Training Recipe)
- Aggressive class weights: 15.0 for Freeze, 1.0 for Walk
- WeightedRandomSampler: Force 50/50 class balance in batches
- Linear warmup (10 epochs): 1e-6 → 1e-3
- Cosine annealing: 1e-3 → 1e-5
- Weight decay: 0.05
- Gradient clipping: 1.0
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from scipy.signal import butter, filtfilt


# Daphnet Activity Labels
DAPHNET_ACTIVITIES = {
    0: "not_worn",      # Sensor not worn (excluded)
    1: "walk_stand",    # Walking or Standing (normal)
    2: "freeze"         # Freezing of Gait (target)
}

# Binary classification: Walk/Stand (0) vs Freeze (1)
# We map: Original 1 -> 0 (Walk), Original 2 -> 1 (Freeze)
NUM_CLASSES = 2

# ============================================================================
# SIGNAL RESCUE: Butterworth Low-Pass Filter
# ============================================================================

def butterworthLowpass(cutoff: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth low-pass filter.
    
    Args:
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 4th order for steep rolloff)
    
    Returns:
        (b, a) filter coefficients
    """
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def applyLowpassFilter(
    data: np.ndarray, 
    cutoff: float = 12.0, 
    fs: float = 64.0, 
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to sensor data.
    
    CRITICAL for Daphnet:
    - The "Freeze" band is 3-8 Hz
    - The Stride-4 tokenizer has Nyquist limit of 8 Hz (64Hz / 8)
    - A 12 Hz cutoff preserves the full freeze signal
    - Eliminates 20-30 Hz sensor jitter that causes aliasing
    
    Args:
        data: Sensor data, shape (N, channels) or (T, channels)
        cutoff: Cutoff frequency in Hz (default: 12 Hz)
        fs: Sampling frequency in Hz (default: 64 Hz for Daphnet)
        order: Filter order (default: 4th order Butterworth)
    
    Returns:
        Filtered data with same shape
    """
    b, a = butterworthLowpass(cutoff, fs, order)
    
    # Apply filter to each channel
    filteredData = np.zeros_like(data)
    
    if len(data) < 13:  # padlen = 3 * order + 1 = 13 for order=4
        # Data too short for filtfilt, use simple convolution
        return data
    
    for ch in range(data.shape[1]):
        try:
            filteredData[:, ch] = filtfilt(b, a, data[:, ch])
        except ValueError:
            # Fallback if filter fails
            filteredData[:, ch] = data[:, ch]
    
    return filteredData.astype(np.float32)


class DaphnetDataset(Dataset):
    """
    Daphnet Freezing of Gait Dataset.
    
    Features:
    - 3 IMU sensors on ankle, thigh, trunk
    - 9 channels (3 × 3 accelerometer axes)
    - Binary: Walk/Stand vs Freeze
    - Extreme imbalance (~10% Freeze)
    
    Signal Rescue Pipeline:
    1. Load raw 64Hz data
    2. Apply 4th order Butterworth low-pass filter (12Hz cutoff)
    3. Downsample from 64Hz to 32Hz (after filtering to avoid aliasing)
    4. Segment into windows
    5. Standardize (Z-Score normalization AFTER filtering)
    
    Args:
        root: Path to Daphnet dataset folder
        split: 'train' or 'test'
        windowSize: Sliding window size (default 64 = 2s at 32Hz)
        stride: Window stride (default 8 = 87% overlap for train)
        downsample: Whether to downsample from 64Hz to 32Hz
        applyFilter: Whether to apply Butterworth low-pass filter (default: True)
        filterCutoff: Low-pass filter cutoff frequency in Hz (default: 12.0)
        normalize: Whether to normalize data (AFTER filtering)
        testSubjects: Subject IDs for test set
        transform: Optional transform
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 64,
        stride: int = 8,
        downsample: bool = True,
        applyFilter: bool = True,  # NEW: Apply low-pass filter
        filterCutoff: float = 12.0,  # NEW: 12Hz cutoff
        normalize: bool = True,
        testSubjects: List[int] = [9, 10],
        transform=None
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        self.stride = stride if split == 'train' else windowSize  # No overlap for test
        self.downsample = downsample
        self.applyFilter = applyFilter
        self.filterCutoff = filterCutoff
        self.normalize = normalize
        self.testSubjects = testSubjects
        self.transform = transform
        
        # Load data (with filtering applied BEFORE normalization)
        self.windows, self.labels = self._loadData()
        
        # Normalization stats (computed AFTER filtering)
        if self.normalize and len(self.windows) > 0:
            self.mean = self.windows.mean(axis=(0, 1), keepdims=True)
            self.std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
            self.windows = (self.windows - self.mean) / self.std
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Daphnet data from txt files with signal rescue preprocessing."""
        
        # Data is in dataset subfolder
        datasetDir = self.root / "dataset"
        if not datasetDir.exists():
            datasetDir = self.root / "dataset_fog_release" / "dataset"
        if not datasetDir.exists():
            datasetDir = self.root
        
        txtFiles = list(datasetDir.glob("S*.txt"))
        if not txtFiles:
            txtFiles = list(self.root.rglob("S*.txt"))
        
        if not txtFiles:
            print("   Warning: Using synthetic Daphnet data for testing")
            return self._generateSyntheticData()
        
        allWindows = []
        allLabels = []
        
        for txtFile in sorted(txtFiles):
            try:
                # Extract subject ID from filename (e.g., S01R01.txt -> 1)
                subjectId = int(txtFile.stem[1:3])
                
                # Check if this subject belongs to current split
                inTest = subjectId in self.testSubjects
                if (self.split == 'train' and inTest) or (self.split == 'test' and not inTest):
                    continue
                
                # Load data: 11 columns
                # [time, ankle_x, ankle_y, ankle_z, thigh_x, thigh_y, thigh_z, 
                #  trunk_x, trunk_y, trunk_z, annotation]
                data = np.loadtxt(txtFile)
                
                # Extract sensor data (columns 1-9) and labels (column 10)
                sensorData = data[:, 1:10].astype(np.float32)  # 9 channels
                labels = data[:, 10].astype(int)  # Last column is annotation
                
                # Remove "not worn" / experiment not started samples (label 0)
                validMask = labels > 0
                sensorData = sensorData[validMask]
                labels = labels[validMask]
                
                if len(labels) == 0:
                    continue
                
                # Map labels: 1 (Walk/Stand) -> 0, 2 (Freeze) -> 1
                labels = labels - 1  # Now: 0 = Walk, 1 = Freeze
                
                # ========================================
                # SIGNAL RESCUE: Apply low-pass filter BEFORE downsampling
                # ========================================
                if self.applyFilter and len(sensorData) >= 13:
                    # Apply 4th order Butterworth low-pass filter at 12Hz
                    # This removes 20-30Hz sensor jitter while preserving 3-8Hz Freeze band
                    sensorData = applyLowpassFilter(
                        sensorData, 
                        cutoff=self.filterCutoff, 
                        fs=64.0,  # Original sampling rate
                        order=4
                    )
                
                # Downsample from 64Hz to 32Hz (AFTER filtering to avoid aliasing)
                if self.downsample:
                    sensorData = sensorData[::2]
                    labels = labels[::2]
                
                # Segment into windows
                windows, windowLabels = self._segmentWindowsFromSubject(sensorData, labels)
                
                if len(windows) > 0:
                    allWindows.append(windows)
                    allLabels.append(windowLabels)
                    
            except Exception as e:
                print(f"Warning: Error loading {txtFile.name}: {e}")
        
        if not allWindows:
            return self._generateSyntheticData()
        
        return np.vstack(allWindows), np.concatenate(allLabels)
    
    def _generateSyntheticData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for testing when real data unavailable."""
        np.random.seed(42 if self.split == 'train' else 43)
        
        nWindows = 2000 if self.split == 'train' else 500
        nChannels = 9
        
        windows = np.random.randn(nWindows, self.windowSize, nChannels).astype(np.float32)
        # Simulate imbalanced data: 90% Walk (0), 10% Freeze (1)
        labels = (np.random.rand(nWindows) < 0.1).astype(np.int64)
        
        return windows, labels
    
    def _segmentWindowsFromSubject(
        self,
        sensorData: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment subject data into windows."""
        windows = []
        windowLabels = []
        
        for start in range(0, len(sensorData) - self.windowSize + 1, self.stride):
            end = start + self.windowSize
            windowLabelArr = labels[start:end]
            
            # Majority label
            uniqueLabels, counts = np.unique(windowLabelArr, return_counts=True)
            majorityLabel = uniqueLabels[counts.argmax()]
            
            # Use window if label is consistent (>70%)
            if (windowLabelArr == majorityLabel).mean() >= 0.7:
                windows.append(sensorData[start:end])
                windowLabels.append(int(majorityLabel))
        
        if not windows:
            return np.array([]), np.array([])
        
        return np.array(windows, dtype=np.float32), np.array(windowLabels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window = torch.from_numpy(self.windows[idx].copy())
        label = int(self.labels[idx])
        
        if self.transform is not None:
            window = self.transform(window)
        
        return window, label
    
    @property
    def numClasses(self) -> int:
        return NUM_CLASSES
    
    @property
    def numChannels(self) -> int:
        if len(self.windows) > 0:
            return self.windows.shape[2]
        return 9


def computeDaphnetClassWeights(labels: np.ndarray, numClasses: int = 2, aggressive: bool = True) -> torch.Tensor:
    """
    Compute class weights for imbalanced Daphnet dataset.
    
    VARIANCE STABILIZER: Aggressive weighting for Freeze detection
    
    Strategy (aggressive=True):
    - Freeze (Class 1): Weight = 15.0
    - Walk/Stand (Class 0): Weight = 1.0
    
    Goal: Force the model to accept False Positives rather than miss Freezes.
    High Recall is scientifically preferred for Parkinson's detection.
    
    Args:
        labels: Array of class labels
        numClasses: Number of classes (2 for Daphnet)
        aggressive: If True, use 15:1 weighting. If False, use inverse frequency.
    
    Returns:
        Tensor of class weights for CrossEntropyLoss
    """
    if aggressive:
        # AGGRESSIVE WEIGHTING: 15:1 ratio for Freeze vs Walk
        # This forces the model to prioritize Freeze detection
        weights = np.array([1.0, 15.0], dtype=np.float32)
    else:
        # Standard inverse frequency weighting
        classCounts = np.bincount(labels.astype(int), minlength=numClasses)
        totalSamples = len(labels)
        
        weights = np.zeros(numClasses, dtype=np.float32)
        for i in range(numClasses):
            if classCounts[i] > 0:
                weights[i] = totalSamples / (numClasses * classCounts[i])
            else:
                weights[i] = 1.0
        
        # Normalize
        weights = weights / weights.mean()
    
    return torch.from_numpy(weights)


def getBalancedSampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler for balanced batches.
    
    Forces 50/50 class balance during training (crucial for Freeze detection).
    """
    classCounts = np.bincount(labels.astype(int))
    weights = 1.0 / classCounts[labels]
    weights = torch.DoubleTensor(weights)
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def getDaphnetLoaders(
    root: str = "./datasets/Daphnet",
    batchSize: int = 32,
    windowSize: int = 64,
    stride: int = 8,
    numWorkers: int = 0,
    useBalancedSampler: bool = True,
    applyFilter: bool = True,  # NEW: Apply low-pass filter
    filterCutoff: float = 12.0,  # NEW: 12Hz cutoff
    returnWeights: bool = False,
    aggressiveWeights: bool = True  # NEW: Use 15:1 weighting
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Daphnet train/test data loaders with Signal Rescue preprocessing.
    
    Signal Rescue Pipeline:
    1. Apply 4th order Butterworth low-pass filter (12Hz cutoff)
    2. Downsample from 64Hz to 32Hz (after filtering)
    3. Standardize (Z-Score normalization)
    
    Args:
        root: Path to Daphnet dataset
        batchSize: Batch size (small recommended: 16-32)
        windowSize: Sliding window size (64 = 2s at 32Hz)
        stride: Window stride (8 = 87% overlap for train)
        numWorkers: DataLoader workers
        useBalancedSampler: Use WeightedRandomSampler for 50/50 class balance
        applyFilter: Apply Butterworth low-pass filter (default: True)
        filterCutoff: Low-pass filter cutoff frequency (default: 12Hz)
        returnWeights: If True, returns (trainLoader, testLoader, classWeights)
        aggressiveWeights: If True, use 15:1 weighting for Freeze class
    """
    
    trainDataset = DaphnetDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        applyFilter=applyFilter,
        filterCutoff=filterCutoff,
        normalize=True
    )
    
    testDataset = DaphnetDataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=windowSize,  # No overlap for test
        applyFilter=applyFilter,
        filterCutoff=filterCutoff,
        normalize=False
    )
    
    # Apply train normalization to test (AFTER filtering)
    if len(trainDataset.windows) > 0 and len(testDataset.windows) > 0:
        testDataset.mean = trainDataset.mean
        testDataset.std = trainDataset.std
        testDataset.windows = (testDataset.windows - trainDataset.mean) / trainDataset.std
        testDataset.normalize = True
    
    # Compute class weights (aggressive 15:1 for Freeze detection)
    classWeights = None
    if len(trainDataset.labels) > 0:
        classWeights = computeDaphnetClassWeights(
            trainDataset.labels, 
            numClasses=NUM_CLASSES,
            aggressive=aggressiveWeights
        )
    
    # Use balanced sampler for training (50/50 Freeze/Walk in each batch)
    sampler = None
    shuffle = True
    if useBalancedSampler and len(trainDataset.labels) > 0:
        sampler = getBalancedSampler(trainDataset.labels)
        shuffle = False  # Sampler handles this
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=numWorkers,
        pin_memory=True
    )
    
    testLoader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True
    )
    
    if returnWeights:
        return trainLoader, testLoader, classWeights
    return trainLoader, testLoader


def getDatasetInfo() -> Dict:
    """Get Daphnet dataset information."""
    return {
        'name': 'Daphnet',
        'numClasses': NUM_CLASSES,
        'activities': {0: 'walk_stand', 1: 'freeze'},
        'subjects': 10,
        'samplingRate': 32,  # After downsampling
        'originalSamplingRate': 64,
        'channels': 9,
        'windowSize': 64,
        'stride': 8,
        'filterCutoff': 12.0,  # Butterworth low-pass
        'filterOrder': 4,
        'freezeBand': '3-8 Hz',
        'classWeights': '15:1 (Freeze:Walk)'
    }


# ============================================================================
# DAPHNET TRAINING RECIPE
# ============================================================================

def getDaphnetTrainingConfig() -> Dict:
    """
    Get the recommended training configuration for Daphnet.
    
    VARIANCE STABILIZER: Training recipe to eliminate ±7% variance
    and force convergence on the minority Freeze class.
    
    Returns:
        Dict with training hyperparameters
    """
    return {
        # Learning Rate Schedule (The "Warmup")
        'lr': 1e-3,                # Peak LR after warmup
        'lrMin': 1e-5,             # Final LR after cosine annealing
        'lrWarmupStart': 1e-6,     # Starting LR for warmup
        'warmupEpochs': 10,        # Linear warmup epochs
        
        # Regularization
        'weightDecay': 0.05,       # High decay to focus on low-freq movements
        'dropout': 0.0,            # No dropout (data is already regularized)
        'labelSmoothing': 0.0,     # OFF - sharp binary decisions needed
        
        # Loss & Sampling
        'classWeights': [1.0, 15.0],  # Aggressive Freeze upweighting
        'useBalancedSampler': True,    # 50/50 batch balance
        
        # Batch Size
        'batchSize': 32,           # Small batches for better gradient estimates
        
        # Training Control
        'gradientClipNorm': 1.0,   # CRITICAL: Prevents gradient explosion
        'epochs': 200,
        'patience': 30,            # INCREASED: Model peaks at epoch 10-15, needs time to confirm
        
        # Signal Preprocessing
        'applyFilter': True,       # Butterworth low-pass
        'filterCutoff': 12.0,      # 12Hz cutoff
    }


if __name__ == '__main__':
    print("Daphnet Dataset Info:")
    info = getDatasetInfo()
    for k, v in info.items():
        print(f"  {k}: {v}")
