"""
PAMAP2 Dataset Loader

PAMAP2: Physical Activity Monitoring for Aging People
- 9 subjects performing 12 activities
- 3 IMUs (hand, chest, ankle) + heart rate
- ~18 sensor channels (can use up to 52)
- Complex multimodal sensor fusion benchmark

This proves BabyMamba handles high-dimensional inputs.

Signal Rescue Strategy:
- Impact Spikes: Activities like Running and Jumping create sharp acceleration spikes
- These spikes become high-frequency artifacts that dominate the signal
- 10Hz Low-Pass Filter: Converts impact spikes into rhythmic curves
- Robust Scaling: IQR-based normalization prevents Running outliers from compressing
  static activities (lying, sitting) into a flat line
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from scipy.signal import butter, filtfilt


# PAMAP2 Activity Labels
PAMAP2_ACTIVITIES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping"
}

# Standard 12 activities for classification
ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
ACTIVITY_TO_IDX = {aid: idx for idx, aid in enumerate(ACTIVITY_IDS)}

# Constants
SAMPLING_RATE = 100  # Hz
NUM_CLASSES = 12


def butterworthLowpass(cutoff: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth low-pass filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order (default 4)
        
    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs
    normalCutoff = cutoff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog=False)
    return b, a


def applyLowpassFilter(data: np.ndarray, cutoff: float = 10.0, fs: float = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to remove impact spikes.
    
    PAMAP2 Signal Rescue:
    - Running/Jumping create sharp acceleration spikes (high frequency)
    - 10Hz cutoff preserves activity rhythm while smoothing impact artifacts
    - Human movement patterns are typically under 10Hz
    
    Args:
        data: Input data of shape (N, C) where N is samples, C is channels
        cutoff: Cutoff frequency in Hz (default 10Hz for PAMAP2)
        fs: Sampling frequency in Hz
        order: Filter order (default 4)
        
    Returns:
        Filtered data with same shape as input (preserves float32 dtype)
    """
    b, a = butterworthLowpass(cutoff, fs, order)
    
    # Apply filter to each channel
    filteredData = np.zeros_like(data)
    for c in range(data.shape[1]):
        # Use filtfilt for zero-phase filtering (no time delay)
        filteredData[:, c] = filtfilt(b, a, data[:, c])
    
    # Ensure float32 dtype (filtfilt may return float64)
    return filteredData.astype(np.float32)


def robustScale(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Robust Scaling (IQR-based) instead of Z-score.
    
    Why Robust Scaling for PAMAP2:
    - Running/Jumping have extreme outliers (impact spikes)
    - Z-score: (x - mean) / std -> outliers inflate std
    - Result: Static activities (lying, sitting) get compressed to near-zero
    - Robust: (x - median) / IQR -> outliers don't affect scaling
    - Preserves the separation between static and dynamic activities
    
    Args:
        data: Input data of shape (N, C) or (W, T, C)
        
    Returns:
        Scaled data, median, IQR (for applying same scale to test data)
    """
    originalShape = data.shape
    
    # Flatten to 2D for scaling computation
    if len(originalShape) == 3:
        # (W, T, C) -> (W*T, C)
        data = data.reshape(-1, originalShape[-1])
    
    # Compute robust statistics per channel (cast to float32)
    q1 = np.percentile(data, 25, axis=0, keepdims=True).astype(np.float32)
    q3 = np.percentile(data, 75, axis=0, keepdims=True).astype(np.float32)
    median = np.median(data, axis=0, keepdims=True).astype(np.float32)
    iqr = (q3 - q1 + 1e-8).astype(np.float32)  # Avoid division by zero
    
    # Apply robust scaling (ensure float32)
    scaledData = ((data - median) / iqr).astype(np.float32)
    
    # Restore original shape
    if len(originalShape) == 3:
        scaledData = scaledData.reshape(originalShape)
    
    return scaledData, median.squeeze(), iqr.squeeze()


class Pamap2Dataset(Dataset):
    """
    PAMAP2 Physical Activity Monitoring Dataset.
    
    Features per sample:
    - Heart rate (1 channel)
    - 3 IMUs with 3D acc + 3D gyro + 3D mag = 9 channels each
    - Total: 1 + 9*3 = 28 channels (or 52 with all features)
    
    We use a sliding window approach for sequence modeling.
    
    Args:
        root: Path to PAMAP2_Dataset folder
        split: 'train' or 'test'
        windowSize: Sliding window size (default 128 = 1.28s at 100Hz)
        stride: Window stride (default 64 = 50% overlap)
        channels: 'compact' (18ch), 'full' (52ch), or list of indices
        normalize: Whether to normalize data
        testSubjects: Subject IDs for test set (default: [105, 106])
        transform: Optional transform
        applyFilter: Apply 10Hz Butterworth low-pass filter (Signal Rescue)
        filterCutoff: Filter cutoff frequency in Hz (default 10Hz)
        useRobustScaling: Use IQR-based robust scaling instead of Z-score
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 128,
        stride: int = 64,
        channels: str = 'compact',
        normalize: bool = True,
        testSubjects: List[int] = [105, 106],
        transform = None,
        applyFilter: bool = True,
        filterCutoff: float = 10.0,
        useRobustScaling: bool = True
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        self.stride = stride
        self.normalize = normalize
        self.testSubjects = testSubjects
        self.transform = transform
        self.applyFilter = applyFilter
        self.filterCutoff = filterCutoff
        self.useRobustScaling = useRobustScaling
        
        # Normalization stats (will be set during loading)
        self.median = None
        self.iqr = None
        self.mean = None
        self.std = None
        
        # Channel selection
        if channels == 'compact':
            # Heart rate + 3 IMUs (acc only) = 1 + 3*3 = 10 channels
            # Actually let's use acc + gyro = 1 + 3*6 = 19 channels
            self.channelIndices = self._getCompactChannels()
        elif channels == 'full':
            self.channelIndices = None  # Use all
        elif isinstance(channels, list):
            self.channelIndices = channels
        else:
            self.channelIndices = self._getCompactChannels()
        
        # Load data
        self.windows, self.labels = self._loadData()
        
        # Normalization: Robust Scaling or Z-score
        if self.normalize and len(self.windows) > 0:
            if self.useRobustScaling:
                # Robust scaling (IQR-based) - prevents Running outliers from 
                # compressing static activities
                self.windows, self.median, self.iqr = robustScale(self.windows)
                print(f"   Applied Robust Scaling (IQR-based)")
            else:
                # Standard Z-score normalization
                self.mean = self.windows.mean(axis=(0, 1), keepdims=True)
                self.std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
                self.windows = (self.windows - self.mean) / self.std
    
    def _getCompactChannels(self) -> List[int]:
        """
        Get compact channel indices (18 channels).
        
        Data layout per row:
        0: timestamp
        1: activityID
        2: heart rate
        3-19: IMU hand (temp, 3D acc 16g, 3D acc 6g, 3D gyro, 3D mag, orientation)
        20-36: IMU chest
        37-53: IMU ankle
        
        Compact: heart rate + 3D acc (16g) + 3D gyro from each IMU
        """
        channels = [0]  # heart rate (index 0 after removing timestamp/label)
        
        # For each IMU: 3D acc (indices 1-3) + 3D gyro (indices 7-9)
        for imuStart in [1, 18, 35]:  # Start of each IMU block
            # 3D accelerometer (16g scale)
            channels.extend([imuStart + i for i in range(3)])
            # 3D gyroscope
            channels.extend([imuStart + 6 + i for i in range(3)])
        
        return channels  # 1 + 6*3 = 19 channels
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and segment data into windows."""
        
        # Try cleaned data first
        cleanedDir = self.root / "Cleaned"
        if cleanedDir.exists():
            return self._loadCleanedData(cleanedDir)
        
        # Otherwise load raw protocol files
        protocolDir = self.root / "Protocol"
        if not protocolDir.exists():
            # Try parent folder structure
            protocolDir = self.root.parent / "PAMAP2_Dataset" / "Protocol"
        
        if not protocolDir.exists():
            raise FileNotFoundError(
                f"PAMAP2 data not found at {self.root}. "
                f"Run downloadBig4.py first."
            )
        
        return self._loadRawData(protocolDir)
    
    def _loadCleanedData(self, cleanedDir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-cleaned numpy arrays."""
        X = np.load(cleanedDir / "X.npy")
        y = np.load(cleanedDir / "y.npy")
        subjects = np.load(cleanedDir / "subjects.npy")
        
        # Split by subject
        if self.split == 'train':
            mask = ~np.isin(subjects, self.testSubjects)
        else:
            mask = np.isin(subjects, self.testSubjects)
        
        X = X[mask]
        y = y[mask]
        
        # Select channels
        if self.channelIndices is not None:
            validChannels = [c for c in self.channelIndices if c < X.shape[1]]
            X = X[:, validChannels]
        
        # Create windows
        return self._segmentWindows(X, y)
    
    def _loadRawData(self, protocolDir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw .dat files and process."""
        
        allWindows = []
        allLabels = []
        
        for datFile in sorted(protocolDir.glob("subject*.dat")):
            subjectId = int(datFile.stem.replace("subject", ""))
            
            # Check if this subject belongs to current split
            inTest = subjectId in self.testSubjects
            if (self.split == 'train' and inTest) or (self.split == 'test' and not inTest):
                continue
            
            try:
                data = np.loadtxt(datFile)
                
                # Remove transient activity (ID 0)
                validMask = data[:, 1] > 0
                # Only keep known activities
                validMask &= np.isin(data[:, 1], ACTIVITY_IDS)
                data = data[validMask]
                
                if len(data) == 0:
                    continue
                
                # Extract labels and sensor data
                labels = data[:, 1].astype(int)
                sensorData = data[:, 2:]  # Remove timestamp and label
                
                # Handle NaNs
                nanMask = np.isnan(sensorData)
                if nanMask.any():
                    for col in range(sensorData.shape[1]):
                        colNans = np.isnan(sensorData[:, col])
                        if colNans.any():
                            validIdx = np.where(~colNans)[0]
                            if len(validIdx) > 0:
                                sensorData[colNans, col] = np.interp(
                                    np.where(colNans)[0],
                                    validIdx,
                                    sensorData[validIdx, col]
                                )
                            else:
                                sensorData[:, col] = 0
                
                # Select channels
                if self.channelIndices is not None:
                    validChannels = [c for c in self.channelIndices if c < sensorData.shape[1]]
                    sensorData = sensorData[:, validChannels]
                
                # Apply Signal Rescue: 10Hz Butterworth low-pass filter
                # Converts impact spikes (Running, Jumping) into rhythmic curves
                if self.applyFilter and len(sensorData) > 30:  # Need enough samples for filter
                    sensorData = applyLowpassFilter(
                        sensorData, cutoff=self.filterCutoff, fs=SAMPLING_RATE
                    )
                
                # Segment into windows
                windows, windowLabels = self._segmentWindowsFromSubject(sensorData, labels)
                
                if len(windows) > 0:
                    allWindows.append(windows)
                    allLabels.append(windowLabels)
                    
            except Exception as e:
                print(f"Warning: Error loading {datFile.name}: {e}")
        
        if not allWindows:
            return np.array([]), np.array([])
        
        return np.vstack(allWindows), np.concatenate(allLabels)
    
    def _segmentWindows(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment continuous data into windows."""
        windows = []
        labels = []
        
        for start in range(0, len(X) - self.windowSize + 1, self.stride):
            end = start + self.windowSize
            windowLabels = y[start:end]
            
            # Use majority label for window
            majorityLabel = np.bincount(windowLabels).argmax()
            
            # Only use window if label is consistent (>80%)
            labelConsistency = (windowLabels == majorityLabel).mean()
            if labelConsistency >= 0.8:
                windows.append(X[start:end])
                # Map activity ID to class index
                if majorityLabel in ACTIVITY_TO_IDX:
                    labels.append(ACTIVITY_TO_IDX[majorityLabel])
        
        if not windows:
            return np.array([]), np.array([])
        
        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)
    
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
            
            # Check consistency
            if (windowLabelArr == majorityLabel).mean() >= 0.8:
                if majorityLabel in ACTIVITY_TO_IDX:
                    windows.append(sensorData[start:end])
                    windowLabels.append(ACTIVITY_TO_IDX[majorityLabel])
        
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
        return len(ACTIVITY_IDS)
    
    @property
    def numChannels(self) -> int:
        if len(self.windows) > 0:
            return self.windows.shape[2]
        return 19  # Default compact channels
    
    @property
    def inputShape(self) -> Tuple[int, int]:
        return (self.windowSize, self.numChannels)


def computeClassWeights(labels: np.ndarray, numClasses: int = 12) -> torch.Tensor:
    """
    Compute class weights for imbalanced PAMAP2 dataset.
    
    Uses inverse frequency weighting: weight = total_samples / (num_classes * class_count)
    This penalizes the model more for misclassifying rare classes.
    
    Args:
        labels: Array of class labels
        numClasses: Number of classes
        
    Returns:
        Tensor of class weights for CrossEntropyLoss
    """
    classCounts = np.bincount(labels.astype(int), minlength=numClasses)
    totalSamples = len(labels)
    
    # Inverse frequency weighting (sklearn's balanced strategy)
    weights = np.zeros(numClasses, dtype=np.float32)
    for i in range(numClasses):
        if classCounts[i] > 0:
            weights[i] = totalSamples / (numClasses * classCounts[i])
        else:
            weights[i] = 1.0  # Default weight for missing classes
    
    # Normalize weights to have mean of 1.0
    weights = weights / weights.mean()
    
    return torch.from_numpy(weights)


def getPamap2Loaders(
    root: str = "./datasets/PAMAP2_Dataset",
    batchSize: int = 64,
    windowSize: int = 128,
    stride: int = 64,
    channels: str = 'compact',
    numWorkers: int = 0,
    testSubjects: List[int] = [105, 106],
    returnWeights: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Get PAMAP2 train/test data loaders.
    
    Args:
        root: Path to PAMAP2 dataset
        batchSize: Batch size
        windowSize: Sliding window size
        stride: Window stride
        channels: 'compact' (18ch) or 'full' (52ch)
        numWorkers: DataLoader workers
        testSubjects: Subject IDs for test set
        returnWeights: If True, returns (trainLoader, testLoader, classWeights)
        
    Returns:
        (trainLoader, testLoader) or (trainLoader, testLoader, classWeights)
    """
    
    # Load train with normalization
    trainDataset = Pamap2Dataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        channels=channels,
        normalize=True,  # Train computes its own stats
        testSubjects=testSubjects
    )
    
    # Load test WITHOUT normalization (we'll apply train stats)
    testDataset = Pamap2Dataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=stride // 2,  # More overlap for test
        channels=channels,
        normalize=False,  # Don't normalize yet
        testSubjects=testSubjects
    )
    
    # Apply train normalization stats to test data
    # Handle both robust scaling (median/iqr) and z-score (mean/std)
    if len(trainDataset.windows) > 0 and len(testDataset.windows) > 0:
        if trainDataset.useRobustScaling and trainDataset.median is not None:
            # Robust scaling: (x - median) / iqr
            testDataset.median = trainDataset.median
            testDataset.iqr = trainDataset.iqr
            testDataset.windows = (testDataset.windows - trainDataset.median) / trainDataset.iqr
            testDataset.normalize = True
            testDataset.useRobustScaling = True
        elif trainDataset.mean is not None:
            # Z-score: (x - mean) / std
            testDataset.mean = trainDataset.mean
            testDataset.std = trainDataset.std
            testDataset.windows = (testDataset.windows - trainDataset.mean) / trainDataset.std
            testDataset.normalize = True
    
    # Compute class weights for imbalanced dataset
    classWeights = None
    if len(trainDataset.labels) > 0:
        classWeights = computeClassWeights(trainDataset.labels, numClasses=len(ACTIVITY_IDS))
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
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
    """Get PAMAP2 dataset information."""
    return {
        'name': 'PAMAP2',
        'numClasses': NUM_CLASSES,
        'activities': PAMAP2_ACTIVITIES,
        'activityIds': ACTIVITY_IDS,
        'subjects': 9,
        'samplingRate': SAMPLING_RATE,
        'sensors': ['heart_rate', 'imu_hand', 'imu_chest', 'imu_ankle'],
        'channelsCompact': 19,
        'channelsFull': 52
    }


def getPamap2TrainingConfig() -> Dict:
    """
    Get PAMAP2-specific training configuration.
    
    PAMAP2 "Shock Absorber" Recipe:
    - Signal Rescue: 10Hz low-pass filter (converts impact spikes to rhythmic curves)
    - Robust Scaling: IQR-based (prevents Running outliers from compressing static activities)
    - Gradient Clipping 1.0 (stabilizes training with extreme dynamics)
    - Weight Decay 0.01 (regularization for 12-class problem)
    
    Returns:
        Dictionary with recommended training hyperparameters
    """
    return {
        # Signal preprocessing
        'applyFilter': True,
        'filterCutoff': 10.0,  # Hz - smooths impact spikes
        'useRobustScaling': True,  # IQR-based, not Z-score
        
        # Loss configuration
        'labelSmoothing': 0.0,  # PAMAP2 has clear activity boundaries
        'useClassWeights': True,  # Apply computed class weights
        
        # Optimizer configuration
        'lr': 1e-3,
        'weightDecay': 0.01,
        'gradClip': 1.0,  # Stabilize training with extreme dynamics
        
        # Scheduler: CosineAnnealing
        'scheduler': 'cosine',
        'warmupEpochs': 5,
        'minLr': 1e-6,
        
        # Training epochs
        'epochs': 100,
        'patience': 20,
        
        # Architecture hints
        'hiddenDim': 64,
        'numLayers': 2,
        'dropout': 0.2,
    }


if __name__ == '__main__':
    # Quick test
    print("PAMAP2 Dataset Info:")
    info = getDatasetInfo()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # Try loading
    try:
        trainLoader, testLoader = getPamap2Loaders(
            root="./datasets/PAMAP2_Dataset",
            batchSize=32
        )
        
        print(f"\nTrain batches: {len(trainLoader)}")
        print(f"Test batches: {len(testLoader)}")
        
        # Check a batch
        for x, y in trainLoader:
            print(f"\nBatch shape: {x.shape}")  # [B, T, C]
            print(f"Labels shape: {y.shape}")
            print(f"Label range: {y.min()} - {y.max()}")
            break
            
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Run: python scripts/downloadBig4.py --dataset pamap2")
