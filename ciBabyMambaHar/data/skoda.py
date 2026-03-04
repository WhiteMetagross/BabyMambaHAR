"""
SKODA Mini Checkpoint Dataset Loader

SKODA: Automotive assembly line gesture recognition
- 1 subject performing 10 manipulative gestures + Null class
- 10 sensors on right arm, 3-axis accelerometer each = 30 channels
- Sampling rate: ~98Hz
- Challenge: High Null class proportion + industrial machine vibration

SIGNAL RESCUE STRATEGY:
- 5Hz Low-Pass Butterworth Filter (removes machine vibration 50-60Hz)
- Human arms cannot manipulate objects faster than 2-3Hz
- Filter prevents industrial vibration aliasing into gesture band
- Label Smoothing 0.1 (fuzzy gesture boundaries)

Label Mapping (from dataset documentation):
- 32: null class
- 48: write on notepad
- 49: open hood
- 50: close hood
- 51: check gaps on the front door
- 52: open left front door
- 53: close left front door
- 54: close both left doors
- 55: check trunk gaps
- 56: open and close trunk
- 57: check steering wheel

Training Strategy (Frozen Architecture):
- Window: 98 samples (~1 second at 98Hz)
- Stride: 24 samples (75% overlap) for training, 98 for test
- Filter: 5Hz Low-Pass Butterworth (kill machine vibration)
- Normalization: Channel-wise Z-Score (after filtering)
- Class Weights: Standard inverse frequency (SKODA is balanced enough)
- Label Smoothing: 0.1 (fuzzy gesture boundaries)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import scipy.io as sio
from scipy.signal import butter, filtfilt


# Original label values from the dataset
ORIGINAL_LABELS = {
    32: 0,   # null -> 0
    48: 1,   # write on notepad
    49: 2,   # open hood
    50: 3,   # close hood
    51: 4,   # check gaps front door
    52: 5,   # open left front door
    53: 6,   # close left front door
    54: 7,   # close both left doors
    55: 8,   # check trunk gaps
    56: 9,   # open and close trunk
    57: 10,  # check steering wheel
}

# SKODA Activity Labels (mapped to 0-10)
SKODA_ACTIVITIES = {
    0: "null",
    1: "write_on_notepad",
    2: "open_hood",
    3: "close_hood",
    4: "check_gaps_front",
    5: "open_left_front_door",
    6: "close_left_front_door",
    7: "close_both_left_doors",
    8: "check_trunk_gaps",
    9: "open_close_trunk",
    10: "check_steering_wheel"
}

NUM_CLASSES = 11
NUM_SENSORS = 10
NUM_CHANNELS = 30  # 10 sensors × 3 axes (calibrated only)
SAMPLING_RATE = 98  # Hz


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


def applyLowpassFilter(data: np.ndarray, cutoff: float = 5.0, fs: float = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to remove machine vibration.
    
    SKODA Signal Rescue:
    - Industrial sensors pick up machine vibration at 50-60Hz
    - Human arms cannot manipulate objects faster than 2-3Hz
    - 5Hz cutoff removes vibration while preserving gesture dynamics
    - Prevents aliasing into the gesture band (2-4Hz after Stride-4 tokenization)
    
    Args:
        data: Input data of shape (N, C) where N is samples, C is channels
        cutoff: Cutoff frequency in Hz (default 5Hz for SKODA)
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


class SkodaDataset(Dataset):
    """
    SKODA Mini Checkpoint Dataset.
    
    Uses right arm data only (standard benchmark configuration).
    
    Data format (from right_classall_clean.mat):
    - Column 1: label
    - Column 2+s*7: sensor id
    - Column 2+s*7+1: X acceleration calibrated
    - Column 2+s*7+2: Y acceleration calibrated
    - Column 2+s*7+3: Z acceleration calibrated
    - Column 2+s*7+4: X acceleration raw
    - Column 2+s*7+5: Y acceleration raw
    - Column 2+s*7+6: Z acceleration raw
    
    Args:
        root: Path to Skoda dataset folder
        split: 'train' or 'test'
        windowSize: Sliding window size (default 98 = ~1s at 98Hz)
        stride: Window stride (default 24 = 75% overlap for train)
        normalize: Whether to normalize data
        testRatio: Ratio for test split (default 0.2)
        transform: Optional transform
        applyFilter: Apply 5Hz Butterworth low-pass filter (Signal Rescue)
        filterCutoff: Filter cutoff frequency in Hz (default 5Hz)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        windowSize: int = 98,
        stride: int = 24,
        normalize: bool = True,
        testRatio: float = 0.2,
        transform=None,
        seed: int = 42,
        applyFilter: bool = True,
        filterCutoff: float = 5.0,
        allowOverlap: bool = True  # Use overlapping windows (75% overlap for train, none for test)
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.windowSize = windowSize
        # Overlapping windows for training (75% overlap), non-overlapping for test
        # TEMPORAL SPLIT within each class: first 80% for train, last 20% for test
        # This prevents data leakage from adjacent windows sharing 75% of their data
        if allowOverlap:
            self.stride = stride if split == 'train' else windowSize
        else:
            self.stride = windowSize  # Non-overlapping for both splits
        self.normalize = normalize
        self.testRatio = testRatio
        self.transform = transform
        self.seed = seed
        self.applyFilter = applyFilter
        self.filterCutoff = filterCutoff
        
        # Will be set during loading
        self.mean = None
        self.std = None
        
        # Load data
        self.windows, self.labels = self._loadData()
        
        # Normalization stats (channel-wise Z-Score)
        if self.normalize and len(self.windows) > 0:
            self.mean = self.windows.mean(axis=(0, 1), keepdims=True)
            self.std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
            self.windows = (self.windows - self.mean) / self.std
    
    def _findMatFile(self) -> Optional[Path]:
        """Find the right arm .mat file."""
        # Check various possible locations
        searchPaths = [
            self.root / "right_classall_clean.mat",
            self.root / "SkodaMiniCP_2015_08" / "right_classall_clean.mat",
        ]
        
        for path in searchPaths:
            if path.exists():
                return path
        
        # Search recursively
        for matFile in self.root.rglob("right_classall_clean.mat"):
            return matFile
        
        return None
    
    def _extractCalibratedAccel(self, rawData: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract calibrated acceleration data and labels from raw .mat data.
        
        Format: [label, (sensor_id, x_cal, y_cal, z_cal, x_raw, y_raw, z_raw) × 10 sensors]
        We want: just the calibrated x, y, z from each of the 10 sensors = 30 channels
        """
        labels = rawData[:, 0].astype(np.int64)
        
        # Extract calibrated accelerometer data for 10 sensors
        # Each sensor has 7 columns: [id, x_cal, y_cal, z_cal, x_raw, y_raw, z_raw]
        # Calibrated data is at columns: 1+s*7+1, 1+s*7+2, 1+s*7+3 for sensor s (0-9)
        channels = []
        for s in range(NUM_SENSORS):
            baseCol = 1 + s * 7  # Column 1 is first sensor block (0-indexed: col 1 = index 1)
            xCal = rawData[:, baseCol + 1]  # x_cal
            yCal = rawData[:, baseCol + 2]  # y_cal
            zCal = rawData[:, baseCol + 3]  # z_cal
            channels.extend([xCal, yCal, zCal])
        
        data = np.stack(channels, axis=1).astype(np.float32)
        
        return data, labels
    
    def _mapLabels(self, labels: np.ndarray) -> np.ndarray:
        """Map original labels (32, 48-57) to consecutive indices (0-10)."""
        mappedLabels = np.full_like(labels, -1)  # -1 for unknown labels
        
        for origLabel, mappedLabel in ORIGINAL_LABELS.items():
            mappedLabels[labels == origLabel] = mappedLabel
        
        return mappedLabels
    
    def _loadData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SKODA data from right_classall_clean.mat file."""
        
        matFile = self._findMatFile()
        
        if matFile is None:
            print(f"   Warning: SKODA right_classall_clean.mat not found in {self.root}")
            print(f"   Using synthetic data for testing pipeline")
            return self._generateSyntheticData()
        
        try:
            mat = sio.loadmat(str(matFile))
            
            # Find the data key
            dataKey = None
            for key in mat.keys():
                if 'right_classall_clean' in key.lower():
                    dataKey = key
                    break
            
            if dataKey is None:
                for key in mat.keys():
                    if not key.startswith('_'):
                        dataKey = key
                        break
            
            if dataKey is None:
                raise ValueError(f"Could not find data in {matFile}")
            
            rawData = mat[dataKey]
            print(f"   Loaded {matFile.name}: shape={rawData.shape}")
            
            # Extract calibrated accelerometer data (30 channels) and labels
            data, labels = self._extractCalibratedAccel(rawData)
            
            # Map original labels (32, 48-57) to 0-10
            labels = self._mapLabels(labels)
            
            # Filter out unlabeled data (any label not in our mapping -> -1)
            validMask = labels >= 0
            data = data[validMask]
            labels = labels[validMask]
            
            # Apply Signal Rescue: 5Hz Butterworth low-pass filter
            # Removes 50-60Hz machine vibration while preserving gesture dynamics
            if self.applyFilter:
                print(f"   Applying {self.filterCutoff}Hz low-pass filter (Signal Rescue)")
                data = applyLowpassFilter(data, cutoff=self.filterCutoff, fs=SAMPLING_RATE)
            
            print(f"   Data: {data.shape}, Labels unique: {np.unique(labels)}")
            
            # Report class distribution
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                actName = SKODA_ACTIVITIES.get(u, "unknown")
                print(f"      Class {u} ({actName}): {c} samples ({100*c/len(labels):.1f}%)")
            
        except Exception as e:
            print(f"   Warning: Error loading SKODA data: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Using synthetic data for testing pipeline")
            return self._generateSyntheticData()
        
        # FIXED: Window FIRST, then TEMPORAL split on windows (no shuffle)
        # SKODA data is organized sequentially by activity.
        # Within each class, we take the FIRST 80% of windows for train and
        # the LAST 20% for test - this ensures no data leakage from overlapping
        # windows (adjacent windows share 75% of their data).
        
        # Create windows from full data (maintains temporal contiguity per activity)
        allWindows, allLabels = self._segmentWindows(data, labels)
        
        if len(allLabels) == 0:
            return allWindows, allLabels
        
        # Temporal split: for each class, take FIRST 80% for train, LAST 20% for test
        # NO SHUFFLE - maintains temporal ordering to prevent leakage
        
        trainIndices, testIndices = [], []
        
        for classIdx in range(NUM_CLASSES):
            classMask = allLabels == classIdx
            classIndices = np.where(classMask)[0]
            
            if len(classIndices) == 0:
                continue
            
            # NO shuffle - keep temporal order within each class
            # This prevents adjacent overlapping windows from ending up in
            # both train and test sets (which would cause data leakage)
            
            nTrain = int(len(classIndices) * (1 - self.testRatio))
            trainIndices.extend(classIndices[:nTrain].tolist())
            testIndices.extend(classIndices[nTrain:].tolist())
        
        # Select train or test portion
        if self.split == 'train':
            indices = np.array(trainIndices)
        else:
            indices = np.array(testIndices)
        
        # Shuffle to randomize batch order (safe - only shuffles within train or test set)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        return allWindows[indices], allLabels[indices]
    
    def _generateSyntheticData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for testing when real data unavailable."""
        np.random.seed(self.seed if self.split == 'train' else self.seed + 1)
        
        nWindows = 2000 if self.split == 'train' else 500
        
        # Simulate imbalanced data: 50% Null (0), 50% gestures (1-10)
        windows = np.random.randn(nWindows, self.windowSize, NUM_CHANNELS).astype(np.float32)
        labels = np.zeros(nWindows, dtype=np.int64)
        labels[nWindows//2:] = np.random.randint(1, NUM_CLASSES, nWindows - nWindows//2)
        
        return windows, labels
    
    def _segmentWindows(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment continuous data into windows."""
        windows = []
        windowLabels = []
        
        for start in range(0, len(data) - self.windowSize + 1, self.stride):
            end = start + self.windowSize
            windowData = data[start:end]
            windowLabelArr = labels[start:end]
            
            # Majority label
            uniqueLabels, counts = np.unique(windowLabelArr, return_counts=True)
            majorityLabel = uniqueLabels[counts.argmax()]
            
            # Only use window if label is consistent (>70%)
            if (windowLabelArr == majorityLabel).mean() >= 0.7:
                windows.append(windowData)
                windowLabels.append(int(majorityLabel))
        
        if not windows:
            return np.array([]).reshape(0, self.windowSize, NUM_CHANNELS), np.array([], dtype=np.int64)
        
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
        return NUM_CHANNELS


def computeSkodaClassWeights(labels: np.ndarray, numClasses: int = 11) -> torch.Tensor:
    """
    Compute class weights for imbalanced SKODA dataset.
    
    Strategy: Down-weight Null class (0.5), up-weight gestures (1.5)
    """
    labels = np.asarray(labels).astype(int)
    classCounts = np.bincount(labels, minlength=numClasses)
    totalSamples = len(labels)
    
    weights = np.zeros(numClasses, dtype=np.float32)
    for i in range(numClasses):
        if classCounts[i] > 0:
            # Inverse frequency weighting
            weights[i] = totalSamples / (numClasses * classCounts[i])
        else:
            weights[i] = 1.0
    
    # Special adjustment: down-weight Null (class 0)
    weights[0] *= 0.5  # Reduce Null class weight
    weights[1:] *= 1.5  # Increase gesture weights
    
    # Normalize
    weights = weights / weights.mean()
    
    return torch.from_numpy(weights)


def getSkodaLoaders(
    root: str = "./datasets/Skoda",
    batchSize: int = 64,
    windowSize: int = 98,
    stride: int = 24,
    numWorkers: int = 0,
    returnWeights: bool = False,
    allowOverlap: bool = True  # Use overlapping windows (75% overlap for train)
) -> Tuple[DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Get SKODA train/test data loaders.
    
    Args:
        root: Path to SKODA dataset
        batchSize: Batch size
        windowSize: Sliding window size (98 = ~1s at 98Hz)
        stride: Window stride for training (24 = 75% overlap)
        numWorkers: DataLoader workers
        returnWeights: If True, returns (trainLoader, testLoader, classWeights)
        allowOverlap: If True (default), uses overlapping windows for training
    """
    
    trainDataset = SkodaDataset(
        root=root,
        split='train',
        windowSize=windowSize,
        stride=stride,
        normalize=True,
        allowOverlap=allowOverlap  # Pass through to prevent leakage
    )
    
    testDataset = SkodaDataset(
        root=root,
        split='test',
        windowSize=windowSize,
        stride=windowSize,  # Test always non-overlapping
        normalize=False,
        allowOverlap=allowOverlap
    )
    
    # Apply train normalization to test
    if len(trainDataset.windows) > 0 and len(testDataset.windows) > 0:
        testDataset.mean = trainDataset.mean
        testDataset.std = trainDataset.std
        testDataset.windows = (testDataset.windows - testDataset.mean) / testDataset.std
        testDataset.normalize = True
    
    # Compute class weights
    classWeights = None
    if len(trainDataset.labels) > 0:
        classWeights = computeSkodaClassWeights(trainDataset.labels, numClasses=NUM_CLASSES)
    
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
    return trainLoader, testLoader, None


def getDatasetInfo() -> Dict:
    """Get SKODA dataset information."""
    return {
        'name': 'SKODA',
        'numClasses': NUM_CLASSES,
        'activities': SKODA_ACTIVITIES,
        'subjects': 1,
        'samplingRate': SAMPLING_RATE,
        'channels': NUM_CHANNELS,
        'windowSize': 98,
        'stride': 24
    }


def getSkodaTrainingConfig() -> Dict:
    """
    Get SKODA-specific training configuration.
    
    SKODA "Speed Run" Recipe:
    - Signal Rescue: 5Hz low-pass filter (removes machine vibration 50-60Hz)
    - Label Smoothing 0.1 (fuzzy gesture boundaries in assembly line)
    - Standard class weights (SKODA is reasonably balanced)
    - Cosine decay LR with moderate warmup
    
    Returns:
        Dictionary with recommended training hyperparameters
    """
    return {
        # Signal preprocessing
        'applyFilter': True,
        'filterCutoff': 5.0,  # Hz - removes machine vibration
        
        # Loss configuration
        'labelSmoothing': 0.1,  # Handle fuzzy gesture boundaries
        'useClassWeights': True,  # Apply computed class weights
        
        # Optimizer configuration
        'lr': 1e-3,
        'weightDecay': 0.01,
        'gradClip': 1.0,
        
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
    print("SKODA Dataset Info:")
    info = getDatasetInfo()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\nLoading SKODA dataset...")
    trainLoader, testLoader, weights = getSkodaLoaders(returnWeights=True)
    print(f"Train batches: {len(trainLoader)}, Test batches: {len(testLoader)}")
    if weights is not None:
        print(f"Class weights: {weights}")
