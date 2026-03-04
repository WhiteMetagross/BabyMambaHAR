"""
Metrics for BabyMamba Evaluation

Includes accuracy, F1-score, confusion matrix, and tracking utilities.
"""

from typing import Dict, List, Optional
import torch
import numpy as np
from collections import defaultdict


class Accuracy:
    """
    Classification accuracy metric with top-k support.
    """
    
    def __init__(self, topk: tuple = (1,)):
        self.topk = topk
        self.reset()
    
    def reset(self):
        self.correct = {k: 0 for k in self.topk}
        self.total = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update with batch predictions.
        
        Args:
            preds: Logits [B, C]
            targets: Labels [B]
        """
        # Handle sequence outputs
        if preds.dim() == 3:
            preds = preds.mean(dim=1)
        
        batchSize = targets.size(0)
        self.total += batchSize
        
        maxk = max(self.topk)
        _, predTopk = preds.topk(maxk, dim=-1, largest=True, sorted=True)
        predTopk = predTopk.t()  # [K, B]
        correct = predTopk.eq(targets.view(1, -1).expand_as(predTopk))
        
        for k in self.topk:
            self.correct[k] += correct[:k].reshape(-1).float().sum().item()
    
    def compute(self) -> Dict[str, float]:
        """Compute accuracy for each k."""
        return {
            f'acc@{k}': self.correct[k] / max(1, self.total) * 100
            for k in self.topk
        }
    
    @property
    def value(self) -> float:
        """Get top-1 accuracy."""
        return self.correct[1] / max(1, self.total) * 100


class F1Score:
    """
    F1 Score metric (macro/micro/weighted).
    """
    
    def __init__(self, numClasses: int, average: str = 'macro'):
        self.numClasses = numClasses
        self.average = average
        self.reset()
    
    def reset(self):
        self.truePositives = torch.zeros(self.numClasses)
        self.falsePositives = torch.zeros(self.numClasses)
        self.falseNegatives = torch.zeros(self.numClasses)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update with batch predictions.
        
        Args:
            preds: Logits [B, C]
            targets: Labels [B]
        """
        preds = preds.argmax(dim=-1)
        
        for c in range(self.numClasses):
            predC = (preds == c)
            targetC = (targets == c)
            
            self.truePositives[c] += (predC & targetC).sum().item()
            self.falsePositives[c] += (predC & ~targetC).sum().item()
            self.falseNegatives[c] += (~predC & targetC).sum().item()
    
    def compute(self) -> Dict[str, float]:
        """Compute F1 scores."""
        precision = self.truePositives / (self.truePositives + self.falsePositives + 1e-8)
        recall = self.truePositives / (self.truePositives + self.falseNegatives + 1e-8)
        f1PerClass = 2 * precision * recall / (precision + recall + 1e-8)
        
        if self.average == 'macro':
            f1 = f1PerClass.mean().item()
        elif self.average == 'micro':
            totalTp = self.truePositives.sum()
            totalFp = self.falsePositives.sum()
            totalFn = self.falseNegatives.sum()
            precMicro = totalTp / (totalTp + totalFp + 1e-8)
            recMicro = totalTp / (totalTp + totalFn + 1e-8)
            f1 = 2 * precMicro * recMicro / (precMicro + recMicro + 1e-8)
            f1 = f1.item()
        else:  # weighted
            support = self.truePositives + self.falseNegatives
            f1 = (f1PerClass * support).sum() / (support.sum() + 1e-8)
            f1 = f1.item()
        
        return {
            'f1': f1 * 100,
            'precision': precision.mean().item() * 100,
            'recall': recall.mean().item() * 100,
            'f1PerClass': f1PerClass.tolist()
        }
    
    @property
    def value(self) -> float:
        """Get macro F1."""
        return self.compute()['f1']


class ConfusionMatrix:
    """
    Confusion matrix computation.
    """
    
    def __init__(self, numClasses: int, classNames: Optional[List[str]] = None):
        self.numClasses = numClasses
        self.classNames = classNames or [str(i) for i in range(numClasses)]
        self.reset()
    
    def reset(self):
        self.matrix = torch.zeros(self.numClasses, self.numClasses, dtype=torch.long)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix."""
        preds = preds.argmax(dim=-1)
        
        for pred, target in zip(preds, targets):
            self.matrix[target, pred] += 1
    
    def compute(self) -> np.ndarray:
        """Return confusion matrix as numpy array."""
        return self.matrix.numpy()
    
    def getNormalized(self) -> np.ndarray:
        """Return row-normalized (per-class recall) confusion matrix."""
        rowSums = self.matrix.sum(dim=1, keepdim=True).float()
        return (self.matrix.float() / (rowSums + 1e-8)).numpy()


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    @property
    def value(self) -> float:
        return self.avg


class MetricsTracker:
    """
    Track multiple metrics during training.
    """
    
    def __init__(self, numClasses: int):
        self.numClasses = numClasses
        self.reset()
    
    def reset(self):
        self.accuracy = Accuracy(topk=(1,))
        self.f1 = F1Score(self.numClasses)
        self.confMatrix = ConfusionMatrix(self.numClasses)
        self.loss = AverageMeter('loss')
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float = None):
        self.accuracy.update(preds, targets)
        self.f1.update(preds, targets)
        self.confMatrix.update(preds, targets)
        if loss is not None:
            self.loss.update(loss, targets.size(0))
    
    def compute(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy.value,
            **self.f1.compute(),
            'loss': self.loss.avg
        }
