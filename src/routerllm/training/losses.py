"""
Custom loss functions for router training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional


class InterIntraLoss(nn.Module):
    """
    Inter-Intra Loss implementation for router training

    This loss function combines:
    1. Inter-class loss: Maximizes distance between different classes
    2. Intra-class loss: Minimizes distance within the same class
    3. Standard classification loss
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        margin: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Inter-Intra Loss

        Args:
            num_classes: Number of classes
            alpha: Weight for classification loss
            beta: Weight for inter-class loss
            gamma: Weight for intra-class loss
            margin: Margin for inter-class separation
            logger: Logger instance
        """
        super(InterIntraLoss, self).__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.logger = logger or logging.getLogger(__name__)

        # Standard classification loss
        self.ce_loss = nn.CrossEntropyLoss()

        self.logger.info(
            f"InterIntraLoss initialized - Classes: {num_classes}, "
            f"Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}, Margin: {margin}"
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: True labels [batch_size]
            features: Feature representations [batch_size, feature_dim]

        Returns:
            Combined loss
        """
        batch_size = logits.size(0)
        device = logits.device

        # 1. Classification loss
        ce_loss = self.ce_loss(logits, labels)

        # 2. Intra-class loss (minimize distance within same class)
        intra_loss = self._compute_intra_class_loss(features, labels)

        # 3. Inter-class loss (maximize distance between different classes)
        inter_loss = self._compute_inter_class_loss(features, labels)

        # Combine losses
        total_loss = (
            self.alpha * ce_loss +
            self.beta * inter_loss +
            self.gamma * intra_loss
        )

        return total_loss

    def _compute_intra_class_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute intra-class loss (minimize variance within classes)

        Args:
            features: Feature representations [batch_size, feature_dim]
            labels: True labels [batch_size]

        Returns:
            Intra-class loss
        """
        batch_size = features.size(0)
        feature_dim = features.size(1)
        device = features.device

        # Compute class centroids
        centroids = []
        intra_losses = []

        for class_id in range(self.num_classes):
            # Get features for this class
            class_mask = (labels == class_id)
            class_features = features[class_mask]

            if class_features.size(0) > 0:
                # Compute centroid
                centroid = torch.mean(class_features, dim=0)
                centroids.append(centroid)

                # Compute distances to centroid
                distances = torch.norm(class_features - centroid.unsqueeze(0), dim=1)
                intra_loss = torch.mean(distances)
                intra_losses.append(intra_loss)

        # Average intra-class loss
        if intra_losses:
            return torch.mean(torch.stack(intra_losses))
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_inter_class_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute inter-class loss (maximize distance between class centroids)

        Args:
            features: Feature representations [batch_size, feature_dim]
            labels: True labels [batch_size]

        Returns:
            Inter-class loss
        """
        device = features.device

        # Compute class centroids
        centroids = []
        valid_classes = []

        for class_id in range(self.num_classes):
            class_mask = (labels == class_id)
            class_features = features[class_mask]

            if class_features.size(0) > 0:
                centroid = torch.mean(class_features, dim=0)
                centroids.append(centroid)
                valid_classes.append(class_id)

        if len(centroids) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stack centroids
        centroids = torch.stack(centroids)  # [num_valid_classes, feature_dim]

        # Compute pairwise distances between centroids
        inter_losses = []
        num_centroids = centroids.size(0)

        for i in range(num_centroids):
            for j in range(i + 1, num_centroids):
                distance = torch.norm(centroids[i] - centroids[j])
                # Use margin-based loss: max(0, margin - distance)
                inter_loss = F.relu(self.margin - distance)
                inter_losses.append(inter_loss)

        # Average inter-class loss
        if inter_losses:
            return torch.mean(torch.stack(inter_losses))
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def get_loss_components(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> dict:
        """
        Get individual loss components for analysis

        Args:
            logits: Model predictions
            labels: True labels
            features: Feature representations

        Returns:
            Dictionary with loss components
        """
        ce_loss = self.ce_loss(logits, labels)
        intra_loss = self._compute_intra_class_loss(features, labels)
        inter_loss = self._compute_inter_class_loss(features, labels)

        total_loss = (
            self.alpha * ce_loss +
            self.beta * inter_loss +
            self.gamma * intra_loss
        )

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'intra_loss': intra_loss,
            'inter_loss': inter_loss,
            'weighted_ce': self.alpha * ce_loss,
            'weighted_intra': self.gamma * intra_loss,
            'weighted_inter': self.beta * inter_loss
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss

        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: True labels [batch_size]

        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()