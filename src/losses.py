import warnings
from typing import Callable, Optional, Sequence, Union

import torch
import math
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, look_up_option, pytorch_after

class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            # warnings.warn(
            #     f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
            #     "Using argmax (as a workaround) to convert target to a single channel."
            # )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        return dice_loss, ce_loss
    
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss"""
    def __init__(self, classification_loss='cross_entropy', include_background=True, to_onehot_y=False, sigmoid=False, softmax=False, class_weights=None, reduction='mean', gamma=2.0):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.classification_loss_name = classification_loss
        if self.classification_loss_name not in ['cross_entropy', 'focal_loss']:
            raise ValueError
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        self.focal_loss = FocalLoss(include_background=True, to_onehot_y=to_onehot_y, gamma=gamma, weight=class_weights, reduction=reduction)
        self.generalized_dice = GeneralizedDiceLoss(include_background=include_background, sigmoid=sigmoid, softmax=softmax, to_onehot_y=to_onehot_y, reduction=reduction)
    
    def ce(self, input: torch.Tensor, target: torch.Tensor):

        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.ce_loss(input, target)
    
    def focal(self, input: torch.Tensor, target: torch.Tensor):
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        return self.focal_loss(input, target)
    
    def gdl(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.generalized_dice(y_pred, y_true)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.classification_loss_name == 'cross_entropy':  
            cls_loss = self.ce(y_pred, y_true) 
        elif self.classification_loss_name == 'focal_loss':
            cls_loss = self.focal(y_pred, y_true)  
        seg_loss = self.gdl(y_pred, y_true)
        return seg_loss, cls_loss
    
class DiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma``, ``focal_weight`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background`` and ``reduction`` are used for both losses
    and other parameters are only used for dice loss.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            focal_weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        # total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return dice_loss, focal_loss 
    
class AngularLoss(_Loss):
    """
    Angular loss: squared angle between predicted and ground truth unit vectors.
    Works for any spatial dimensions (H, W, D, ...).
    """

    def __init__(self, eps=1e-6, eps_angle=1e-4, reduction='mean_batch'):
        """
        Args:
            eps_angle: small epsilon to avoid acos domain errors
            reduction: 'none' | 'sum' | 'mean' | 'mean_batch'
        """
        super().__init__(reduction=reduction)
        self.eps = eps
        self.eps_angle = eps_angle
        assert reduction in ['none', 'sum', 'mean', 'mean_batch'], "Invalid reduction"

    def forward(self, pred, gt, mask):
        """
        pred: predicted direction vectors, shape (B, C, ...)
        gt: ground truth direction vectors, shape (B, C, ...)
        mask: binary mask, shape (B, 1, ...) or (B, ...)
        """
        B, C = pred.shape[:2]

        # reshape to (B, C, N)
        pred_vector = pred.reshape(B, C, -1)
        gt_vector = (gt.reshape(B, C, -1))
        mask_vector = mask.reshape(B, -1).float()

        # dot product per voxel
        cos_sim = torch.sum(pred_vector * gt_vector, dim=1)  # shape (B, N)
        cos_sim = torch.clamp(cos_sim, -1 + self.eps_angle, 1 - self.eps_angle)

        # angle error normalized to [0,1]
        angle_errors = torch.acos(cos_sim) / math.pi
        loss_per_voxel = angle_errors ** 2 * mask_vector  # (B, N)

        # apply reduction
        if self.reduction == 'none':
            return loss_per_voxel
        elif self.reduction == 'sum':
            return torch.sum(loss_per_voxel)
        elif self.reduction == 'mean':
            # mean over all valid voxels in batch
            return torch.sum(loss_per_voxel) / (torch.sum(mask_vector) + self.eps)
        elif self.reduction == 'mean_batch':
            # mean per batch element, then mean across batch
            per_batch = torch.sum(loss_per_voxel, dim=1) / (torch.sum(mask_vector, dim=1) + self.eps)
            return torch.mean(per_batch)

class FocalDiceBCELoss(nn.Module):
    """
    Combined loss for 1-channel segmentation: BCE + Focal Dice.
    
    Args:
        alpha (float): weight for BCE loss. Focal Dice weight = 1 - alpha
        gamma (float): focusing exponent for Focal Dice
        bce_weight (float, optional): weight for positive class in BCE (default=1.0)
        smooth (float, optional): smoothing factor for Dice calculation
    """
    def __init__(self, alpha=0.5, gamma=1.0, bce_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weight))
        self.focal_dice = DiceFocalLoss(sigmoid=True, gamma=gamma)

    def forward(self, logits, target):
        """
        Args:
            logits: raw network output, shape (B, 1, D, H, W)
            target: ground truth, shape (B, 1, D, H, W)
        """
        bce = self.bce_loss(logits, target.float())
        dice_loss, focal_loss  = self.focal_dice(logits, target)
        focal_dice_loss = 0.1 * focal_loss + 0.9 * dice_loss
        return self.alpha * bce + (1 - self.alpha) * focal_dice_loss
    
class DiceBCELoss(nn.Module):
    """
    Combined loss for 1-channel segmentation: BCE + Dice.
    
    Args:
        alpha (float): weight for BCE loss. Dice weight = 1 - alpha
        bce_weight (float, optional): weight for positive class in BCE (default=1.0)
        smooth (float, optional): smoothing factor for Dice calculation
    """
    def __init__(self, alpha=0.5, bce_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weight))
        self.dice_loss = DiceLoss(sigmoid=True) 

    def forward(self, logits, target):
        """
        Args:
            logits: raw network output, shape (B, 1, D, H, W)
            target: ground truth, shape (B, 1, D, H, W)
        """
        bce = self.bce_loss(logits, target.float())
        dice_loss = self.dice_loss(logits, target)
        return self.alpha * bce + (1 - self.alpha) * dice_loss
    
#Angular loss tests

def test_zero_loss():
    """Test: Perfect alignment → loss ≈ 0"""
    loss_fn = AngularLoss()
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gt = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones(1, 2)
    loss = loss_fn(pred, gt, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected 0, got {loss.item()}"
    print("test_zero_loss passed.")


def test_opposite_vectors():
    """Test: Opposite direction → maximum angular loss (~1.0 normalized)"""
    loss_fn = AngularLoss()
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gt = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])
    mask = torch.ones(1, 2)
    loss = loss_fn(pred, gt, mask)
    expected = (1.0 ** 2)  # π/π = 1, squared = 1
    assert abs(loss.item() - expected) < 1e-5, f"Expected ~1, got {loss.item()}"
    print("test_opposite_vectors passed.")


def test_half_angle():
    """Test: 90° misalignment → θ = π/2 → normalized θ=0.5 → loss=0.25"""
    loss_fn = AngularLoss()
    pred = torch.tensor([[[1.0], [0.0]]])
    gt   = torch.tensor([[[0.0], [1.0]]])  
    mask = torch.ones(1, 1)
    loss = loss_fn(pred, gt, mask)
    expected = 0.25  # (π/2 / π)^2 = 0.25
    assert abs(loss.item() - expected) < 1e-5, f"Expected ~0.25, got {loss.item()}"
    print("test_half_angle passed.")


def test_masking():
    """Test: Mask excludes some pixels correctly"""
    loss_fn = AngularLoss()
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gt = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    mask = torch.tensor([[1.0, 0.0]])  # only first pixel counts
    loss = loss_fn(pred, gt, mask)
    expected = (0.5 ** 2)  # 90° on one valid pixel
    assert abs(loss.item() - expected) < 1e-5, f"Expected ~0.25, got {loss.item()}"
    print("test_masking passed.")


def test_batch_mean():
    """Test: batch averaging is consistent"""
    loss_fn = AngularLoss()
    pred = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],   # sample 1
        [[1.0, 0.0], [0.0, 1.0]]    # sample 2
    ])
    gt = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],   # aligned
        [[0.0, 1.0], [1.0, 0.0]]    # 90° misaligned
    ])
    mask = torch.ones(2, 2)
    loss = loss_fn(pred, gt, mask)
    expected = (0.0 + 0.25) / 2  # mean over 2 samples
    assert abs(loss.item() - expected) < 1e-5, f"Expected {expected}, got {loss.item()}"
    print("test_batch_mean passed.")
    
if __name__ == "__main__":
    torch.manual_seed(0)
    test_zero_loss()
    test_opposite_vectors()
    test_half_angle()
    test_masking()
    test_batch_mean()