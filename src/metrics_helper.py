import torch
import warnings
import numpy as np
import cupy as cp
from collections import defaultdict
from monai.metrics import MSEMetric
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.networks.utils import one_hot

class MetricsHelper:
    def __init__(self, args, group_names = ["binary", "multiclass", "distance"]):
        """
        args: config with attributes like include_background_metrics, metric_intervals, etc.
        """
        self.args = args
        
        # Metric groups
        self.metrics_group_names = group_names
        self.metrics_segmentation_binary = [
            DiceMetric(include_background=args.include_background_metrics, reduction=args.reduction, ignore_empty=True),
            # SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=args.reduction, symmetric=True),
            # HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
            #                         percentile=None, get_not_nans=False, directed=False, reduction=args.reduction)
        ]
        self.metrics_segmentation_multiclass = [
            DiceMetric(include_background=args.include_background_metrics, reduction="none", ignore_empty=True),
            # SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=args.reduction, symmetric=True),
            # HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
            #                         percentile=None, get_not_nans=False, directed=False, reduction=args.reduction),
        ]
        self.distance_metrics = [MSEMetric(reduction=args.reduction)]
        # HD95: accumulated as a list of [B, C] cpu float tensors (one per update call).
        # We deliberately do NOT use HausdorffDistanceMetric here: that requires the full
        # one-hot [B, C, H, W, D] tensor which OOMs on full val volumes.  Instead we loop
        # class-by-class with [B, 1, H, W, D] bool masks (cucim-accelerated via MONAI).
        self._hd95_buffer = []         # list of [B, C] cpu float tensors
        self._vol_diagonals_hd95 = []  # per-sample volume diagonals for challenge HD95
        
        # Results and best tracking
        self.results = {"train": {}, "val": {}}
        self.metric_higher_is_better = {}
        self.device_metrics = torch.device(args.metrics_device)  

        # Initialize metric values with unique keys
        for group_name, metrics in [
            (self.metrics_group_names[0], self.metrics_segmentation_binary),
            (self.metrics_group_names[1], self.metrics_segmentation_multiclass),
            (self.metrics_group_names[2], self.distance_metrics),
        ]:
            for m in metrics:
                key = self._metric_key(m, group_name)
                if isinstance(m, DiceMetric):
                    self.metric_higher_is_better[key] = True
                else:
                    self.metric_higher_is_better[key] = False
                    
                    
        # Initialize best_results
        self.best_results = {"train": {}, "val": {}}
        for phase in ["train", "val"]:
            for name, higher in self.metric_higher_is_better.items():
                self.best_results[phase][name] = -np.inf if higher else np.inf

    def _metric_key(self, metric, group_name, idx=None):
        """Generate unique key for a metric instance."""
        key = f"{group_name}_{metric.__class__.__name__}"
        if idx is not None:
            key += f"_{idx}"
        return key

    def update(self, outputs, labels, epoch, volume_shape=None):
        """
        Accumulate predictions & labels for metrics that should be computed this epoch.
        """
        
        binary_output, multiclass_output, distance_map_output = outputs
        binary_label, multiclass_label, distance_map_label = labels

        with torch.inference_mode(), warnings.catch_warnings(), cp.cuda.Device(int(self.args.metrics_device[-1])):
            warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics.utils")
            if epoch % self.args.binary_metrics_interval == 0 and binary_output is not None and binary_label is not None:
                b_out = binary_output.detach().to(self.device_metrics, non_blocking=True)
                b_lab = binary_label.detach().to(self.device_metrics, non_blocking=True)
                y_pred = (torch.sigmoid(b_out) > 0.5).float().to(self.device_metrics)
                y = b_lab.to(self.device_metrics)
                for metric in self.metrics_segmentation_binary:
                    metric(y_pred, y)
                del b_out, b_lab, y_pred, y

            hd95_interval = getattr(self.args, 'hd95_metrics_interval', 100)
            do_multiclass = epoch % self.args.multiclass_metrics_interval == 0
            do_hd95       = epoch % hd95_interval == 0

            if do_multiclass or do_hd95:
                # Move to GPU once — shared by both dice and HD95
                mc_out = multiclass_output.detach().to(self.device_metrics, non_blocking=True)
                mc_lab = multiclass_label.detach().to(self.device_metrics, non_blocking=True)
                # argmax is always [B, H, W, D]; keepdim variant used only for dice one-hot
                argmax = mc_out.argmax(dim=1)       # [B, H, W, D]
                label  = mc_lab[:, 0].long()        # [B, H, W, D]
                del mc_out                          # logits no longer needed

                if do_multiclass:
                    if epoch >= self.args.multiclass_metrics_firstTime_log:
                        if self.args.is_multiclass_one_hot:
                            y_pred_oh = one_hot(argmax.unsqueeze(1), num_classes=self.args.out_channels, dim=1)
                            y_oh      = one_hot(label.unsqueeze(1),  num_classes=self.args.out_channels, dim=1)
                            for metric in self.metrics_segmentation_multiclass:
                                metric(y_pred_oh, y_oh)
                            del y_pred_oh, y_oh
                        else:
                            for metric in self.metrics_segmentation_multiclass[:1]:  # only DiceMetric
                                metric(argmax.unsqueeze(1), label.unsqueeze(1))
                        # Per-class dice is extracted from the DiceMetric buffer in compute()
                    else:
                        # binary version for early epochs
                        y_pred = (argmax.unsqueeze(1) != 0).long()
                        y      = (label.unsqueeze(1)  >= 1).long()
                        for metric in self.metrics_segmentation_multiclass:
                            metric(y_pred, y)
                        del y_pred, y

                if do_hd95:
                    B, C    = argmax.shape[0], self.args.out_channels
                    start_c = 0 if self.args.include_background_metrics else 1
                    # Accumulate scalars on CPU — never materialise full one-hot on GPU
                    hd95_row = torch.full((B, C), float('nan'), dtype=torch.float32)
                    for c in range(start_c, C):
                        pred_c = (argmax == c).unsqueeze(1).float()  # [B, 1, H, W, D] on GPU
                        gt_c   = (label  == c).unsqueeze(1).float()  # [B, 1, H, W, D] on GPU
                        try:
                            res = compute_hausdorff_distance(
                                pred_c, gt_c,
                                include_background=True,  # single channel — skip background strip
                                distance_metric='euclidean',
                                percentile=95,
                                directed=False,
                            )  # [B, 1]; nan = GT absent, inf = pred absent
                            hd95_row[:, c] = res[:, 0].float().cpu()
                        except Exception:
                            pass
                        del pred_c, gt_c

                    self._hd95_buffer.append(hd95_row)  # [B, C] on cpu
                    if volume_shape is not None:
                        diag = float(np.sqrt(sum(d ** 2 for d in volume_shape)))
                        self._vol_diagonals_hd95.extend([diag] * B)

                del argmax, label, mc_lab

            if epoch % self.args.distance_metrics_interval == 0:
                d_out = distance_map_output.detach().to(self.device_metrics, non_blocking=True)
                y = distance_map_label.detach().to(self.device_metrics, non_blocking=True)
                y_pred = torch.sigmoid(d_out)
                for metric in self.distance_metrics:
                    metric(y_pred, y)
                del d_out, y, y_pred
        
    def compute(self, epoch, phase="train"):
        """
        Aggregate metric buffers for this epoch and update bests.
        Returns: dict with computed metrics.
        """
        results = {}

        def compute_group(metrics, group_name, interval, min_epoch=None):
            group_results = {}
            if epoch % interval == 0 and (min_epoch is None or epoch >= min_epoch):
                for m in metrics:
                    try:
                        val = m.aggregate()
                    except (ValueError, RuntimeError):
                        m.reset()
                        continue  # no data was accumulated (e.g. no pulp head)
                    if val is None:
                        continue
                    if isinstance(val, torch.Tensor) and val.numel() > 1:
                        val = float(torch.nanmean(val.float()))
                    elif hasattr(val, "item"):
                        val = val.item()
                    else:
                        val = float(val)

                    key = self._metric_key(m, group_name)
                    group_results[key] = val

                    higher_is_better = self.metric_higher_is_better[key]
                    if higher_is_better:
                        self.best_results[phase][key] = max(self.best_results[phase][key], val)
                    else:
                        self.best_results[phase][key] = min(self.best_results[phase][key], val)

                    m.reset()  # free buffers

            return group_results

        results.update(compute_group(self.metrics_segmentation_binary, self.metrics_group_names[0], self.args.binary_metrics_interval))

        # Multiclass: extract per-class dice from aggregate() BEFORE compute_group calls reset()
        # reduction="none" → aggregate() returns [N_samples, C]; nanmean over samples → [C]
        if (epoch % self.args.multiclass_metrics_interval == 0
                and epoch >= self.args.multiclass_metrics_firstTime_log):
            for m in self.metrics_segmentation_multiclass:
                if isinstance(m, DiceMetric):
                    agg = m.aggregate()  # [N, C] with reduction="none"
                    if agg is not None and agg.numel() > 0:
                        per_class = torch.nanmean(agg.float(), dim=0)  # [C]
                        results['multiclass_per_class_dice'] = per_class.cpu().numpy()
                    break

        results.update(compute_group(self.metrics_segmentation_multiclass, self.metrics_group_names[1],
                                     self.args.multiclass_metrics_interval,
                                     min_epoch=self.args.multiclass_metrics_firstTime_log))
        results.update(compute_group(self.distance_metrics, self.metrics_group_names[2], self.args.distance_metrics_interval))

        # HD95 per-class (every hd95_metrics_interval epochs)
        hd95_interval = getattr(self.args, 'hd95_metrics_interval', 100)
        if epoch % hd95_interval == 0:
            if self._hd95_buffer:
                hd95_all = torch.cat(self._hd95_buffer, dim=0)  # [N_samples, C] cpu
                # treat inf as nan so nanmean ignores both absent-GT and absent-pred uniformly
                hd95_float = torch.where(torch.isinf(hd95_all),
                                         torch.full_like(hd95_all, float('nan')),
                                         hd95_all)

                # Per-class HD95: nanmean over samples.  NaN stays for classes absent in all samples.
                per_class_hd95 = torch.nanmean(hd95_float, dim=0)  # [C]
                results['multiclass_per_class_hd95'] = per_class_hd95.numpy()

                # Scalar mean HD95 — ignore missing classes
                results['multiclass_mean_hd95'] = float(torch.nanmean(hd95_float))

                # Challenge HD95: missing class → volume diagonal.  Validation only.
                if phase == "val" and self._vol_diagonals_hd95:
                    hd95_challenge = hd95_all.clone()  # [N, C]
                    n_samples = min(len(self._vol_diagonals_hd95), hd95_challenge.shape[0])
                    for i in range(n_samples):
                        # replace inf (pred absent) AND nan (GT absent) with diagonal
                        bad = torch.isinf(hd95_challenge[i]) | torch.isnan(hd95_challenge[i])
                        hd95_challenge[i][bad] = self._vol_diagonals_hd95[i]
                    results['multiclass_challenge_hd95'] = float(hd95_challenge[:n_samples].mean())

            self._hd95_buffer.clear()
            self._vol_diagonals_hd95.clear()

        self.results[phase] = results

    def reset(self):
        for group in [self.metrics_segmentation_binary,
                      self.metrics_segmentation_multiclass,
                      self.distance_metrics]:
            for m in group:
                m.reset()
        self._hd95_buffer.clear()
        self._vol_diagonals_hd95.clear()

    def get_best(self, phase="train"):
        """Return dict of best metrics for a given phase."""
        return dict(self.best_results[phase])

    def get_current(self, phase="train"):
        """Return last computed metric values for a given phase."""
        return dict(self.results[phase])
    
if __name__ == "__main__":
    class Args:
        include_background_metrics = False
        reduction = "mean"
        binary_metrics_interval = 10
        distance_metrics_interval = 5
        multiclass_metrics_interval = 20
        multiclass_metrics_firstTime_log = 40
        is_multiclass_one_hot = True
        metrics_device = "cuda:1"
        out_channels = 5 # e.g. 0, 1,...4


    args = Args()
    metrics_helper = MetricsHelper(args)

    num_epochs = 100
    batch_size = 2
    shape = (batch_size, 1, 16, 16, 16)
    
    binary_label = torch.randint(0, 2, size=shape).float()
    multiclass_label = torch.randint(0, args.out_channels, size=shape).float()
    distance_map_label = torch.rand(shape)
    labels = (binary_label, multiclass_label, distance_map_label)
            
    for epoch in range(1, num_epochs+1):
        print(f"\n=== Epoch {epoch} ===")

        for batch in range(3):  # simulate 3 batches
            # Dummy outputs
            binary_output = torch.randn(size=shape).float()
            multiclass_output = torch.randn(size=(batch_size, args.out_channels, *shape[2:])).float()
            distance_map_output = torch.rand(shape)
            outputs = (binary_output, multiclass_output, distance_map_output)
            
            metrics_helper.update(outputs, labels, epoch)

        metrics_helper.compute(epoch, phase="train")
        metrics_helper.reset()
        
        print(f"Current: {metrics_helper.get_current(phase='train')}") 
        print(f"Best metrics: {metrics_helper.get_best(phase='train')}")