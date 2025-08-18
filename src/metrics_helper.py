import torch
import warnings
import numpy as np
import cupy as cp
from collections import defaultdict
from monai.metrics import MSEMetric
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
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
            DiceMetric(include_background=args.include_background_metrics, reduction=args.reduction, ignore_empty=True),
            # SurfaceDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean', reduction=args.reduction, symmetric=True),
            # HausdorffDistanceMetric(include_background=args.include_background_metrics, distance_metric='euclidean',
            #                         percentile=None, get_not_nans=False, directed=False, reduction=args.reduction),
        ]
        self.distance_metrics = [MSEMetric(reduction=args.reduction)]
        
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

    def update(self, outputs, labels, epoch):
        """
        Accumulate predictions & labels for metrics that should be computed this epoch.
        """
        
        binary_output, multiclass_output, distance_map_output = outputs
        binary_label, multiclass_label, distance_map_label = labels

        with torch.inference_mode(), warnings.catch_warnings(), cp.cuda.Device(int(self.args.metrics_device[-1])):
            warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics.utils")
            if epoch % self.args.binary_metrics_interval == 0:
                b_out = binary_output.detach().to(self.device_metrics, non_blocking=True)
                b_lab = binary_label.detach().to(self.device_metrics, non_blocking=True)
                y_pred = (torch.sigmoid(b_out) > 0.5).float().to(self.device_metrics)
                y = b_lab.to(self.device_metrics)
                for metric in self.metrics_segmentation_binary:
                    metric(y_pred, y)
                del b_out, b_lab, y_pred, y

            if epoch % self.args.multiclass_metrics_interval == 0:
                mc_out = multiclass_output.detach().to(self.device_metrics, non_blocking=True)
                mc_lab = multiclass_label.detach().to(self.device_metrics, non_blocking=True)
                if epoch >= self.args.multiclass_metrics_firstTime_log:
                    y_pred = mc_out.argmax(dim=1, keepdim=True).to(self.device_metrics)
                    y = mc_lab.to(self.device_metrics)
                    if self.args.is_multiclass_one_hot: # required for surface distance and hausdorff metrics
                        y_pred = one_hot(y_pred, num_classes=self.args.out_channels, dim=1)
                        y = one_hot(y, num_classes=self.args.out_channels, dim=1)
                        for metric in self.metrics_segmentation_multiclass:
                            metric(y_pred, y)
                    else:
                        for metric in self.metrics_segmentation_multiclass[:1]: #only dice metric - IGNORE, others require one-hot
                            metric(y_pred, y)
                    del y_pred, y
                else:
                    #binary version for early epochs
                    y_pred=(mc_out.argmax(dim=1, keepdim=True) != 0).long().to(self.device_metrics)
                    ones = torch.ones_like(mc_lab, device=self.device_metrics)
                    zeros = torch.zeros_like(mc_lab, device=self.device_metrics)
                    y = torch.where(mc_lab >= 1, ones, zeros).to(torch.long)
                    for metric in self.metrics_segmentation_multiclass: 
                        metric(y_pred,y)
                    del y_pred, y, ones, zeros
                del mc_out, mc_lab
        
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
                    val = m.aggregate()
                    if val is None:
                        continue  # skip if metric did not accumulate any data
                    val = val.item() if hasattr(val, "item") else val

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
        results.update(compute_group(self.metrics_segmentation_multiclass, self.metrics_group_names[1],
                                     self.args.multiclass_metrics_interval,
                                     min_epoch=self.args.multiclass_metrics_firstTime_log))
        results.update(compute_group(self.distance_metrics, self.metrics_group_names[2], self.args.distance_metrics_interval))

        self.results[phase] = results

    def reset(self):
        for group in [self.metrics_segmentation_binary,
                      self.metrics_segmentation_multiclass,
                      self.distance_metrics]:
            for m in group:
                m.reset()

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