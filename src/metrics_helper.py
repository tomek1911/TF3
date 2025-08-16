import torch
import warnings
import numpy as np
from collections import defaultdict
from monai.metrics import MSEMetric
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
from monai.networks.utils import one_hot

class MetricsHelper:
    def __init__(self, args):
        """
        args: config with attributes like include_background_metrics, metric_intervals, etc.
        """
        self.args = args

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
        self.results = {"train": {}, "val": {}}
        self.metric_higher_is_better = {}
        self.device_metrics = torch.device(args.metrics_device)  

        for m in self.metrics_segmentation_binary + \
                self.metrics_segmentation_multiclass + \
                self.distance_metrics:
            if isinstance(m, DiceMetric):
                self.metric_higher_is_better[m.__class__.__name__] = True
            else:
                # Hausdorff, SurfaceDistance, MSE, etc
                self.metric_higher_is_better[m.__class__.__name__] = False

        # Initialize best_results based on this
        self.best_results = {"train": {}, "val": {}}
        for phase in ["train", "val"]:
            for name, higher in self.metric_higher_is_better.items():
                if higher:
                    self.best_results[phase][name] = -np.inf
                else:
                    self.best_results[phase][name] = np.inf
                    

    def update(self, outputs, labels, epoch):
        """
        Accumulate predictions & labels for metrics that should be computed this epoch.
        """
        
        multiclass_output, binary_output, distance_map_output = (t.detach() for t in outputs)
        multiclass_label, binary_label, distance_map_label = (t.detach() for t in labels)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="monai.metrics.utils")
            if epoch % self.args.binary_metrics_interval == 0:
                y_pred = (torch.sigmoid(binary_output) > 0.5).float().to(self.device_metrics)
                y = binary_label.to(self.device_metrics)
                for metric in self.metrics_segmentation_binary:
                    metric(y_pred, y)

            if epoch % self.args.multiclass_metrics_interval == 0:
                if epoch >= self.args.multiclass_metrics_firstTime_log:
                    y_pred = multiclass_output.argmax(dim=1, keepdim=True).to(self.device_metrics)
                    y = multiclass_label.to(self.device_metrics)
                    if self.args.is_multiclass_one_hot:
                        y_pred = one_hot(y_pred, num_classes=self.args.out_channels, dim=1)
                        y = one_hot(y, num_classes=self.args.out_channels, dim=1)
                    for metric in self.metrics_segmentation_multiclass:
                        metric(y_pred, y)
                else:
                    #binary version for early epochs
                    y_pred=(multiclass_output.argmax(dim=1, keepdim=True) != 0).long().to(self.device_metrics)
                    y=torch.where(multiclass_label >= 1, 1, 0).to(self.device_metrics)
                    for metric in self.metrics_segmentation_multiclass:
                        metric(y_pred,y)
        
            if epoch % self.args.distance_metrics_interval == 0:
                for metric in self.distance_metrics:
                    metric(y_pred=torch.sigmoid(distance_map_output), y=distance_map_label)
        
                
    def compute(self, epoch, phase="train"):
        """
        Aggregate metric buffers for this epoch and update bests.
        Returns: dict with computed metrics.
        """
        results = {}

        def compute_group(metrics, interval, min_epoch=None, name_postfix=""):
            group_results = {}
            if epoch % interval == 0 and (min_epoch is None or epoch >= min_epoch):
                for m in metrics:
                    val = m.aggregate()
                    if val is None:
                        continue  # skip if metric did not accumulate any data

                    # If aggregate() returns tensor, convert to scalar
                    val = val.item() if hasattr(val, "item") else val
                    unique_name = m.__class__.__name__ + '_' + name_postfix
                    group_results[unique_name] = val

                    # Determine if higher is better
                    name = m.__class__.__name__ #metric type name
                    higher_is_better = self.metric_higher_is_better.get(name, True)  # default True
                    if higher_is_better:
                        if val > self.best_results[phase][name]:
                            self.best_results[phase][name] = val
                    else:
                        if val < self.best_results[phase][name]:
                            self.best_results[phase][name] = val

                    m.reset()  # free buffers

            return group_results

        results.update(compute_group(self.metrics_segmentation_binary, self.args.binary_metrics_interval, name_postfix="pulp"))
        results.update(compute_group(self.metrics_segmentation_multiclass, self.args.multiclass_metrics_interval,
                       min_epoch=self.args.multiclass_metrics_firstTime_log, name_postfix="multiclass"))
        results.update(compute_group(self.distance_metrics,
                       self.args.distance_metrics_interval, name_postfix="distance"))

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


    args = Args()
    metrics_helper = MetricsHelper(args)

    num_epochs = 100
    batch_size = 2
    shape = (1, 16, 16, 16)
    
    binary_label = torch.randint(0, 2, size=shape).float()
    multiclass_label = torch.randint(0, 3, size=shape).float()
    distance_map_label = torch.rand(shape)
    labels = (binary_label, multiclass_label, distance_map_label)
            
    for epoch in range(1, num_epochs+1):
        metrics_helper.reset(phase="train")
        print(f"\n=== Epoch {epoch} ===")

        for batch in range(3):  # simulate 3 batches
            # Dummy predictions and labels
            binary_output = torch.randint(0, 2, size=shape).float()
            multiclass_output = torch.randint(0, 3, size=shape).float()
            distance_map_output = torch.rand(shape)
            outputs = (binary_output, multiclass_output, distance_map_output)
            metrics_helper.update(outputs, labels, epoch)

        metrics_helper.compute(epoch, phase="train")
        print(f"Current: {metrics_helper.get_best(phase='train')}") 
        print(f"Best metrics: {metrics_helper.get_best(phase='train')}")