import torch
from prettytable import PrettyTable
from monai.metrics import CumulativeAverage

class LogHelper:
    def __init__(self, experiment, args):
        self.experiment = experiment
        self.args = args
        self.loss_averagers = {}  # key: loss name, value: CumulativeAverage()
        self.last_losses = {}      # key: loss name → last batch value
        self.loss_weights = getattr(args, "loss_weights", {})  # from config, if present
    
    def _avg_add(self, avg_obj, value):
        """Robustly add a scalar to MONAI CumulativeAverage across versions."""
        # ensure python float
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        elif isinstance(value, np.generic):
            value = float(value)
        else:
            value = float(value)

        if hasattr(avg_obj, "append"):
            avg_obj.append(value)      # preferred for current MONAI
        elif hasattr(avg_obj, "update"):
            avg_obj.update(value)      # fallback for older MONAI
        else:
            raise RuntimeError("CumulativeAverage has neither append() nor update().")
        
    def update_losses(self, loss_dict):
        """
        Call this every batch with UNWEIGHTED component losses.
        Example: {"seg_loss": 0.42, "dist_loss": 0.11}
        """
        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.loss_averagers:
                self.loss_averagers[loss_name] = CumulativeAverage()

            avg = self.loss_averagers[loss_name]
            self._avg_add(avg, loss_value)

            # keep last batch value for tqdm
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.detach().item()
            self.last_losses[loss_name] = float(loss_value)

    def get_last_loss(self, name=None):
        """
        Get last batch loss (or total weighted loss if name=None).
        """
        if name is not None:
            return self.last_losses.get(name, None)

        # weighted total of last batch
        weighted_total = 0.0
        for loss_name, last_val in self.last_losses.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            weighted_total += last_val * weight
        return weighted_total
    
    def log_losses(self, epoch, phase="train"):

        # Compute weighted total loss
        weighted_total = 0.0
        for loss_name, avg in self.loss_averagers.items():
            weight = self.loss_weights.get(loss_name, 1.0)  # default weight 1.0
            weighted_total += avg.aggregate() * weight

        # Log to Comet ML
        for loss_name, avg in self.loss_averagers.items():
            self.experiment.log_metric(f"{phase}/loss/{loss_name}", avg.aggregate(), step=epoch)
        self.experiment.log_metric("loss/weighted_total", weighted_total, step=epoch)

        # Console pretty print
        table = PrettyTable()
        table.field_names = [f"Epoch {phase}", "Loss Name", "Average Value", "Weight", "Weighted Value"]
        for loss_name, avg in self.loss_averagers.items():
            weight = self.loss_weights.get(loss_name, 1.0)
            table.add_row([
                epoch,
                loss_name,
                f"{avg.aggregate():.6f}",
                f"{weight:.3f}",
                f"{avg.aggregate() * weight:.6f}"
            ])
        table.add_row([epoch, "Weighted Total", "-", "-", f"{weighted_total:.6f}"])
        print(table)

    def log_metrics(self, epoch: int, metrics_results: dict, phase: str):
        """
        metrics_results: dict from MetricsHelper.compute()
        Logs only metrics that should be computed at this epoch .
        """
        if not metrics_results:
            return  # nothing to log

        # Build pretty table
        table = PrettyTable()
        table.field_names = ["Epoch", "Phase", "Metric", "Value"]

        # Log only eligible metrics
        for name in metrics_results[phase].keys():
            if (
                epoch % self.args.binary_metrics_interval == 0
                or (epoch % self.args.multiclass_metrics_interval == 0 and 
                    epoch >= self.args.multiclass_metrics_firstTime_log)
                or epoch % self.args.distance_metrics_interval == 0
            ):
                comet_name = f"{phase}/metric/{name}"
                self.experiment.log_metric(comet_name, metrics_results[phase][name], step=epoch)
                table.add_row([epoch, phase, name, f"{metrics_results[phase][name]:.4f}"])

        if table._rows:
            print(table)
            

if __name__ == "__main__":
    
    import numpy as np
    import torch
    from metrics_helper import MetricsHelper
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from monai.metrics import CumulativeAverage
    from prettytable import PrettyTable
    from tqdm import tqdm

    # ---- Dummy Experiment (replace with comet_ml.Experiment in real run) ----
    class DummyExperiment:
        def log_metric(self, name, value, step=None):
            print(f"[CometML] step (epoch)={step} metric={name}: {value:.6f}")
    class Args:
        loss_weights = {"seg_loss": 1.0, "dist_loss": 0.5}
        reduction = "mean"
        include_background_metrics = False
        binary_metrics_interval = 2
        distance_metrics_interval = 1
        multiclass_metrics_interval = 4
        multiclass_metrics_firstTime_log = 4
        
    args = Args()
    experiment = DummyExperiment()
    metrics_helper = MetricsHelper(args)
    log_helper = LogHelper(experiment, args)

    n_epochs = 10
    batches_per_epoch = 20
    batch_size = 2
    shape = (1, 16, 16, 16)
    
    # Define metrics groups
    metrics_segmentation_binary = [DiceMetric(include_background=True, reduction="mean")]
    metrics_segmentation_multiclass = [DiceMetric(include_background=True, reduction="mean")]
    distance_metrics = [HausdorffDistanceMetric(include_background=True, reduction="mean")]


    binary_label = torch.randint(0, 2, size=shape).float()
    multiclass_label = torch.randint(0, 3, size=shape).float()
    distance_map_label = torch.rand(shape)
    labels = (binary_label, multiclass_label, distance_map_label)
    

    for epoch in range(1, n_epochs + 1):
        metrics_helper.reset(phase="train")
        print(f"\n=== Epoch {epoch} ===")

        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            # Simulate predictions
            binary_output = torch.randint(0, 2, size=shape).float()
            multiclass_output = torch.randint(0, 3, size=shape).float()
            distance_map_output = torch.rand(shape)
            outputs = (binary_output, multiclass_output, distance_map_output)

            # Update metrics
            metrics_helper.update(outputs, labels, epoch)

            # Simulate batch losses
            loss_dict = {
                "seg_loss": torch.rand(1).item(),
                "dist_loss": torch.rand(1).item()
            }
            log_helper.update_losses(loss_dict)

            # tqdm shows last weighted loss
            pbar.set_postfix({"loss": f"{log_helper.get_last_loss():.4f}"})

        # Compute metrics for this epoch (respects intervals)
        metrics_helper.compute(epoch, phase="train")

        # Log metrics to experiment + print table
        log_helper.log_metrics(epoch, metrics_helper.results, phase="train")