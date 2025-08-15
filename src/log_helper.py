from monai.utils import CumulativeAverage
from prettytable import PrettyTable

class LogHelper:
    def __init__(self, experiment, args):
        self.experiment = experiment
        self.args = args
        self.loss_averagers = {}  # key: loss name, value: CumulativeAverage()

    def log_losses(self, epoch, loss_dict):
        """
        loss_dict: {"dist_loss": float, "seg_loss": float, ...}
        All values are **unweighted losses** from the training step.
        """

        # Init CumulativeAverage trackers if not already present
        for loss_name in loss_dict.keys():
            if loss_name not in self.loss_averagers:
                self.loss_averagers[loss_name] = CumulativeAverage()

        # Update trackers with raw loss values
        for loss_name, loss_value in loss_dict.items():
            self.loss_averagers[loss_name].update(loss_value)

        # Compute weighted total loss
        weighted_total = 0.0
        for loss_name, avg in self.loss_averagers.items():
            weight = self.loss_weights.get(loss_name, 1.0)  # default weight 1.0
            weighted_total += avg.aggregate() * weight

        # Log to Comet ML
        for loss_name, avg in self.loss_averagers.items():
            self.experiment.log_metric(f"loss/{loss_name}", avg.aggregate(), step=epoch)
        self.experiment.log_metric("loss/weighted_total", weighted_total, step=epoch)

        # Console pretty print
        table = PrettyTable()
        table.field_names = ["Epoch", "Loss Name", "Average Value", "Weight", "Weighted Value"]
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

    def log_metrics(self, epoch, metrics_results):
        """
        metrics_results: dict from MetricsHelper.compute()
        """
        for name, value in metrics_results.items():
            self.experiment.log_metric(f"metric/{name}", value, step=epoch)

        # Console output
        table = PrettyTable()
        table.field_names = ["Epoch", "Metric", "Value"]
        for name, value in metrics_results.items():
            table.add_row([epoch, name, f"{value:.4f}"])
        print(table)

    def log_images(self, epoch, image_dict):
        """
        image_dict: {"input": np.ndarray, "prediction": np.ndarray, ...}
        You will handle conversion & inversion before passing here.
        """
        for name, img in image_dict.items():
            self.experiment.log_image(img, name=f"{name}_epoch{epoch}")
