import io
import os

import numpy as np
import torch
from prettytable import PrettyTable
from monai.metrics import CumulativeAverage
from PIL import Image as PILImage

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
    
    def log_lr(self, scheduler, epoch):
        if self.args.scheduler_name == "cosine_annealing":
            self.experiment.log_metric("lr_rate", scheduler.get_last_lr()[0], epoch=epoch)
        elif self.args.scheduler_name == "warmup_cosine":
            self.experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
        elif self.args.scheduler_name == "warmup_cosine_restarts":
            self.experiment.log_metric("lr_rate", scheduler.get_lr(), epoch=epoch)
        else:
            raise UserWarning(f"Unknown scheduler: {self.args.scheduler_name}, omitting logging.")
    
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
                f"{weight:.4g}",
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

        phase_results = metrics_results.get(phase, {})

        # Handle per-class / HD95 arrays separately (not scalars)
        per_class_dice = phase_results.pop('multiclass_per_class_dice', None)
        per_class_hd95 = phase_results.pop('multiclass_per_class_hd95', None)
        mean_hd95      = phase_results.pop('multiclass_mean_hd95', None)
        challenge_hd95 = phase_results.pop('multiclass_challenge_hd95', None)

        # Build pretty table for scalar metrics
        table = PrettyTable()
        table.field_names = ["Epoch", "Phase", "Metric", "Value"]

        hd95_interval = getattr(self.args, 'hd95_metrics_interval', 100)
        for name, value in phase_results.items():
            if (
                epoch % self.args.binary_metrics_interval == 0
                or (epoch % self.args.multiclass_metrics_interval == 0 and
                    epoch >= self.args.multiclass_metrics_firstTime_log)
                or epoch % self.args.distance_metrics_interval == 0
            ):
                self.experiment.log_metric(f"{phase}/metric/{name}", value, step=epoch)
                table.add_row([epoch, phase, name, f"{value:.4g}"])

        if table._rows:
            print(table)

        # Per-class dice: log individual metrics + Comet table + console table
        if per_class_dice is not None:
            # Individual metrics (one per class — gives time-series chart in Comet)
            for i, v in enumerate(per_class_dice):
                self.experiment.log_metric(f"{phase}/dice_per_class/class_{i:02d}", float(v), step=epoch)

            # Comet table (viewable in Assets & Artifacts)
            headers = ["class_idx", "dice"]
            rows = [[i, round(float(v), 6)] for i, v in enumerate(per_class_dice)]
            self.experiment.log_table(f"{phase}_dice_per_class.csv", tabular_data=rows, headers=headers)

            # Console table — 8 dice values per row; "cls" = starting class index, "+N" = offset
            cols = 8
            n = len(per_class_dice)
            cls_table = PrettyTable()
            cls_table.field_names = ["cls"] + [f"+{c}" for c in range(cols)]
            for row_start in range(0, n, cols):
                row = [row_start]
                for c in range(cols):
                    idx = row_start + c
                    row.append(f"{per_class_dice[idx]:.4f}" if idx < n else "")
                cls_table.add_row(row)
            print(f"\nPer-class Dice ({phase}, epoch {epoch}):")
            print(cls_table)

        # HD95 scalar metrics (mean + challenge)
        if mean_hd95 is not None or challenge_hd95 is not None:
            hd95_scalar_table = PrettyTable()
            hd95_scalar_table.field_names = ["Epoch", "Phase", "Metric", "Value"]
            if mean_hd95 is not None:
                self.experiment.log_metric(f"{phase}/metric/multiclass_mean_hd95", mean_hd95, step=epoch)
                hd95_scalar_table.add_row([epoch, phase, "multiclass_mean_hd95", f"{mean_hd95:.4g}"])
            if challenge_hd95 is not None:
                self.experiment.log_metric(f"{phase}/metric/multiclass_challenge_hd95", challenge_hd95, step=epoch)
                hd95_scalar_table.add_row([epoch, phase, "multiclass_challenge_hd95", f"{challenge_hd95:.4g}"])
            print(hd95_scalar_table)

        # Per-class HD95: NaN/inf → 0 for display and Comet logging
        if per_class_hd95 is not None:
            per_class_hd95_display = np.where(np.isfinite(per_class_hd95), per_class_hd95, 0.0)

            # Individual metrics in Comet (one per class — time-series chart)
            for i, v in enumerate(per_class_hd95_display):
                self.experiment.log_metric(f"{phase}/hd95_per_class/class_{i:02d}", float(v), step=epoch)

            # Comet table (Assets & Artifacts)
            headers = ["class_idx", "hd95"]
            rows = [[i, round(float(v), 4)] for i, v in enumerate(per_class_hd95_display)]
            self.experiment.log_table(f"{phase}_hd95_per_class.csv", tabular_data=rows, headers=headers)

            # Console table — 8 values per row
            cols = 8
            n = len(per_class_hd95_display)
            cls_table = PrettyTable()
            cls_table.field_names = ["cls"] + [f"+{c}" for c in range(cols)]
            for row_start in range(0, n, cols):
                row = [row_start]
                for c in range(cols):
                    idx = row_start + c
                    row.append(f"{per_class_hd95_display[idx]:.2f}" if idx < n else "")
                cls_table.add_row(row)
            print(f"\nPer-class HD95 ({phase}, epoch {epoch}):")
            print(cls_table)

    # -------------------------------------------------------------------------
    # 2-D slice preview helpers
    # -------------------------------------------------------------------------

    def _load_colormap(self):
        """Load class→RGB colormap from data/colormap.csv (47 classes, 0=background)."""
        csv_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'colormap.csv')
        )
        colors = np.zeros((47, 3), dtype=np.uint8)
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 4:
                    idx = int(parts[0])
                    if idx < 47:
                        colors[idx] = [int(parts[1]), int(parts[2]), int(parts[3])]
        return colors

    def _colorize_seg(self, seg_slice_np):
        """Map 2-D integer class array → [H, W, 3] uint8 RGB using the loaded colormap."""
        clamped = np.clip(seg_slice_np, 0, len(self._colormap) - 1)
        return self._colormap[clamped]

    def _viridis_colormap(self, val_np):
        """
        Apply matplotlib viridis to a 2-D float array in [0, 1].
        Returns [H, W, 3] uint8 RGB.
        """
        import matplotlib
        cmap = matplotlib.colormaps['viridis']
        rgba = cmap(val_np)          # [H, W, 4] float64 in [0, 1]
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _dir_slice_to_rgb(self, dir_sl_3hw, gt_binary_sl):
        """
        Convert a 3-channel direction slice to viridis RGB, one channel at a time,
        masked by the GT binary mask.  Values are in [-1, 1]; map to [0, 1] for display.
        Channels are visualised as three side-by-side viridis images then stacked vertically
        (or could be shown as R/G/B — here we show them stacked as a single channel
        with per-channel labels).  For simplicity: average the 3 channels into one
        magnitude image, then apply viridis.

        dir_sl_3hw   : [3, h, w]  float direction values in [-1, 1]
        gt_binary_sl : [h, w]     binary mask  (1 inside object, 0 outside)
        Returns [h, w*3, 3] uint8 — three viridis panels (x, y, z channels) side-by-side.
        """
        panels = []
        for c in range(3):
            ch = dir_sl_3hw[c].astype(np.float32)      # [-1, 1]
            ch_norm = (ch + 1.0) / 2.0                  # -> [0, 1]
            ch_norm = ch_norm * gt_binary_sl             # mask outside object
            panels.append(self._viridis_colormap(ch_norm))  # [h, w, 3]
        return np.concatenate(panels, axis=1)            # [h, w*3, 3]

    def log_slices_2d(self, epoch: int, seg_logits, dist_tensor, dir_tensor,
                      gt_label, gt_dist_tensor, gt_dir, phase: str):
        """
        Log axial / coronal / sagittal center-slice previews to Comet ML.
        Each image has two rows:
          Row 1 (prediction): [colorized seg | greyscale dist | viridis dir (x|y|z)]
          Row 2 (GT):         [colorized GT seg | greyscale GT dist | viridis GT dir (x|y|z)]

        seg_logits      : [B, C, H, W, D]  raw logits (argmax taken internally)
        dist_tensor     : [B, 1, H, W, D]  raw distance logits (sigmoid applied internally)
        dir_tensor      : [B, 3, H, W, D]  predicted direction, values in [-1, 1]
        gt_label        : [B, 1, H, W, D]  GT class labels (integer)
        gt_dist_tensor  : [B, 1, H, W, D]  GT distance map, values in [0, 1]
        gt_dir          : [B, 3, H, W, D]  GT direction map, values in [-1, 1]
        """
        if not getattr(self.args, 'is_log_qualitative_2d', False):
            return
        if epoch % max(1, self.args.log_slice_2d_interval) != 0:
            return

        # from PIL import Image as PILImage

        if not hasattr(self, '_colormap'):
            self._colormap = self._load_colormap()

        # --- work on the first sample in the batch ---
        seg_argmax = seg_logits[0].detach().float().cpu().argmax(dim=0).numpy().astype(np.int32)  # [H, W, D]
        dist_np    = torch.sigmoid(dist_tensor[0, 0].detach().float().cpu()).numpy()              # [H, W, D]
        dir_np     = dir_tensor[0].detach().float().cpu().numpy()                                 # [3, H, W, D]

        gt_seg_np  = gt_label[0, 0].detach().float().cpu().numpy().astype(np.int32)              # [H, W, D]
        gt_dir_np  = gt_dir[0].detach().float().cpu().numpy()                                    # [3, H, W, D]
        gt_bin_np  = (gt_seg_np > 0).astype(np.float32)                                          # [H, W, D]
        gt_dist_np = gt_dist_tensor[0, 0].detach().float().cpu().numpy()                          # [H, W, D]

        H, W, D = seg_argmax.shape

        def _slice(volume, ax, idx):
            """Extract 2-D slice from [H,W,D] volume along given axis at idx."""
            if ax == 0: return volume[idx, :, :]
            if ax == 1: return volume[:, idx, :]
            return volume[:, :, idx]

        def _slice3(volume, ax, idx):
            """Extract 2-D slice from [3,H,W,D] volume along given spatial axis."""
            if ax == 0: return volume[:, idx, :, :]
            if ax == 1: return volume[:, :, idx, :]
            return volume[:, :, :, idx]

        planes = [
            ('axial',    2, D // 2),
            ('coronal',  1, W // 2),
            ('sagittal', 0, H // 2),
        ]

        for plane_name, ax, idx in planes:
            # --- prediction slices ---
            seg_sl      = _slice(seg_argmax, ax, idx)          # [h, w]
            dist_sl     = _slice(dist_np,    ax, idx)          # [h, w]
            dir_sl      = _slice3(dir_np,    ax, idx)          # [3, h, w]
            pred_bin_sl = (seg_sl > 0).astype(np.float32)      # [h, w]

            seg_rgb   = self._colorize_seg(seg_sl)                                   # [h, w, 3]
            dist_rgb  = np.stack([(dist_sl * 255).astype(np.uint8)] * 3, axis=-1)   # [h, w, 3]
            dir_rgb   = self._dir_slice_to_rgb(dir_sl, pred_bin_sl)                 # [h, w*3, 3]

            pred_row = np.concatenate([seg_rgb, dist_rgb, dir_rgb], axis=1)         # [h, (1+1+3)w, 3]

            # --- GT slices ---
            gt_seg_sl  = _slice(gt_seg_np,  ax, idx)           # [h, w]
            gt_dist_sl = _slice(gt_dist_np, ax, idx)           # [h, w]  (binary proxy)
            gt_dir_sl  = _slice3(gt_dir_np, ax, idx)           # [3, h, w]
            gt_bin_sl  = (gt_seg_sl > 0).astype(np.float32)    # [h, w]

            gt_seg_rgb  = self._colorize_seg(gt_seg_sl)                                    # [h, w, 3]
            gt_dist_rgb = np.stack([(gt_dist_sl * 255).astype(np.uint8)] * 3, axis=-1)    # [h, w, 3]
            gt_dir_rgb  = self._dir_slice_to_rgb(gt_dir_sl, gt_bin_sl)                    # [h, w*3, 3]

            gt_row = np.concatenate([gt_seg_rgb, gt_dist_rgb, gt_dir_rgb], axis=1)        # [h, 5w, 3]

            # --- stack rows ---
            panel = np.concatenate([pred_row, gt_row], axis=0)   # [2h, 5w, 3]

            img = PILImage.fromarray(panel, 'RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)

            self.experiment.log_image(
                buf,
                name=f"{phase}/preview/{plane_name}",
                step=epoch,
            )


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
        is_multiclass_one_hot = True
        metrics_device = "cuda:1"
        out_channels = 5 
        
    args = Args()
    experiment = DummyExperiment()
    metrics_helper = MetricsHelper(args)
    log_helper = LogHelper(experiment, args)

    n_epochs = 20
    batches_per_epoch = 20
    batch_size = 2
    shape = (batch_size, 1, 16, 16, 16)
    
    # Dummy labels
    binary_label = torch.randint(0, 2, size=shape).float()
    multiclass_label = torch.randint(0, args.out_channels, size=shape).float()
    distance_map_label = torch.rand(shape)
    labels = (binary_label, multiclass_label, distance_map_label)
    

    for epoch in range(1, n_epochs + 1):
        print(f"\n=== Epoch {epoch} ===")

        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}")
        for batch_idx in pbar:
            # Dummy outputs
            binary_output = torch.randn(size=shape).float()
            multiclass_output = torch.randn(size=(batch_size, args.out_channels, *shape[2:])).float()
            distance_map_output = torch.rand(shape)
            outputs = (binary_output, multiclass_output, distance_map_output)

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
        metrics_helper.reset()
        
        log_helper.log_metrics(epoch, metrics_helper.results, phase="train")
        log_helper.log_losses(epoch, phase="train")