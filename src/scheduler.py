import math
import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim import Optimizer

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        t_total: int,
        min_lr: float = 0.0,
        cycles: float = 0.5,
        last_epoch: int = -1,
        warmup_start_multiplier: float = 0,
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            min_lr: Minimum LR at the end of cosine decay.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
            warmup_start_multiplier: if provided, starts the linear warmup from this fraction of the initial lr.
                Must be in 0..1 interval. Defaults to 0
        Returns:
            None
        """
        self.warmup_steps = min(max(warmup_steps, 0), t_total)
        self.warmup_multiplier = warmup_start_multiplier
        self.t_total = t_total
        self.cycles = cycles
        self.min_lr = min_lr
        if self.warmup_multiplier < 0 or self.warmup_multiplier > 1:
            raise ValueError("warmup_multiplier must be in 0..1 range")
        super().__init__(optimizer, self.lr_lambda, last_epoch)
    
    def lr_lambda(self, step: int):
        if step < self.warmup_steps:
            # Linear warmup from warmup_start_multiplier -> 1
            f = float(step) / max(1.0, self.warmup_steps)
            return self.warmup_multiplier + (1.0 - self.warmup_multiplier) * f
        # Cosine decay to min_lr
        progress = float(step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress))
        return self.min_lr + (1.0 - self.min_lr) * cosine_decay
    
def main():
    import torch
    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR
    import matplotlib.pyplot as plt
        
    model_param = torch.nn.Parameter(torch.randn(1))
    optimizer = SGD([model_param], lr=0.1)

    # Scheduler parameters
    t_total = 1000
    warmup_steps = 100
    min_lr = 0.1
    cycles = 0.5
    warmup_start_multiplier = 0.0

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        t_total=t_total,
        min_lr=min_lr,
        cycles=cycles,
        warmup_start_multiplier=warmup_start_multiplier
    )

    # Track learning rates
    lrs = []
    for step in range(t_total):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Warmup + Cosine LR Schedule")
    plt.grid(True)
    plt.savefig("warmup_cosine_lr.png")
    print("Saved plot as warmup_cosine_lr.png")

if __name__ == "__main__":
    main()