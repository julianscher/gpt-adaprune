

class CustomLRScheduler:
    def __init__(self, initial_lr: int, decay_strategy: str="step", decay_rate: float=0.5,
                 step_size: float=1000, min_lr: float=1e-6) -> None:
        self.initial_lr = float(initial_lr)
        self.current_lr = float(initial_lr)
        self.decay_strategy = decay_strategy
        self.decay_rate = float(decay_rate)
        self.step_size = int(step_size)
        self.min_lr = float(min_lr)
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1

        if self.decay_strategy == "step":
            if self.epoch % self.step_size == 0:
                self.current_lr = max(self.min_lr, self.current_lr * self.decay_rate)

        elif self.decay_strategy == "exponential":
            self.current_lr = max(self.min_lr, self.initial_lr * (self.decay_rate ** self.epoch))

        elif self.decay_strategy == "linear":
            self.current_lr = max(self.min_lr, self.initial_lr - self.decay_rate * self.epoch)

    def get_lr(self) -> float:
        return self.current_lr
