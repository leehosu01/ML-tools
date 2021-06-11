import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
class LRFinder(Callback):
    """
    Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.

    baseline from https://www.avanwyk.com/finding-a-learning-rate-in-tensorflow-2/

    add best_lr by @leehosu01
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.75):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def best_lr(self):
        #loss_crit = np.percentile(self.losses, [75])
        #return min([(self.losses[i+1]-self.losses[i-1], self.lrs[i]) for i in range(1, len(self.losses) - 1) if self.losses[i] < loss_crit])[1]
        return min([(self.losses[i+1]-self.losses[i-1], self.lrs[i]) for i in range(10, len(self.losses) - 10) if self.losses[i-10] > self.losses[i+10]])[1]
            
    def plot(self, title = None):
        if title == None:
            title = 'best_lr = %.3e'%self.best_lr()
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.set_title(title)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.set_ylim([min(self.losses)-0.01, max(self.losses[0] + 0.01, np.percentile(self.losses, [75])) ])
        ax.axvline(x=self.best_lr())
        ax.plot(self.lrs, self.losses)
