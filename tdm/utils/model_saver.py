import os

import torch


class ModelSaver:
    def __init__(self, save_dir, mode='max'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(save_dir)
        self.mode = mode
        if self.mode == 'max':
            self.best_score = 0
        elif self.mode == 'min':
            self.best_score = float('inf')
        else:
            raise AttributeError(
                f"Expected mode to be 'min' or 'max'. Got {self.mode}.")

    def update(self, model, score):
        if self.mode == 'max':
            if score >= self.best_score:
                self._save(model, score)

        elif self.mode == 'min':
            if score <= self.best_score:
                self._save(model, score)

    def _save(self, model, score):
        torch.save(model, os.path.join(self.save_dir, 'best_model.pt'))
        self.best_score = score
