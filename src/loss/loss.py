from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss  = np.mean(sample_losses)
        return data_loss