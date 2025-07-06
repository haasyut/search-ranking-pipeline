import abc
import torch
from torch import nn


class BaseGBModule(nn.Module, abc.ABC):
    """Base class for gradient boosting modules.

    This abstract base class defines the common interface and functionality that all
    gradient boosting modules should implement.

    Attributes:
        min_hess (float) : minimum hessian value
    """

    def __init__(self, min_hess=0.0):
        super(BaseGBModule, self).__init__()
        self.min_hess = 0.0
        self.grad = None
        self.hess = None

    @abc.abstractmethod
    def _input_checking_setting(self, input_data):
        """Validate and prepare input data.

        Args:
            input_data: Input data in model-specific format

        Returns:
            Processed input data ready for model
        """
        pass

    @abc.abstractmethod
    def forward(self, input_data, return_tensor: bool = True):
        """Forward pass through the model.

        Args:
            input_data: Input data in model-specific format
            return_tensor: Whether to return predictions as PyTorch tensor

        Returns:
            Model predictions as tensor or numpy array
        """
        pass

    def _get_grad_hess_FX(self):
        grad = self.FX.grad * self.FX.shape[0]

        # parameters are independent row by row, so we can
        # at least calculate hessians column by column by
        # considering the sum of the gradient columns
        hesses = []
        for i in range(self.output_dim):
            hesses.append(
                torch.autograd.grad(grad[:, i].sum(), self.FX, retain_graph=True)[0][
                    :, i : (i + 1)
                ]
            )
        hess = torch.maximum(torch.cat(hesses, axis=1), torch.Tensor([self.min_hess]))
        return grad, hess

    @abc.abstractmethod
    def gb_step(self):
        """Perform one gradient boosting step.

        This method should implement the logic for:
        1. Getting gradients/hessians
        2. Training one boosting iteration
        3. Updating predictions
        """
        pass


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=32):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def transform(self, X_np):
        # numpy -> torch -> numpy
        with torch.no_grad():
            if hasattr(X_np, "toarray"):
                X_np = X_np.toarray()

            x_tensor = torch.from_numpy(X_np).float()
            transformed = self.forward(x_tensor).numpy()
        return transformed