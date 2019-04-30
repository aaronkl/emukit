import torch
import numpy as np
import torch.nn as nn
from pybnn.util.layers import AppendLayer


# def get_architecture(input_dimensionality: int) -> torch.nn.Module:
#     class AppendLayer(nn.Module):
#         def __init__(self, bias=True, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             if bias:
#                 self.bias = nn.Parameter(torch.DoubleTensor(1, 1))
#             else:
#                 self.register_parameter('bias', None)
#
#         def forward(self, x):
#             return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)
#
#     def init_weights(module):
#         if type(module) == AppendLayer:
#             nn.init.constant_(module.bias, val=np.log(1e-3))
#         elif type(module) == nn.Linear:
#             nn.init.xavier_normal_(module.weight)
#
#             nn.init.constant_(module.bias, val=0.0)
#
#     return nn.Sequential(
#         nn.Linear(input_dimensionality, 100), nn.Tanh(),
#         nn.Linear(100, 100), nn.Tanh(),
#         nn.Linear(100, 1),
#         AppendLayer()
#     ).apply(init_weights)
def get_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-5)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            mean = x[:, None, 0]
            #             log_var = x[:, None, 1] - 10
            #             log_var = x[:, None, 1] * (np.log(1e-1**2) - np.log(1e-3**2)) + np.log(1e-3**2)
            return self.sigma_layer(mean)

    #             std = torch.sigmoid(self.sigma_layer(x))

    #             return torch.cat((mean, log_var), dim=1)

    return Architecture(n_inputs=input_dimensionality)

# def get_cost_architecture(input_dimensionality: int) -> torch.nn.Module:
#     class Architecture(nn.Module):
#         def __init__(self, n_inputs, n_hidden=500):
#             super(Architecture, self).__init__()
#             self.fc1 = nn.Linear(n_inputs, n_hidden)
#             self.fc2 = nn.Linear(n_hidden, n_hidden)
#             self.fc3 = nn.Linear(n_hidden, 2)
#             self.sigma_layer = AppendLayer(noise=1e-3)
#
#         def forward(self, x):
#             x = torch.tanh(self.fc1(x))
#             x = torch.tanh(self.fc2(x))
#             x = self.fc3(x)
#
#             #             log_var = x[:, None, 1] - 10
#             #             log_var = x[:, None, 1] * (np.log(1e-1**2) - np.log(1e-3**2)) + np.log(1e-3**2)
#             return self.sigma_layer(x)
#
#     #             std = torch.sigmoid(self.sigma_layer(x))
#
#     #             return torch.cat((mean, log_var), dim=1)
#
#     return Architecture(n_inputs=input_dimensionality)


def get_cost_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)

            #             log_var = x[:, None, 1] - 10
            #             log_var = x[:, None, 1] * (np.log(1e-1**2) - np.log(1e-3**2)) + np.log(1e-3**2)
            return self.sigma_layer(x)

    #             std = torch.sigmoid(self.sigma_layer(x))

    #             return torch.cat((mean, log_var), dim=1)

    return Architecture(n_inputs=input_dimensionality)