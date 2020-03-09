import torch
import torch.nn as nn
from src.utils.ConfigBase import ConfigBase


class MLPConfig(ConfigBase):

    def __init__(self, name='mlp', mlp_conf=None):
        super(MLPConfig, self).__init__(name=name, mlp=mlp_conf)

        self.mlp = {
            'input_dimension': 32,
            'output_dimension': 32,
            'activation': 'mish',
            'out_activation': 'mish',
            'num_neurons': [32],
        }


TORCH_ACTIVATION_LIST = ['ReLU',
                         'Sigmoid',
                         'SELU',
                         'leaky_relu',
                         'Softplus']

ACTIVATION_LIST = ['mish', 'swish', None]


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(nn.functional.softplus(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


def get_nn_activation(activation: 'str'):
    if not activation in TORCH_ACTIVATION_LIST + ACTIVATION_LIST:
        raise RuntimeError("Not implemented activation function!")

    if activation in TORCH_ACTIVATION_LIST:
        act = getattr(torch.nn, activation)()

    if activation in ACTIVATION_LIST:
        if activation == 'mish':
            act = Mish()
        elif activation == 'swish':
            act = Swish()
        elif activation is None:
            act = nn.Identity()

    return act


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 num_neurons=[32, 32],
                 activation='mish',
                 out_activation='mish',
                 ):

        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dimension
        self.output_dim = output_dimension
        self.num_neurons = num_neurons
        self.activation = get_nn_activation(activation)
        self.out_activation = get_nn_activation(out_activation)

        self.num_layers = len(num_neurons)

        input_dims = [input_dimension] + num_neurons
        output_dims = num_neurons + [output_dimension]

        # Input -> the last hidden layer
        self.layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims[:-1], output_dims[:-1])):
            self.layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

        self.layers.append(nn.Linear(in_features=input_dims[-1], out_features=output_dims[-1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.num_layers:
                x = self.activation(x)
            else:
                x = self.out_activation(x)

        return x

    def check_input_spec(self, input_spec):
        if isinstance(input_spec, list):
            # output layer will not be normalized
            assert len(input_spec) == len(self.num_neurons) + 1, "the length of input_spec list should " \
                                                                 "match with the number of hidden layers + 1"
            _list_type = True
        else:
            _list_type = False

        return _list_type


if __name__ == "__main__":
    A = MLPConfig()
