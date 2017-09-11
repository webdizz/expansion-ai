# AI model for WarLight

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import Params

logger = logging.getLogger('Model')


def normalized_columns_initializer(weights, std_div=1.0):
    """ Initialize and set the variance of a tensor of weights """
    output = torch.randn(weights.size())
    # thanks to this initialization, we have var(out) = std^2
    output *= std_div / torch.sqrt(output.pow(2).sum(1, keepdim=True))
    return output


def init_weights(model):
    """ Initialize weights of the NN for an optimal learning """
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        weight_shape = list(model.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])  # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]  # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / fan_in + fan_out)
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
        weight_shape = list(model.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space, params=Params()):
        """ Initis Actor Critic """
        super(ActorCritic, self).__init__()
        self.params = params

        logger.info("Init ActorCritic num_inputs: %s and action_space: %s" % (num_inputs, action_space))
        hidden_layer_neuron_size = self.params.board_size * self.params.board_size
        self.fc1 = nn.Linear(self.params.board_size, hidden_layer_neuron_size)
        self.fc2 = nn.Linear(hidden_layer_neuron_size, hidden_layer_neuron_size)
        self.fc3 = nn.Linear(hidden_layer_neuron_size, num_inputs)

        self.lstm = nn.LSTMCell(self.params.board_dimension_size * self.params.board_size, self.params.lstm_size)

        # num_outputs = int(action_space.high[0])  # action_space.num_discrete_space
        num_outputs = int(action_space.n)  # action_space.num_discrete_space

        self.critic_linear = nn.Linear(256, 1)  # output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs)  # output = Q(S, a)

        self.apply(init_weights)

        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, current_state):
        current_state, (hx, cx) = current_state
        x = F.relu(self.fc1(current_state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # flattening the last layer into this 1D vector x
        x = x.view(-1, self.params.board_dimension_size * self.params.board_size)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        # print("= AC input size {}".format(x.size()))
        critic_of_x = self.critic_linear(x)
        actor_of_x = self.actor_linear(x)
        return critic_of_x, actor_of_x, (hx, cx)
