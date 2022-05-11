import torch
import torch.nn.functional as F
from torch import nn

# Taken (+ modified) from: https://github.com/serre-lab/hgru-pytorch/blob/master/hgru.py
# Paper: https://arxiv.org/abs/2010.15314

class hConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        nn.init.orthogonal_(self.u1_gate.weight)
        nn.init.orthogonal_(self.u2_gate.weight)
        nn.init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1.0)  # ???
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        if self.input_size != self.hidden_size:
            self.input_gate = nn.Conv2d(input_size, hidden_size, 1)
            nn.init.orthogonal_(self.input_gate.weight)

        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        nn.init.orthogonal_(self.w_gate_inh)
        nn.init.orthogonal_(self.w_gate_exc)
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu= nn.Parameter(torch.empty((hidden_size, 1, 1)))
        nn.init.constant_(self.alpha, 0.1)
        nn.init.constant_(self.gamma, 1.0)
        nn.init.constant_(self.kappa, 0.5)
        nn.init.constant_(self.w, 0.5)
        nn.init.constant_(self.mu, 1)

        self.bn = nn.ModuleList([nn.GroupNorm(hidden_size, hidden_size) for i in range(4)])
        # self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(4)])       
        for bn in self.bn:
            nn.init.constant_(bn.weight, 0.1)

    def forward(self, input_, state):  # , time=0):
        
        if self.input_size != self.hidden_size:
            input_ = self.input_gate(input_)
        if torch.sum(state) == 0.0:  # if time == 0:
            state = torch.empty_like(input_)
            nn.init.xavier_normal_(state)

        g1_t = torch.sigmoid(self.bn[0](self.u1_gate(state)))
        c1_t = self.bn[1](F.conv2d(state * g1_t, self.w_gate_inh, padding=self.padding))
        next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*state + self.mu)))
        g2_t = torch.sigmoid(self.bn[2](self.u2_gate(next_state1)))
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
        h2_t = F.relu(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)           
        return (1 - g2_t) * state + g2_t * h2_t
