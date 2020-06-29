import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class f_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(f_mlp, self).__init__()
        self.l1 = nn.Linear(input_dim[1], hidden_dim * 2)
        self.l2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        b, t, c = x.shape
        x = x.view(-1, c)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x.view(b, t, -1)

        return x

class b_mlp(nn.Module):
    def __init__(self, b_mlp_layers_input_dim, hidden_dim, output_dim):
        super(b_mlp, self).__init__()
        self.l1 = nn.Linear(b_mlp_layers_input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))

        return x

class mlp_rnn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, bidirectional=False):
        super(mlp_rnn, self).__init__()

        self.f_mlp = f_mlp(input_dim, hidden_dim)

        self.rnn_module = nn.LSTM(input_size=hidden_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=4,
                                  batch_first=True,
                                  dropout=0,
                                  bidirectional=True if bidirectional else False)

        b_mlp_layers_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.b_mlp = b_mlp(b_mlp_layers_input_dim * input_dim[0], output_dim, output_dim)

    def forward(self, x):
        
        x = self.f_mlp(x)
        x, _ = self.rnn_module(x)
        b, t, c = x.shape
        # x = x[:,t//2,:]
        x = x.contiguous().view(b, -1)

        x = self.b_mlp(x)

        return x

class Mlp_Encoder(nn.Module):
    def __init__(self, input_dim_x, input_dim_p, output_dim, depth = 3):
        super(Mlp_Encoder, self).__init__()
        hidden_dim_x = output_dim * 2 if input_dim_x > output_dim else output_dim // 2
        self.pre_x = nn.ModuleList([nn.Linear(input_dim_x, hidden_dim_x)] + \
                                   [nn.Linear(hidden_dim_x, hidden_dim_x) for i in range(2)] + \
                                   [nn.Linear(hidden_dim_x, output_dim)] )

        hidden_dim_p = output_dim * 2 if input_dim_p > output_dim else output_dim // 2
        self.pre_p = nn.ModuleList([nn.Linear(input_dim_p, hidden_dim_p)] + \
                                   [nn.Linear(hidden_dim_p, hidden_dim_p) for i in range(2)] + \
                                   [nn.Linear(hidden_dim_p, output_dim)] )

        self.encoder = nn.ModuleList([nn.Linear(2*output_dim, output_dim)] + \
                                     [nn.Linear(output_dim, output_dim) for i in range(depth)] )


    def forward(self, x, p):
        for layer in self.pre_x:
            x = F.relu(layer(x))

        for layer in self.pre_p:
            p = F.relu(layer(p))

        output = torch.cat([x,p],1)
        for layer in self.encoder:
            output = torch.tanh(layer(output))

        return output

class Driver_Net(nn.Module):
    def __init__(self, p_dim, s_dim, v_dim, a_dim, hidden_dim=512, depth=10):
        super(Driver_Net, self).__init__()
        self.s_encoder = mlp_rnn(s_dim, hidden_dim)
        self.sp_encoder = Mlp_Encoder(hidden_dim, p_dim, hidden_dim)
        self.vp_encoder = Mlp_Encoder(v_dim*2, p_dim, hidden_dim)
        self.sv_encoder = Mlp_Encoder(hidden_dim, hidden_dim, hidden_dim)
        self.svp_encoder = Mlp_Encoder(hidden_dim, p_dim, hidden_dim)

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(depth)])

        self.linear_output = nn.Linear(hidden_dim, a_dim)

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))
    def forward(self, p, s, v, v_next):
        s = self.s_encoder(s)
        sp = self.sp_encoder(s ,p)
        vp = self.vp_encoder(torch.cat([v,v_next],1),p)
        sv = self.sv_encoder(sp,vp)
        output = self.svp_encoder(sv,p)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        output = self.linear_output(output)
        return output


class Driver_Net_Cat(nn.Module):
    def __init__(self, p_dim, s_dim, v_dim, a_dim, hidden_dim=512, depth=10):
        super(Driver_Net_Cat, self).__init__()
        self.s_encoder = mlp_rnn(s_dim, hidden_dim)
        self.encoder = nn.ModuleList([nn.Linear(hidden_dim+p_dim+v_dim+v_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(depth)])

        self.linear_output = nn.Linear(hidden_dim, a_dim)

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))
    def forward(self, p, s, v, v_next):
        s = self.s_encoder(s)
        output = torch.cat([s,p,v,v_next],1)

        for layer in self.encoder:
            output = layer(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        output = self.linear_output(output)
        return output

class Motive_Net_Cat(nn.Module):
    def __init__(self, p_dim, s_dim, a_dim, o_dim, hidden_dim=512, depth=10):
        super(Motive_Net_Cat, self).__init__()
        self.s_encoder = mlp_rnn(s_dim, hidden_dim)
        self.encoder = nn.ModuleList([nn.Linear(hidden_dim+p_dim+a_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(depth)])

        self.linear_o = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)] + \
                                      [nn.ReLU(), nn.Linear(hidden_dim, o_dim)])

        self.linear_s = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)] + \
                                      [nn.ReLU(), nn.Linear(hidden_dim, s_dim[-1])])

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, p, s, a):
        s = self.s_encoder(s)
        output = torch.cat([s,p,a],1)

        for layer in self.encoder:
            output = layer(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        for layer in self.linear_o:
            output_o = layer(output)

        for layer in self.linear_s:
            output_s = layer(output)

        return output_o,output_s

class DiagGaussian(torch.distributions.Normal):
    """Diagonal Gaussian distribution."""

    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

class Driver_Net_Cat_PPO(nn.Module):
    def __init__(self, p_dim, s_dim, v_dim, a_dim, s_encoder, hidden_dim=512, depth=10, sigma_train=False, _max=1):
        super().__init__()
        self.dist_fn = DiagGaussian
        self.sigma_train = sigma_train
        self._max = _max
        self.s_encoder = s_encoder
        self.encoder = nn.Sequential(*[nn.Linear(hidden_dim+p_dim+v_dim+v_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])

        self.mu = nn.Linear(hidden_dim, a_dim)
        
        if self.sigma_train:
            self.sigma = nn.Linear(hidden_dim, a_dim)
        else:
            self.sigma = nn.Parameter(torch.zeros(a_dim, 1))

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, p, s, v, v_next):
        s = self.s_encoder(s)
        output = torch.cat([s,p,v,v_next],1)
        output = self.encoder(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        mu = self._max * torch.tanh(self.mu(output))
        
        if self.sigma_train:
            sigma = torch.exp(self.sigma(output))
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
            
            
        return (mu, sigma)

class Motive_Net_Cat_PPO(nn.Module):
    def __init__(self, p_dim, s_dim, a_dim, o_dim,s_encoder, hidden_dim=512, depth=10, sigma_train=False, _max=1):
        super().__init__()
        self.dist_fn = DiagGaussian
        self.sigma_train = sigma_train
        self._max = _max
        self.s_encoder = s_encoder
        self.encoder = nn.Sequential(*[nn.Linear(hidden_dim+p_dim+a_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(depth)])
        
        self.output_o = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, o_dim))
        self.output_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(), 
                                      nn.Linear(hidden_dim, s_dim[-1]))

        #self.mu = nn.Linear(o_dim+s_dim[-1], o_dim+s_dim[-1])
        
        if self.sigma_train:
            self.sigma = nn.Linear(o_dim+s_dim[-1], o_dim+s_dim[-1])
        else:
            self.sigma = nn.Parameter(torch.zeros(o_dim+s_dim[-1], 1))
        
        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, p, s, a):
        s = self.s_encoder(s)
        output = torch.cat([s,p,a],1)
        output = self.encoder(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))
            
        output_o = self.output_o(output)
        output_s = self.output_s(output)
        mu = self._max * torch.tanh(torch.cat([output_o,output_s],1))
        
        #mu = self._max * torch.tanh(self.mu(output))
        if self.sigma_train:
            sigma = torch.exp(self.sigma(torch.cat([output_o,output_s],1)))
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
            
            
        return (mu, sigma)
    
class Critic(nn.Module):
    def __init__(self, p_dim, s_dim, v_dim, a_dim,s_encoder, hidden_dim=512, depth=10,):
        super().__init__()        
        self.s_encoder = s_encoder
        self.encoder = nn.Sequential(*[nn.Linear(hidden_dim+p_dim+v_dim+v_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])

        self.v = nn.Sequential(nn.Linear(hidden_dim, 128),
                               nn.ReLU(),
                               nn.Linear(128, 1),
                               )

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, p, s, v, v_next):
        s = self.s_encoder(s)
        output = torch.cat([s,p,v,v_next],1)
        output = self.encoder(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        logits = self.v(output)
        #logits = F.sigmoid(logits)
        #print(logits)
        return logits
    

class Discriminator(nn.Module):
    def __init__(self, s_dim, a_dim, o_dim, hidden_dim=512, depth=3):
        super().__init__()
        self.encoder = nn.ModuleList([nn.Linear(s_dim[1]*2+a_dim+o_dim, hidden_dim),nn.ReLU()])

        self.resweight = nn.Parameter(torch.zeros(depth), requires_grad=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(depth)])

        self.linear_output = nn.Sequential(*[nn.Linear(hidden_dim, 128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 1)])

        for i in range(depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

    def forward(self, s, s_next, a, o):
        output = torch.cat([s,s_next,a,o],1)

        for layer in self.encoder:
            output = layer(output)

        for i, _ in enumerate(self.linear_layers):
            output = output + self.resweight[i] *  torch.relu(self.linear_layers[i](output))

        output = self.linear_output(output)
        output = F.sigmoid(output)

        return output


if __name__ == "__main__":
    p = torch.ones([2,5000])
    s = torch.ones([2,6,512]) * 0.1
    v = torch.ones([2,1])
    v_next = torch.ones([2,1])
    a = torch.ones([2,16])
    o = torch.ones([2,4]) * -0.1

    net = Driver_Net(5000,[6,512],1,16)
    net = Motive_Net_Cat(5000,[6,512],16,4)
    net = Driver_Net_Cat(5000,[6,512],1,16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01,)
    Loss = nn.L1Loss()
    for i in range(10000):
        a_pre = net(p, s, v, v_next)
        optimizer.zero_grad()
        loss = Loss(a_pre, a)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Step: ",i, "Loss:",loss.item())

    for i in range(10000):
        o_pre, s_pre = net(p, s, a)
        optimizer.zero_grad()
        loss = Loss(o_pre, o) + Loss(s_pre, s[:,-1,:])
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Step: ",i, "Loss:",loss.item())