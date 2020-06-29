from collections import namedtuple
import numpy as np
import torch
import time
import tianshou
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy.dist import DiagGaussian

from ppo import PPOPolicy
from trj_sample import GwmGailDataset
from Ee_Dataset import Drive_control_Dataset

from rl_net import Driver_Net_Cat_PPO,Motive_Net_Cat_PPO,Discriminator,Critic,mlp_rnn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

writer = SummaryWriter("./output/log")

def get_net(data):
            
    p_dim = data.obs.p[0].shape[0]
    s_dim = data.obs.s[0].shape
    v_dim = data.obs.v[0].shape[0]
    a_dim = data.act.a[0].shape[0]
    o_dim = data.act.o[0].shape[0]
    #print(p_dim, s_dim, v_dim, a_dim, o_dim)
    
    s_encoder = mlp_rnn(s_dim, output_dim=512)

    driver_net = Driver_Net_Cat_PPO(p_dim, s_dim, v_dim, a_dim, s_encoder)
    motive_net = Motive_Net_Cat_PPO(p_dim, s_dim, a_dim, o_dim, s_encoder)
    critic_net = Critic(p_dim, s_dim, v_dim, a_dim, s_encoder)
    discriminator = Discriminator(s_dim, a_dim, o_dim)

    return [driver_net, motive_net, critic_net, discriminator]

def get_dataloader_and_net(train_file="./train_batch.pkl", val_file="./train_batch.pkl"):
    train_loader = GwmGailDataset("./train_batch.pkl")
    val_loader = None

    nets = get_net(train_loader[0])
    dataloaders = (train_loader, val_loader)

    return dataloaders, nets
        
        
def train(dataloaders, nets, params):
    train_loader, val_loader = dataloaders
    optims = get_optim(nets)
    dist = DiagGaussian
    
    policy = PPOPolicy(
        nets, optims, dist, params.gamma,
        max_grad_norm=params.max_grad_norm,
        eps_clip=params.eps_clip,
        vf_coef=params.vf_coef,
        ent_coef=params.ent_coef,
        reward_normalization=params.rew_norm,
        # dual_clip=args.dual_clip,
        # dual clip cause monotonically increasing log_std :)
        value_clip=params.value_clip,
        # action_range=[env.action_space.low[0], env.action_space.high[0]],)
        # if clip the action, ppo would not converge :)
        gae_lambda=params.gae_lambda,
        batch_file = params.batch_file,
        params=params)
    
    
    for epoch in range(params.epochs):
        batch = train_loader.trj_sample(params.trj_num)
        batch.to_torch(torch.float)
        loss = policy.learn(batch, 
                            params.step_batch_size,
                            repeat = 1,
                            epoch = epoch)
        for k,v in loss.items():
            writer.add_scalar('Train/'+k, np.mean(np.array(v)), epoch)
        


def get_optim(nets):
    driver_net, motive_net, critic_net, discriminator = nets
    
    for m in list(driver_net.modules()) + list(motive_net.modules())+ list(critic_net.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    optim_p = torch.optim.Adam(list(set(list(driver_net.parameters()) 
                             + list(motive_net.parameters())
                             + list(critic_net.parameters()))), 
                             lr=params.lr_g,
                             weight_decay=params.weigt_deacy)
    
    optim_d = torch.optim.Adam(discriminator.parameters(),
                                 lr=params.lr_d,
                                 weight_decay=params.weigt_deacy)

    return optim_p, optim_d


def main(params):
    dataloaders, nets = get_dataloader_and_net()
    
    if torch.cuda.is_available():
        for i,net in enumerate(nets):
             nets[i] = net.cuda()

    train(dataloaders, nets, params)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


if __name__ == "__main__":
    params = AttributeDict()
    params.update({
        "epochs": 10000,
        "lr_g": 0.00005,
        "lr_d": 0.00005,
        "trj_steps" : 1799,
        "trj_clip_steps" : 6,
        "step_batch_size": 1024,
        "trj_num": 1024*8,
        "weigt_deacy": 1e-4,
        "gamma" : 0.99,
        "max_grad_norm" : 0.5,
        "eps_clip" : 0.2,
        "vf_coef" : 0.5,
        "ent_coef" : 0.01,
        "rew_norm" : 1,
        "value_clip" : 1,
        "gae_lambda" : 0.95,
        "batch_file" : "train_batch.pkl",
        'device': 'cpu',
        }
    )
    torch.cuda.set_device(6)
    main(params)