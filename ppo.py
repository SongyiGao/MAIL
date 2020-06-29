import time
import copy
import torch
import pickle
import random
import numpy as np
from torch import nn
from typing import Dict, List, Tuple, Union, Optional

from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch


class PPOPolicy(PGPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic
        network.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1], defaults to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation,
        defaults to ``None``.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper, defaults to 0.2.
    :param float vf_coef: weight for value loss, defaults to 0.5.
    :param float ent_coef: weight for entropy loss, defaults to 0.01.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation, defaults to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound,
        defaults to 5.0 (set ``None`` if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1,
        defaults to ``True``.
    :param bool reward_normalization: normalize the returns to Normal(0, 1),
        defaults to ``True``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 nets, optims, 
                 dist_fn: torch.distributions.Distribution,
                 discount_factor: float = 0.99,
                 max_grad_norm: Optional[float] = None,
                 eps_clip: float = .2,
                 vf_coef: float = .5,
                 ent_coef: float = .01,
                 action_range: Optional[Tuple[float, float]] = None,
                 gae_lambda: float = 0.95,
                 dual_clip: Optional[float] = None,
                 value_clip: bool = True,
                 reward_normalization: bool = True,
                 params = None,
                 batch_file = None,
                 **kwargs) -> None:
        super().__init__(None, None, dist_fn, discount_factor, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = True
        self.driver, self.motive, self.critic, self.discriminator = nets
        self.optim_p, self.optim_d = optims
        assert 0 <= gae_lambda <= 1, 'GAE lambda should be in [0, 1].'
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1, \
            'Dual-clip PPO parameter should greater than 1.'
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        self.device = params.device
        self.trj_steps = params.trj_steps
        self.trj_clip_steps = params.trj_clip_steps
        self.d_loss = None
 

    def get_reward_by_discriminator(self,batch):           
        rews = []       
        v_ = []
        for b in batch.split(self.step_batch_size, shuffle=False):
            expert_label = torch.full((len(b), 1), 1)
            with torch.no_grad():
                prob_expert = self.discriminator(b.obs.s[:, -1, :].cuda(), b.act.s_next.cuda(), b.act.a.cuda(), b.act.o.cuda())
                rews.append(-torch.log(1-prob_expert))
                v_.append(self.critic(b.obs.p.cuda(), 
                                      b.obs.s.cuda(), 
                                      b.obs.v.cuda(), 
                                      b.obs.v_next.cuda()
                                     ))
        rew = torch.cat(rews, dim=0).contiguous().cpu().view(-1)
        v_ = torch.cat(v_, dim=0).contiguous().cpu().view(-1)
        return rew,v_
    
    
    def get_mask_indices(self,):
        #获得不用数据的索引和done
        is_done = True 
        not_done = False
        
        indices=self.indices
        trj_steps=self.trj_steps
        trj_clip_steps=self.trj_clip_steps

        use_steps = trj_steps// trj_clip_steps * trj_clip_steps

        def get_done(index):
            return is_done if (index+1) % 5 ==0 else not_done

        mask_indices = []
        done = np.full_like(indices, is_done).astype(bool)
        copy_done = np.array([get_done(i) for i in range(use_steps) ])

        for ulp_index in range(indices.shape[0] // trj_steps):

            start_index = random.choice(range(trj_clip_steps-1)) 
            stop_index = use_steps + start_index

            start_index = start_index + (ulp_index * trj_steps)
            stop_index = stop_index + (ulp_index * trj_steps)

            done[start_index:stop_index] = copy_done

            mask_indices += range((ulp_index * trj_steps), start_index)
            mask_indices += range(stop_index, (ulp_index+1) * trj_steps)

        return mask_indices,done
        

    def process_fn(self, batch: Batch, buffer=None,
                   indice=None) -> Batch:
        v_ = None
        rew,v_ = self.get_reward_by_discriminator(batch)

        batch.rew = rew 

        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if not np.isclose(std.cpu().numpy(), 0):
                batch.rew = (batch.rew - mean) / std
        if self._lambda in [0, 1]:
            return self.compute_episodic_return(
                batch, None, gamma=self._gamma, gae_lambda=self._lambda)
        batch.to_numpy()
        batch = self.compute_episodic_return(
            batch, v_, gamma=self._gamma, gae_lambda=self._lambda)
        batch.to_torch()
        return batch
        
    def trj_by_policy(self,batch):
        #return batch
        assert len(batch) % self.trj_clip_steps == 0
        infer_batch = None
        old_step_batch = None
        o_len = batch.act.o.shape[-1]
        policy_batch_list = []
        for step in range(self.trj_clip_steps):
            indices = np.arange(step, len(batch), self.trj_clip_steps,)
            step_batch = batch[indices]
            if infer_batch is not None:
                #修正obs
                #print(step)
                #print(step_batch.obs.s[0])
                s = torch.cat([old_step_batch.obs.s, 
                                              infer_batch.act_m[:,o_len:].cpu().reshape(len(old_step_batch),1,-1)],1)
                s = s[:,1:,:]
                step_batch.obs.s = s
                #print(step_batch.obs.s[0])
                
                del infer_batch
                
            infer_batch = self(step_batch)

            #修正act
            #print(step_batch.act)
            step_batch.act.a = infer_batch.act_d.cpu()
            step_batch.act.o = infer_batch.act_m[:,:o_len].contiguous().cpu()
            step_batch.act.s_next = infer_batch.act_m[:,o_len:].contiguous().cpu()
            #print(step_batch.act)

            old_step_batch = step_batch
            policy_batch_list.append(step_batch)

        policy_batch = Batch()
        policy_batch.cat_list(policy_batch_list)


        re_indices = []
        for i in range(len(policy_batch_list[0])):
            for j in range(self.trj_clip_steps):
                re_indices.append(i + (j*len(policy_batch_list[0])))
        re_indices = np.array(re_indices).astype(np.int32)      
        policy_batch = policy_batch[re_indices]

        return policy_batch
            
            


    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
  
        logits_d = self.driver(batch.obs.p.cuda(), batch.obs.s.cuda(), batch.obs.v.cuda(), batch.obs.v_next.cuda(),)
        if isinstance(logits_d, tuple):
            dist_d = self.dist_fn(*logits_d)
        else:
            dist_d = self.dist_fn(logits_d)
        act_d = dist_d.sample()
        
        if self._range:
            act_d = act_d.clamp(-1, 1)
        
        logits_m = self.motive(batch.obs.p.cuda(), batch.obs.s.cuda(), act_d.detach(),)
        if isinstance(logits_m, tuple):
            dist_m = self.dist_fn(*logits_m)
        else:
            dist_m = self.dist_fn(logits_m)
        act_m = dist_m.sample()

        if self._range:
            act_m = act_m.clamp(-1, 1)

        return Batch(logits_d = logits_d,
                     logits_m = logits_m,
                     act_d = act_d,
                     act_m = act_m,
                     dist_d = dist_d,
                     dist_m = dist_m)

    def learn(self, batch, step_batch_size: int, repeat: int, epoch : int,
              **kwargs) -> Dict[str, List[float]]:
        self.step_batch_size = step_batch_size
        #1.生成轨迹
        batch_expert = batch
        batch_policy = copy.deepcopy(batch)
        batch_policy = self.trj_by_policy(batch_policy)
        #print(batch_policy)
        #2.训练判别器
        loss_d = torch.nn.MSELoss(reduce=True, size_average=True)
        d_loss = None
        if True:
            expert_label = torch.full((step_batch_size, 1), 1)
            policy_label = torch.full((step_batch_size, 1), 0)
            if torch.cuda.is_available():
                expert_label = expert_label.cuda()
                policy_label = policy_label.cuda()
            discriminator_batch =  zip(batch_expert.split(step_batch_size, shuffle=True),
                                             batch_policy.split(step_batch_size, shuffle=True))
            for b_expert,b_policy in discriminator_batch:
                prob_expert = self.discriminator(b_expert.obs.s[:, -1, :].cuda(),
                                                 b_expert.act.s_next.cuda(), 
                                                 b_expert.act.a.cuda(), 
                                                 b_expert.act.o.cuda())
                                                 
                prob_policy = self.discriminator(b_policy.obs.s[:, -1, :].cuda(), 
                                                 b_policy.act.s_next.cuda(), 
                                                 b_policy.act.a.cuda(), 
                                                 b_policy.act.o.cuda())
                
                

                expert_loss = loss_d(prob_expert, expert_label)
                policy_loss = loss_d(prob_policy, policy_label)
                d_loss  = (expert_loss.mean()+policy_loss.mean()) / 2.0
                self.optim_d.zero_grad()
                d_loss.backward()
                if d_loss.item() >= 0.1:
                    self.optim_d.step()
                else:
                    break
                
        
        #3.使用训练后的判别器对生成轨迹计算reward并应用GAE;
        batch_policy = self.process_fn(batch_policy)
        #4.使用PPO训练策略网络
        losses, clip_losses, vf_losses, ent_losses, adv_list = [], [], [], [], []
        v = []
        old_log_prob_d = []
        old_log_prob_m = []
        with torch.no_grad():
            #critic评估当前状态价值
            for b in batch_policy.split(self.step_batch_size, shuffle=False):
                v.append(self.critic(b.obs.p.cuda(), 
                                     b.obs.s.cuda(), 
                                     b.obs.v.cuda(), 
                                     b.obs.v_next.cuda()
                                     )
                                        )
                batch_infer = self(b)
                old_log_prob_d.append(batch_infer.dist_d.log_prob(b.act.a.cuda()).detach())
                old_log_prob_m.append(batch_infer.dist_m.log_prob(torch.cat([b.act.o, b.act.s_next - b.obs.s[:, -1, :]],1).cuda()).detach())
                
        batch_policy.v = torch.cat(v, dim=0).cpu()
        #batch.act = to_torch(batch.act, v[0])
        batch_policy.old_log_prob_d = torch.cat(old_log_prob_d, dim=0)
        batch_policy.old_log_prob_m = torch.cat(old_log_prob_m, dim=0)
        batch_policy.returns = batch_policy.returns.reshape(batch_policy.v.shape)

        #print(a)
        
        if self._rew_norm:
            mean, std = batch_policy.returns.mean(), batch_policy.returns.std()
            if not np.isclose(std.item(), 0):
                batch_policy.returns = (batch_policy.returns - mean) / std
        batch_policy.adv = batch_policy.returns - batch_policy.v
        #print(batch_policy.adv)
        if self._rew_norm:
            mean, std = batch_policy.adv.mean(), batch_policy.adv.std()
            if not np.isclose(std.item(), 0):
                batch_policy.adv = (batch_policy.adv - mean) / std 
        #print(batch_policy.adv)        
        for _ in range(repeat):
            for b in batch_policy.split(self.step_batch_size, shuffle=True):
                batch_dist = self(b)
                dist_d = batch_dist.dist_d
                dist_m = batch_dist.dist_m
                value = self.critic(b.obs.p.cuda(), 
                                     b.obs.s.cuda(), 
                                     b.obs.v.cuda(), 
                                     b.obs.v_next.cuda()
                                     )
            
                ratio = ((dist_d.log_prob(b.act.a.cuda()) \
                         +dist_m.log_prob(torch.cat([b.act.o, b.act.s_next - b.obs.s[:, -1, :]],1).cuda())) \
                         - (b.old_log_prob_d + b.old_log_prob_m)).exp().float()
                
                #time.sleep(1)
                b.adv = b.adv.cuda()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                
                if self._value_clip:
                    b.v = b.v.cuda()
                    b.returns = b.returns.cuda()
                    v_clip = b.v + (value - b.v).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    b.returns = b.returns.cuda()
                    vf_loss = .5 * (b.returns - value.reshape(-1)).pow(2).mean()
                
                e_loss = (dist_d.entropy().mean() + dist_m.entropy().mean()) / 2.0
                
                #loss = vf_loss
                loss = clip_loss + self._w_vf * vf_loss  - self._w_ent * e_loss
                #loss = clip_loss
                losses.append(loss.item())   
                clip_losses.append(clip_loss.item())
                vf_losses.append(self._w_vf * vf_loss.item())
                ent_losses.append(-self._w_ent * e_loss.item())
                adv_list.append(b.adv.mean().item())
                
                self.optim_p.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.driver.parameters()) 
                                         + list(self.motive.parameters())
                                         + list(self.critic.parameters()),
                    self._max_grad_norm)
                self.optim_p.step()
                
                #5.记录训练过程信息
                print("Loss:{},clip_loss:{};vf_loss:{};e_loss:{};adv:{};Dis:{}".format(loss.item(),
                                                                          clip_loss.item(),
                                                                 self._w_vf * vf_loss.item(),
                                                                self._w_ent * e_loss.item(),
                                                                b.adv.mean().item(),
                                                                d_loss.item() ))  
        torch.cuda.empty_cache()
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
            'loss/discriminator':[d_loss.item(),],
            'adv':adv_list,
        }
