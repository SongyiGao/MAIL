import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tianshou.data import Batch

class GwmGailDataset(Dataset):
    def __init__(self,batch_file='train_batch.pkl',trj_steps=1799,trj_clip_steps=6):
        with open(batch_file, 'rb') as f:
            self.batch,indices = pickle.load(f)
        
        trj_nums = len(indices) % trj_steps
        assert len(indices) > 0 and  trj_nums == 0
        self.sample_indices = []
        for i in range(len(indices) // trj_steps):
            self.sample_indices +=  [i * trj_steps + j 
                               for j in range(trj_steps)
                               if j <= (trj_steps - trj_clip_steps)]
            
        self.trj_steps = trj_steps
        self.trj_clip_steps = trj_clip_steps
        self.done = np.array([False for _ in range(trj_clip_steps-1)] + [True,])
        
    def __len__(self):
        return len(self.sample_indices)
        
    def __getitem__(self, idx):
        start_index = self.sample_indices[idx]
        clip_trj = self.batch[start_index:start_index+self.trj_clip_steps]
        clip_trj.done = self.done
        
        return clip_trj
    
    def trj_sample(self,trj_num=None):
        if trj_num is None:
            indices = np.random.permutation(self.__len__())
        else:
            assert trj_num <= self.__len__()
            indices = np.random.permutation(self.__len__())[:trj_num]
        batch_list = [self.__getitem__(i) for i in indices]

        batch = Batch()
        batch.cat_list(batch_list)
        #print(len(batch),batch.keys())
        #print(aaa)
        return batch
    
class GwmVaeDataset(Dataset):
    def __init__(self,batch_file='train_batch.pkl',trj_steps=1799,trj_clip_steps=6,val=False):
        with open(batch_file, 'rb') as f:
            self.batch,indices = pickle.load(f)
        
        trj_nums = len(indices) % trj_steps
        assert len(indices) > 0 and  trj_nums == 0
        self.sample_indices = []
        for i in range(len(indices) // trj_steps):
            self.sample_indices +=  [i * trj_steps + j 
                               for j in range(trj_steps)
                               if j <= (trj_steps - trj_clip_steps)]
            
        self.trj_steps = trj_steps
        self.trj_clip_steps = trj_clip_steps
        self.done = np.array([False for _ in range(trj_clip_steps-1)] + [True,])
        if not val:
            self.batch = self.batch[:self.trj_steps*16]
        else:
            self.batch = self.batch[self.trj_steps*16:]
        
    def __len__(self):
        return len(self.batch)
        
    def __getitem__(self, idx):
        s = self.batch[idx].obs.s[-1].astype('float')
        return torch.from_numpy(s)
    
        
    
if __name__ == "__main__":
    data = GwmGailDataset()
    
    import time
    t1 = time.time()
    for i in range(len(data)):
        d = data[i]
    print("采样全部样本时间: ", time.time() - t1)
    
    t1 = time.time()
    steps = 5
    for i in range(steps):
        batch_trj = data.trj_sample()
    print("采样全部轨迹时间: ", (time.time() - t1) / steps)