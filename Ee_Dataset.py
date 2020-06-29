#!/usr/bin/env python
# Created at 2020/5/28
import warnings
import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import preprocessing

warnings.filterwarnings("ignore")

def max_min_scaler(data,cols):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 0.000001) * 2 - 1
    data[cols] = data[cols].apply(max_min_scaler)

    return data

def log1p_scaler(data,cols):
    log1p_scaler = lambda x: np.log1p(x+0.000001)
    data[cols] = data[cols].apply(log1p_scaler)

    return data


class Drive_control_Dataset(Dataset):
    def __init__(self, data_path, time_step=6, rl=False, s_path="./data/common_columns_647.csv"):
        #all_cols = set(pd.read_csv(s_path)["0"])

        self.df = pd.read_csv(data_path)

        all_cols = list(self.df.keys())
        self.df = self.df.loc[:, list(all_cols)]
       # print(self.df.mean())
       # print(self.df.std().to_numpy())
        #print((self.df.std()>0.00000001).to_numpy())


        self.a_t_cols = ['IN_Rail_pressure_feedback',
                         'IN_Gear_ratio',
                         'IN_Coolant_temperature',
                         'ITD_Main_timing_dmnd',
                         'FQD_Chkd_inj_fuel_dmnd',
                         'IN_Pedal_position',
                         'IN_Air_mass_flow',
                         'IN_Vehicle_speed',
                         'T_D_Actual_indicated_torque',
                         'AFC_Air_fuel_ratio',
                         'IN_Manifold_abs_pressure',
                         'IN_Egrh_position',
                         'IN_Vgth_position',
                         'IN_Ext_air_temp',
                         'IN_Throttle_position',
                         'IN_Swirl_position']  # 驾控变量
        self.a_t_cols = [i for i in self.a_t_cols if i in all_cols]
        self.v_t_cols = ['车速(km/h)']  # 或者 'IN_Engine_cycle_speed' 实际车速
        self.v_t_plus_cols = ['理论车速']

        # print(self.df.shape)

        self.o_t_cols = ['CO(ppm)','CO2(%)','THC(ppmC)','NOx(ppm)']

        #self.o_t_cols = ['CO2(%)']

        self.drop_cols = ['时间(s)', 'CO(ppm)', 'CO2(%)', 'THC(ppmC)', 'NOx(ppm)', 'CH4(ppm)', '理论车速',
                          '车速上限', '车速下限', '距离', '累积流量', '颗粒物样气温度', '颗粒物样气进口压力', '颗粒物样气差压',
                          '颗粒物背景气进口温度', '颗粒物背景气进口压力', '主稀释管路空气温度', 'DLS样气瞬时流量', 'DLS背景气瞬时流量',
                          'CPC Count', 'PND1测量值', 'PND2测量值', 'PN Total DF', 'CPC流量', 'SPCS样气压力',
                          'non-CH4', 'N2O', 'CO_MASS(mg)', 'CO2_MASS(mg)', 'THC_MASS(mg)',
                          'NOx_MASS(mg)', 'CH4_MASS(mg)', 'N2O_MASS(mg)', 'Time', 'IN_Engine_cycle_speed']
        self.s_t_cols = list(
            set(all_cols) - set(self.drop_cols + self.v_t_cols + self.v_t_plus_cols + self.o_t_cols + self.a_t_cols))
        self.s_t_cols = [i for i in self.s_t_cols if i in all_cols]

        self.p_cols = [i for i in range(217)]

        cols = self.a_t_cols + self.v_t_cols + self.v_t_plus_cols + self.o_t_cols + self.s_t_cols
        self.df = self.df.loc[:, list(cols)]
        

        drop_cols=[x for i,x in enumerate(self.df.columns) if self.df.iloc[:,i].std()==0]
        self.df=self.df.drop(drop_cols,axis=1) #利用drop方法将含有特定数值的列删除
        all_cols = list(self.df.keys())
        self.a_t_cols = [i for i in self.a_t_cols if i in all_cols]
        self.v_t_cols = ['车速(km/h)']  # 或者 'IN_Engine_cycle_speed' 实际车速
        self.v_t_plus_cols = ['理论车速']
        self.o_t_cols = ['CO(ppm)','CO2(%)','THC(ppmC)','NOx(ppm)']  
        self.s_t_cols = [i for i in self.s_t_cols if i in all_cols]
        cols = self.a_t_cols + self.v_t_cols + self.v_t_plus_cols + self.o_t_cols + self.s_t_cols
        self.df = self.df.loc[:, list(cols)]        
        
        #self.length = self.df.shape[0]
        self.time_step = time_step
        self.steps_every_ulp = 1800
        self.ulp_nums = int(self.df.shape[0] / self.steps_every_ulp)
        self._preprocess()
        self.rl = rl
        self._data_dim()
        self.parse_ulp()


    #数据归一化
    def normalize(self):
        columns = self.df.columns

        self.df = max_min_scaler(self.df, columns)
        self.df = log1p_scaler(self.df, columns)
        self.df = max_min_scaler(self.df, columns)
        #check none data  
        check_null = list(set(list(self.df.isnull().values.reshape(-1))))
        assert (len(check_null) == 1 and False in check_null)


    def parse_ulp(self,ulp_file = "./data/ulp.npy"):
        self.ulps = np.load(ulp_file)


    #数据预处理
    def _preprocess(self):
        self.normalize()
        columns = self.df.columns
        self.ulps_df = {}
        for ulp_index in range(self.ulp_nums):
            data = self.df.iloc[ulp_index*self.steps_every_ulp:(ulp_index+1)*self.steps_every_ulp].to_numpy()
            #
            f_data = np.tile(data[:1,:],(self.time_step-1,1))
            b_data = np.tile(data[-1:,:],(self.time_step-1,1))

            data = np.concatenate([f_data,data,b_data],0)
            self.ulps_df[ulp_index] = pd.DataFrame(data, columns=columns)
            #把下一秒车速赋值给理论车速
            self.ulps_df[ulp_index].loc[self.time_step:self.time_step+self.steps_every_ulp-1,self.v_t_plus_cols] \
                = self.ulps_df[ulp_index].loc[self.time_step+1:self.time_step+self.steps_every_ulp,self.v_t_cols].values
            #print(ulps_df[ulp_index].loc[100:102,['理论车速','车速(km/h)']])


    def _data_dim(self,):
        if self.rl:
            self.dims = {
                "p":len(self.p_cols),
                "s":[self.time_step,len(self.s_t_cols)],
                "v":len(self.v_t_cols),
                "a":len(self.a_t_cols),
                "o":len(self.o_t_cols),
            }
            #self.dims = [len(self.s_t_cols), len(self.v_t_cols), len(self.a_t_cols), len(self.v_t_plus_cols), len(self.s_t_cols)]
        else:

            self.dims = [[2*self.time_step-1,len(self.s_t_cols + self.a_t_cols + self.v_t_cols)], len(self.o_t_cols)]

    def __len__(self):
        if self.rl:
            return self.ulp_nums*(self.steps_every_ulp-self.time_step+1)
        else:
            return self.ulp_nums*self.steps_every_ulp

    #监督学习
    def __getitem__(self, idx):
        if self.rl:
            return self._rl__getitem__(idx)
        else:
            return self._sl__getitem__(idx)

    def _sl__getitem__(self,idx):
        x_cols = self.s_t_cols + self.a_t_cols + self.v_t_cols
        #if self.dims is None:
        #    self.dims = [[2*self.time_step-1,len(x_cols)], len(self.o_t_cols)]
        ulp_index = idx //(self.steps_every_ulp)
        idx = idx % ( self.steps_every_ulp)
        df = self.ulps_df[ulp_index]
        x = df.loc[idx:idx+2*self.time_step-2, x_cols].to_numpy().astype('float')
        y = df.loc[idx+self.time_step-1, self.o_t_cols].to_numpy().astype('float')

        return [torch.from_numpy(i).float() for i in [x,y]]

    def _rl__getitem__(self,idx):
        p_cols = self.p_cols
        s_cols = self.s_t_cols
        v_cols = self.v_t_cols
        a_cols = self.a_t_cols
        v_next_cols = self.v_t_plus_cols
        s_next_cols = self.s_t_cols
        o_cols = self.o_t_cols

        ulp_index = idx //(self.steps_every_ulp-self.time_step+1)
        idx = idx % ( self.steps_every_ulp-self.time_step+1)
        df = self.ulps_df[ulp_index]

        p = self.ulps[ulp_index].astype('float')
        s = df.loc[idx:idx+self.time_step-1, s_cols].to_numpy().astype('float')
        v = df.loc[idx+self.time_step-1, v_cols].to_numpy().astype('float')
        a = df.loc[idx+self.time_step-1, a_cols].to_numpy().astype('float')
        v_next = df.loc[idx+self.time_step - 1, v_next_cols].to_numpy().astype('float')
        s_next = df.loc[idx+self.time_step, s_next_cols].to_numpy().astype('float')
        o = df.loc[idx+self.time_step-1, o_cols].to_numpy().astype('float')

        return [torch.from_numpy(i).float() for i in [p,s,v,a,v_next,s_next,o]]
    




if __name__ == '__main__':

    train_dataset = Drive_control_Dataset(data_path='./data/train.csv',rl=True)
    #gen_datasets()
    #for (s_t, a_t, v_t, v_t_plus, o_t) in data_loader:
    #    print(s_t.shape, a_t.shape, v_t.shape, v_t_plus.shape, o_t.shape)
    for i in range(1598,3800):
        for j in train_dataset[i]:
            #j = train_dataset[j]
            #print(j.shape,j[0])
            print(j.shape)
        print("-----------------------")
        time.sleep(1000)