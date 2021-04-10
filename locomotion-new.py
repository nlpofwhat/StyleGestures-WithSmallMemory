import os
import numpy as np
from .motion_data import MotionDataset, TestDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation
import gc   # save the memory

def mirror_data(data):
    aa = data.copy()
    aa[:,:,3:15]=data[:,:,15:27]
    aa[:,:,3:15:3]=-data[:,:,15:27:3]
    aa[:,:,15:27]=data[:,:,3:15]
    aa[:,:,15:27:3]=-data[:,:,3:15:3]
    aa[:,:,39:51]=data[:,:,51:63]
    aa[:,:,39:51:3]=-data[:,:,51:63:3]
    aa[:,:,51:63]=data[:,:,39:51]
    aa[:,:,51:63:3]=-data[:,:,39:51:3]
    aa[:,:,63]=-data[:,:,63]
    aa[:,:,65]=-data[:,:,65]
    return aa

def reverse_time(data):
    aa = data[:,-1::-1,:].copy()
    aa[:,:,63] = -aa[:,:,63]
    aa[:,:,64] = -aa[:,:,64]
    aa[:,:,65] = -aa[:,:,65]
    return aa

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def fit_and_standardize(data):
    shape = data.shape
    #flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    flat = data.reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    #flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    flat = data.reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
  
def create_synth_test_data(n_frames, nFeats, scaler):

    syth_data = np.zeros((7,n_frames,nFeats))
    lo_vel = 1.0
    hi_vel = 2.5
    lo_r_vel = 0.08
    hi_r_vel = 0.08
    syth_data[0,:,63:65] = 0
    syth_data[1,:,63] = lo_vel
    syth_data[2,:,64] = lo_vel
    syth_data[3,:,64] = hi_vel
    syth_data[4,:,64] = -lo_vel
    syth_data[5,:,64] = lo_vel
    syth_data[5,:,65] = lo_r_vel
    syth_data[6,:,64] = hi_vel
    syth_data[6,:,65] = hi_r_vel
    syth_data = standardize(syth_data, scaler)
    syth_data[:,:,:63] = np.zeros((syth_data.shape[0],syth_data.shape[1],63))
    return syth_data.astype(np.float32)

class Locomotion():
    
    #first time run this
    def __init__(self, hparams):
    
        data_root = hparams.Dir.data_root

        #load data
        train_series = np.load(os.path.join(data_root, 'all_locomotion_train_'+str(hparams.Data.framerate)+'fps.npz'))
        train_data = train_series['clips'].astype(np.float32)
        test_series = np.load(os.path.join(data_root, 'all_locomotion_test_'+str(hparams.Data.framerate)+'fps.npz'))
        test_data = test_series['clips'].astype(np.float32)
        
        print("input_data: " + str(train_data.shape))
        print("test_data: " + str(test_data.shape))

        # Split into train and val sets
        # validation_data is the last 100 samples
        validation_data = train_data[-100:,:,:]
        train_data = train_data[:-100,:,:]
        #print("num 1\n")
        # Data augmentation
        if hparams.Data.mirror:
            mirrored = mirror_data(train_data)
            train_data = np.concatenate((train_data,mirrored),axis=0)
            del mirrored
        #print("num 2\n")
        if hparams.Data.reverse_time:
            rev = reverse_time(train_data)
            train_data = np.concatenate((train_data,rev),axis=0)
            del rev
        #print("num 3\n")
        # Standardize
        train_data, scaler = fit_and_standardize(train_data)
        print("train_data shape ",train_data.shape)
        validation_data = standardize(validation_data, scaler)
        print("alidation_data shape ",validation_data.shape)
        all_test_data = standardize(test_data, scaler)
        print("all_test_data shape ",all_test_data.shape)
        synth_data2 = create_synth_test_data(test_data.shape[1], test_data.shape[2], scaler)
        all_test_data = np.concatenate((all_test_data, synth_data2), axis=0)
        del synth_data2

        self.n_test = all_test_data.shape[0]
        print("self.n_test %d \n",self.n_test)
        n_tiles = 1+hparams.Train.batch_size//self.n_test
        #all_test_data = np.tile(all_test_data.copy(), (n_tiles,1,1))
        all_test_data = np.tile(all_test_data, (n_tiles, 1, 1))
        self.scaler = scaler
        self.frame_rate = hparams.Data.framerate
               
        # Create pytorch data sets
        #
        #print("end1")
        #the last 3 data is control data
        train_data_path = []
        new_train_lomotion_data_path  = os.path.join(data_root, "train")
        if not os.path.exists(new_train_lomotion_data_path):
            os.makedirs(new_train_lomotion_data_path)
        for i in range(train_data.shape[0]):
            dir_i = os.path.join(new_train_lomotion_data_path, "{0}".format(i) +".npz")
            np.savez(dir_i,train_data[i,:,:])
            train_data_path.append(dir_i)
        del train_data

        val_data_path = []
        new_val_lomotion_data_path = os.path.join(data_root, "val")
        if not os.path.exists(new_val_lomotion_data_path):
            os.makedirs(new_val_lomotion_data_path)
        for i in range(validation_data.shape[0]):
            dir_i = os.path.join(new_val_lomotion_data_path, "{0}".format(i) + ".npz")
            np.savez(dir_i, validation_data[i, :, :])
            val_data_path.append(dir_i)
        del validation_data

        test_data_path = []
        new_test_lomotion_data_path = os.path.join(data_root, "test")
        if not os.path.exists(new_test_lomotion_data_path):
            os.makedirs(new_test_lomotion_data_path)
        for i in range(all_test_data.shape[0]):
            dir_i = os.path.join(new_test_lomotion_data_path, "{0}".format(i) + ".npz")
            np.savez(dir_i, all_test_data[i, :, :])
            test_data_path.append(dir_i)
        del all_test_data

        #self.train_dataset = MotionDataset(train_data[:,:,-3:], train_data[:,:,:-3], hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)
        self.train_dataset = MotionDataset(train_data_path, hparams.Data.seqlen,hparams.Data.n_lookahead, hparams.Data.dropout)
        #print("end2")
        self.test_dataset = TestDataset(test_data_path)
        #print("end3")

        self.validation_dataset = MotionDataset(val_data_path, hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)
        #print("end4")
        self.seqlen = hparams.Data.seqlen
        gc.collect()
        #save memory

    def save_animation(self, control_data, motion_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)

    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		