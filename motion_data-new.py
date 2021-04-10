import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    # I don't have to put all the data here
    #instead I use the data path
    # data_path is a list of files
    def __init__(self,data_path, seqlen, n_lookahead, dropout):
        self.data = data_path
        self.seqlen = seqlen
        self.n_lookahead = n_lookahead
        self.dropout  = dropout

    def data_transform(self, control_data, joint_data, seqlen, n_lookahead, dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.seqlen = seqlen
        self.dropout=dropout
        seqlen_control = seqlen + n_lookahead + 1
        
        #For LSTM network
        #n_frames = joint_data.shape[1]
        n_frames = joint_data.shape[0]
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # every we call concat_sequence the data will be repeated onece
        # this results in the memory expolosion

        # Joint positions for n previous frames

        #print("ok1\n")
        #autoreg = self.concat_sequence(self.seqlen, joint_data[:,:n_frames-n_lookahead-1,:])
        autoreg = self.concat_sequence(self.seqlen, joint_data[:n_frames - n_lookahead - 1, :])
        #print("ok2\n")

        # Control for n previous frames + current frame
        control = self.concat_sequence(seqlen_control, control_data)
        #print("ok3\n")
        # conditioning
        
        #print("autoreg:" + str(autoreg.shape))
        #print("control:" + str(control.shape))
        #new_cond = np.concatenate((autoreg,control),axis=2)
        new_cond = np.concatenate((autoreg, control), axis=1)

        # joint positions for the current frame
        x_start = seqlen
        #new_x = self.concat_sequence(1, joint_data[:,x_start:n_frames-n_lookahead,:])
        new_x = self.concat_sequence(1, joint_data[x_start:n_frames - n_lookahead, :])
        
        self.x = new_x
        self.cond = new_cond
        
        #TODO TEMP swap C and T axis to match existing implementation
        #self.x = np.swapaxes(self.x, 1, 2)
        self.x = np.swapaxes(self.x, 0, 1)
        #self.cond = np.swapaxes(self.cond, 1, 2)
        self.cond = np.swapaxes(self.cond, 0, 1)
        #print("self.x:" + str(self.x.shape))
        #print("self.cond:" + str(self.cond.shape))
        return self.x, self.cond


    def n_channels(self):
        #print("path_0",self.data[0])
        tmp_data = np.load(self.data[0])["arr_0"]
        #print(type(tmp_data))
        #print(tmp_data.files)
        control_data, joint_data = tmp_data[:, -3:], tmp_data[:, :-3]
        x, cond = self.data_transform(control_data, joint_data, self.seqlen, self.n_lookahead, self.dropout)
        return x.shape[0], cond.shape[0]

    def concat_sequence(self, seqlen, data):

        #nn, n_timesteps, n_feats = data.shape
        n_timesteps, n_feats = data.shape
        L = n_timesteps - (seqlen - 1)
        inds = np.zeros((L, seqlen)).astype(int)

        # create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        #print("rng ", rng)

        #print("p1\n")
        for ii in range(0, seqlen):
            #print("rng[ii:(n_timesteps-(seqlen-ii-1))]", rng[ii:(n_timesteps - (seqlen - ii - 1))].shape)
            # inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])
            inds[:, ii] = rng[ii:(n_timesteps - (seqlen - ii - 1))]
            #print("ids", inds[:, ii].shape)
        #print("p2\n")
        # slice each sample into L sequences and store as new samples
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +the code will result in memory explosion
        # cc=data[:,inds,:].copy()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # print ("cc: " + str(cc.shape))

        # reshape all timesteps and features into one dimention per sample
        # dd = cc.reshape((nn, L, seqlen*n_feats))
        #print("data", data[:, inds, :].shape)
        #print("data", data[inds, :].shape)
        #print("nn L seqlen*n_feats", nn, L, seqlen * n_feats)
        #print("L seqlen*n_feats", L, seqlen * n_feats)
        # dd = data[:,inds,:].reshape((nn, L, seqlen*n_feats))
        # dd = data[:, inds, :].reshape((nn, L, -1))
        #dd = np.reshape(data[:, inds, :], (nn, L, -1))
        dd = np.reshape(data[inds, :], (L, -1))
        #print("p3\n")
        # print ("dd: " + str(dd.shape))
        return dd


    def __len__(self):
        #return self.x.shape[0]
        return len(self.data)

    # we read the data here

    # dropout the original data randomly
    def __getitem__(self, idx):
    #When we read data in this not the init will save the memeory
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """

        #train_data[:, :, -3:], train_data[:, :, :-3]
        train_data = np.load(self.data[idx])["arr_0"]
        control_data, joint_data = train_data[:, -3:], train_data[:, :-3]
        self.x, self.cond = self.data_transform(control_data, joint_data, self.seqlen, self.n_lookahead, self.dropout)
        
        if self.dropout>0.:
            #n_feats, tt = self.x[idx,:,:].shape
            n_feats, tt = self.x.shape
            #cond_masked = self.cond[idx,:,:].copy()
            cond_masked = self.cond.copy()
            
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            #print(keep_pose)
            n_cond = cond_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            #print(mask)

            cond_masked = cond_masked*mask
            #sample = {'x': self.x[idx,:,:], 'cond': cond_masked}
            sample = {'x': self.x, 'cond': cond_masked}
        else:
            #sample = {'x': self.x[idx,:,:], 'cond': self.cond[idx,:,:]}
            sample = {'x': self.x, 'cond': self.cond}
            
        return sample


class TestDataset(Dataset):
    """Test dataset."""

    def __init__(self, data_path):
        """
        Args:
        control_data: The control input
        joint_data: body pose input
        Both with shape (samples, time-slices, features)
        """
        # Joint positions
        self.data = data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        test_data = np.load(self.data[idx])["arr_0"]
        self.autoreg, self.control = test_data[:, -3:], test_data[:, :-3]
        sample = {'autoreg': self.autoreg, 'control': self.control}
        return sample
