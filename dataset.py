import os
import torch
from torch.utils import data
import torchaudio as ta
import numpy as np


class myDataset(data.Dataset):
    def __init__(self, path, batch_size, uttr_len, fre_size):
        # self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
        self.classes = {0: 'W', 1: 'L', 2: 'A', 3: 'T'}
        self.batch_size = batch_size
        self.uttr_len = uttr_len
        self.fre_size = fre_size
        self.get_berlin_dataset(path)

    def get_berlin_dataset(self, path):
        # males = ['03', '10', '11', '12']
        # females = ['08', '09', '13', '14']
        # train_dic = ['03','08','09','10','11','13']
        # valid_dic = ['12','14']
        # test_dic = ['15', '16']
        try:
            classes = {v: k for k, v in self.classes.iteritems()}
        except AttributeError:
            classes = {v: k for k, v in self.classes.items()}
        self.label = []
        self.data = []
        for audio in os.listdir(path):
            if(audio[5] not in classes.keys()):
                continue
            audio_path = os.path.join(path, audio)
            x, sr = ta.load_wav(audio_path)
            x = ta.transforms.Spectrogram(
                win_length=int(20 * sr * 0.001),
                hop_length=int(10 * sr * 0.001),
                n_fft=800)(x)
            self.data.append(x)
            self.label.append(classes[audio[5]])

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        uttr_len = x.shape[2]
        if uttr_len > self.uttr_len:
            start_idx = np.random.randint(low=0,
                                          high=uttr_len - self.uttr_len)
            x = x[:, :self.fre_size, start_idx:start_idx + self.uttr_len]
        elif uttr_len < self.uttr_len:
            nb_dup = int(self.uttr_len / uttr_len) + 1
            x = np.tile(x, (1, 1, nb_dup))[:, :self.fre_size, :self.uttr_len]
            x = torch.from_numpy(x)
        else:
            x = x[:, :self.fre_size, :]
        return x, y

if __name__ == '__main__':
    my_path = '/data3/mahaoxin/emotion/data/wav/'
    my_batch_size = 50
    my_uttr_len = 300
    my_fre_size = 200

    trnset = myDataset(path=my_path + 'train',
                       batch_size=my_batch_size,
                       uttr_len=my_uttr_len,
                       fre_size=my_fre_size)
    trnset_gen = data.DataLoader(trnset,
                                 batch_size=my_batch_size,
                                 shuffle=True,
                                 drop_last=True)
    i=0
    for m_data, m_label in trnset_gen:
        print(i)
        print(m_data.shape)
        print(m_data)
        print(m_label)
        i=i+1