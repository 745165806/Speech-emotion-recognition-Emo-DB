import os
import torch
import torch.nn as nn
from torch.utils import data

from dataset import  myDataset
from model import myNet

gpu_idx = [1]
my_path = '/data3/mahaoxin/emotion/data/wav/'
# save_dir = '/data3/mahaoxin/emotion/exp/models7sgd/best20-0.32692.pt'
save_dir = '/data3/mahaoxin/emotion/exp/models4sgd/best60-0.72603.pt'
my_batch_size = 64
my_uttr_len = 300
my_fre_size = 200

if __name__ == '__main__':
    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % gpu_idx[0] if cuda else 'cpu')
    torch.cuda.set_device(device)

    evalset = myDataset(path=my_path+'test',
                       batch_size=my_batch_size,
                       uttr_len=my_uttr_len,
                       fre_size=my_fre_size)
    evalset_gen = data.DataLoader(evalset,
                                  batch_size=my_batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)

    # load model
    model = myNet().to(device)
    model.load_state_dict(torch.load(save_dir))
    if len(gpu_idx) > 1:
        model = nn.DataParallel(model, device_ids=gpu_idx)

    model.eval()
    correct = list(0. for i in range(7))
    total = list(0. for i in range(7))
    with torch.set_grad_enabled(False):
        for m_batch, m_label in evalset_gen:
            m_batch, m_label = m_batch.to(device), m_label.to(device)
            output = model(m_batch)

            prediction = torch.argmax(output, 1)

            res = prediction == m_label
            for label_idx in range(len(m_label)):
                label_single = m_label[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1
    acc_str = 'Accuracy: %f' % (sum(correct) / sum(total))
    eval_acc = sum(correct) / float(sum(total))
    for acc_idx in range(7):
        try:
            acc = correct[acc_idx] / total[acc_idx]
        except:
            acc = 0
        finally:
            acc_str += '\n classID:%d\tacc:%f\t' % (acc_idx, acc)
    print("-------Test-------")
    print(acc_str)

    print('Finished Testing')