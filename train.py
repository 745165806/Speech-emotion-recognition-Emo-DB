import os
import torch
import torch.nn as nn
from torch.utils import data

from dataset import  myDataset
from model import myNet

gpu_idx = [1]
my_path = '/data3/mahaoxin/emotion/data/wav/'
save_dir = '/data3/mahaoxin/emotion/exp/'
my_batch_size = 64
my_uttr_len = 300
my_fre_size = 200
epochs = 100
classnum=4

if __name__ == '__main__':
    # experiment = Experiment(api_key="YOUR API KEY",
    #                         project_name="YOUR_PROJECT_NAME", workspace="YOUR_WORKSPACE_NAME",
    #                         disabled=True)
    # experiment.set_name("mySERpro")

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % gpu_idx[0] if cuda else 'cpu')
    torch.cuda.set_device(device)

    # define dataset generators
    devset = myDataset(path=my_path+'valid',
                       batch_size=my_batch_size,
                       uttr_len=my_uttr_len,
                       fre_size=my_fre_size)
    devset_gen = data.DataLoader(devset,
                                 batch_size=my_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=0)
    # set save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'results/'):
        os.makedirs(save_dir + 'results/')
    if not os.path.exists(save_dir + 'models4sgd/'):
        os.makedirs(save_dir + 'models4sgd/')

    # define model
    model = myNet().to(device)
    if len(gpu_idx) > 1:
        model = nn.DataParallel(model, device_ids=gpu_idx)
    # set ojbective funtions
    criterion = nn.CrossEntropyLoss()
    # set optimizer
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
    # optimizer = torch.optim.Adam(params,
    #                              lr=0.0005,
    #                              weight_decay=0.0001,
    #                              amsgrad=True)
    ##########################################
    # train/val################################
    ##########################################
    # define dataset generators
    trnset = myDataset(path=my_path + 'train',
                       batch_size=my_batch_size,
                       uttr_len=my_uttr_len,
                       fre_size=my_fre_size)
    trnset_gen = data.DataLoader(trnset,
                                 batch_size=my_batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=4)
    best_acc = 0.0
    # train phase

    for epoch in range(epochs):
        print("-----start training: %d------" % epoch)
        model.train()
        running_loss = 0

        tcorrect = list(0. for i in range(classnum))
        ttotal = list(0. for i in range(classnum))

        for m_data, m_label in trnset_gen:
            if cuda:
                m_data, m_label = m_data.to(device), m_label.to(device)
            optimizer.zero_grad()
            output = model(m_data)
            loss = criterion(output, m_label)
            loss.backward()
            optimizer.step()

            running_loss += loss
            prediction = torch.argmax(output, 1)
            tres = prediction == m_label
            for label_idx in range(len(m_label)):
                label_single = m_label[label_idx]
                tcorrect[label_single] += tres[label_idx].item()
                ttotal[label_single] += 1
        acc_str = 'Accuracy: %f' % (sum(tcorrect) / sum(ttotal))
        for acc_idx in range(classnum):
            try:
                tacc = tcorrect[acc_idx] / ttotal[acc_idx]
            except:
                tacc = 0
            finally:
                acc_str += '\n classID:%d\tacc:%f\t' % (acc_idx, tacc)
        print(acc_str)
        print('[%d] loss: %.3f' % (epoch, running_loss))

        if(epoch % 5 ==0):
            print('------------VALID----------')
            correct = list(0. for i in range(classnum))
            total = list(0. for i in range(classnum))
            model.eval()
            with torch.set_grad_enabled(False):
                for m_batch, m_label in devset_gen:
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
            for acc_idx in range(classnum):
                try:
                    acc = correct[acc_idx] / total[acc_idx]
                except:
                    acc = 0
                finally:
                    acc_str += '\n classID:%d\tacc:%f\t' % (acc_idx, acc)
            print(acc_str)
            if(eval_acc > best_acc):
                best_acc = eval_acc
                torch.save(model.state_dict(), save_dir + 'models4sgd/best%d-%.5f.pt'%(epoch, best_acc))

        print('Finished Training')