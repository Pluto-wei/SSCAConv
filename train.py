import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from SSCANet import SSCANet
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchstat import stat
from torchsummary import summary



# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0003 # 0.001
initepoch=1000
epochs = 1500
ckpt = 50
batch_size = 32
device=torch.device('cuda:1')

model = SSCANet(32,5,True).to(device) #LAC+CC
# summary(model, input_size = [(1, 64, 64), (8, 64, 64)], batch_size=1)

model.load_state_dict(torch.load('weights/DK4/1cNet_1000.pth'))
criterion = nn.MSELoss().to(device)
optimizer1 = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1

#if os.path.exists('train_logs_ca'):  # for tensorboard: copy dir of train_logs
#   shutil.rmtree('train_logs_ca')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
writer = SummaryWriter('train_logs_ca')


def save_checkpoint(model,f, epoch):  # save model function
    #if not os.path.exists(model_out_path):
    #    os.makedirs(model_out_path)
    torch.save(model.state_dict(), 'weights/DK4/'+f+'cNet_{}.pth'.format(epoch))

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '>'] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader):
    print('Start training...')

    for epoch in range(initepoch, epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, pan = Variable(batch[0], requires_grad=False).to(device), \
                                     Variable(batch[1]).to(device), \
                                     Variable(batch[2]).to(device)

            optimizer1.zero_grad()  # fixed
            out = model(pan, lms)
            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
            loss.backward()  # fixed
            optimizer1.step()  # fixed

 #       lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, '1', epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms ,pan= Variable(batch[0], requires_grad=False).to(device), \
                                         Variable(batch[1]).to(device), \
                                         Variable(batch[2]).to(device)

                out = model(pan,lms)
                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())
                
        v_loss = np.nanmean(np.array(epoch_val_loss))
        writer.add_scalars('loss:',{'tra_loss':t_loss,'val_loss':v_loss},epoch)
        print('validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    train_set = Dataset_Pro('../data/pansharpening/training_wv3/train_wv3.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('../data/pansharpening/training_wv3/valid_wv3.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    ###################################################################
    #train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)