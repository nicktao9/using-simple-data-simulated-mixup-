import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torchData
# from sklearn import metrics
from classify import Net
import fire
batch_size = 1024
num_epoch = 1000
learn_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------load_data-------------------------------------------------------
train_data = np.load("./data/train_data.npz")["datasets"]
tr_x = train_data[:,[0,1]]
tr_y = train_data[:,2]
tr_x = np.array(tr_x,dtype = np.float32)
tr_y = np.array(tr_y,dtype = np.int64)
tr_x = torch.tensor(tr_x)
tr_y = torch.tensor(tr_y)
tr_datasets = torchData.TensorDataset(tr_x,tr_y)
tr_loader = torchData.DataLoader(tr_datasets,batch_size = batch_size,shuffle = True)

test_data = np.load("./data/test_data.npz")["datasets"]
te_x = torch.tensor(test_data,dtype= torch.float32)
te_datasets = torchData.TensorDataset(te_x)
te_loader = torchData.DataLoader(te_datasets,batch_size = 1024,shuffle = True)

model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

#-----------------------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#---------------------------------------------------------------------------------
def train(epoch:int,mixup:bool):
    model.train()
    total_step = len(tr_datasets)
    data_pre = []
    data_true = [] 
    total = 0
    correct = 0
    for i, (x,y) in enumerate(tr_loader):
        inputs = x.to(device)
        label = y.to(device)
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs,label,1.,False)
        outputs = model(inputs)
        if mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        if mixup:
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            correct += predicted.eq(label.data).cpu().sum().float()
        if (i + 1) % 1 == 0:
            print('After [{}/{}] epoch | [{}/{}] batch,loss is {:0.4f}'.format(epoch + 1,num_epoch,i+1,total_step//batch_size,loss.item()))
    acc = (correct/total).item()
    return acc
def test(mixup:bool):
    model.eval()
    point_of_predict = []
    with torch.no_grad():
        for x in te_loader:
            data = x[0].to(device)
            outputs = model(data)
            outputs = nn.Softmax(dim = 1)(outputs)
            for i,j in enumerate(outputs):
                if j[0]>j[1]:  # 第0类
                    x = data[i].numpy()
                    point_of_predict.append(x)
    point_of_predict = np.array(point_of_predict)
    if mixup:
        np.savez("./data/test_result_mixup.npz",datasets = point_of_predict)  # 把测试结果中的第0类对应的原始数据保存起来
    else:
        np.savez("./data/test_result_ERM.npz",datasets = point_of_predict)  # 把测试结果中的第0类对应的原始数据保存起来
    print("Saved the test_result")

def main(mixup):
    ac_ = 0
    for i in range(num_epoch):
        acc = train(i,mixup)
        if acc > ac_:
            ac_ = acc
            torch.save(model.state_dict(),"./model/checkpoint.pth")   #保存最优模型
    print("acc is {}".format(ac_))
    print("---------start test-------------")
    model.load_state_dict(torch.load("./model/checkpoint.pth"))
    test(mixup)      
if __name__ == '__main__':
    fire.Fire()
