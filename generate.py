import numpy as np
import matplotlib.pyplot as plt
import os,fire
def generate_train_data(display:bool = True):
    t = np.arange(0,20,0.05)
    num = len(t)
    noise_x = np.random.uniform(-0.1,0.1,size = num)
    noise_y = np.random.uniform(-0.1,0.1,size = num)
    x1_data = 0.5 * np.cos(t) + noise_x
    y1_data = 0.5 * np.sin(t) + noise_y
    x2_data = np.cos(t) + 0.5* noise_x
    y2_data = np.sin(t) + 0.5* noise_y
    dataset1,dataset2 = [],[]
    for i in range(len(x1_data)):
        dataset1.append([x1_data[i],y1_data[i],0])
        dataset2.append([x2_data[i],y2_data[i],1])
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)
    dataset = np.vstack((dataset1,dataset2))
    if display:
        plt.title("train_data")
        plt.scatter(dataset1[:,0] ,dataset1[:,1],marker = '.',c = 'darkorange')
        plt.scatter(dataset2[:,0] ,dataset2[:,1],marker = '.',c = 'g')
        plt.savefig("./result_fig/train_data.png")
        plt.show()
    np.savez("./data/train_data.npz",datasets = dataset)
    print("Saved the data to file_[train_data.npz],the train_dataset.shape is {0}".format(dataset.shape))
def generate_test_data(display:bool = True):
    # 生成一个比训练数据更大一些的点,然后用这些点来测试整个网络
    t = np.random.random(size=138000) * 2 * np.pi - np.pi
    num = len(t)
    x = 1.5 * np.cos(t) 
    y = 1.5 * np.sin(t) 
    i_set = np.arange(0,138000,1)
    datasets = []
    for i in i_set:
        len1 = np.sqrt(np.random.random())
        x[i] = x[i] * len1
        y[i] = y[i] * len1
        datasets.append([x[i],y[i]])
    datasets = np.array(datasets)
    if display:
        plt.title('test_data')
        plt.scatter(datasets[:,0],datasets[:,1],marker = '.',c = 'c')
        plt.savefig("./result_fig/test_data.png")
        plt.show()
    np.savez("./data/test_data",datasets = datasets)
    print("Saved the data to file_[test_data.npz],the test_dataset.shape is {0}".format(datasets.shape))
if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('result_fig'):
        os.makedirs('result_fig')
    fire.Fire()