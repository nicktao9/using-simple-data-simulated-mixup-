import matplotlib.pyplot as plt
import numpy as np
import fire
def myplot(train_dataset,result_dataset,mixup):
    dataset1 = train_dataset[:400]
    dataset2 = train_dataset[400:]
    plt.scatter(result_dataset[:,0] ,result_dataset[:,1],marker = '.',c = 'c')
    plt.scatter(dataset2[:,0] ,dataset2[:,1],marker = '.',c = 'g')
    plt.scatter(dataset1[:,0] ,dataset1[:,1],marker = None,c = 'darkorange')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['predict_c2','c1','c2'],loc='upper right')
    if mixup:
        plt.title('mixup')
        plt.savefig('./result_fig/final_result' + "_mixup.png")
    else:
        plt.title('ERM')
        plt.savefig('./result_fig/final_result' + "_ERM.png")
    plt.show()


def generate_fig(mixup):
    dataset_1 = np.load("./data/train_data.npz")["datasets"]
    if mixup:
        dataset_2 = np.load("./data/test_result_mixup.npz")["datasets"]  # result_data.npz 是从test.py中获得的
    else:
        dataset_2 = np.load("./data/test_result_ERM.npz")["datasets"]  # result_data.npz 是从test.py中获得的

    myplot(dataset_1,dataset_2,mixup)

if __name__ == '__main__':
    fire.Fire()