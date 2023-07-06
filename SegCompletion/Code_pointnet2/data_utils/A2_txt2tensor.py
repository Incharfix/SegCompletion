import numpy as np
import torch

# 1.load txt  -->  numpy  -->  tensor

def  loadtxt_my( txtfile):
    data = np.loadtxt(txtfile)
    data_tensor = torch.tensor(data).float()

    return data_tensor

if __name__ == '__main__':
    txtfile = './10000.txt'
    dsaafd = loadtxt_my(txtfile)

    print(type(dsaafd))
    print(dsaafd.shape)



