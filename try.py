import numpy as np

txtfile = './[20220815]1Cloud.txt'
data = np.loadtxt(txtfile)  # 用numpy  torch 会出问题
print(data.shape)

data = data[:, :4]
print(data.shape)

np.savetxt( './xxxcxcx.txt', data, fmt='%.4f')

