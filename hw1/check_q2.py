import numpy as np
import matplotlib.pyplot as plt


# load the kernel from 
kernel=np.load('hw1/data/kernel.npy')

input=np.load('hw1/data/samples_0.npy')

# plot kernels on the same figure there are eight kernels in total
plt.figure()
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(kernel[i,0,:,:],cmap='gray')
    plt.title('kernel'+str(i+1))    

plt.show()

# plot input

plt.figure()
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(input[i,0,:,:],cmap='gray')
    plt.title('input'+str(i+1))    

plt.show()