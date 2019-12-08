import numpy as np

arr= np.arange(15).reshape(3,5)
panjangarray= len(arr.ravel())

list_hasil=[]
for i in range (0,panjangarray,2):
    hasil=arr.ravel()[i]
    list_hasil.append(hasil)
print(list_hasil)
