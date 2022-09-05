x = 626
padding1 = 8
padding2 = 8
stride1 = 16
stride2 = 8
o1 = 16
o2 = 0
x1 = (x-1)*stride1 - 2*padding1 + (32-1) +1 + o1
x2 = (x1-1)*stride2 - 2*padding2 + (32-1) +1 + o2
print(x2)
array = [9,3,3,3,4,5,5,6,7]
inds = [0,3,4,2,5]
print(array[inds])