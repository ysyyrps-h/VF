import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
from matplotlib.pyplot import MultipleLocator

import matplotlib.animation as animation
import imageio.v2 as imageio
import os

filename = os.path.join('dongtu.txt')  ##########两段轨迹怎么解决
with open(filename, 'r') as f:
    lines = f.readlines()
x = []
y = []
for line in lines:
    data = line.strip()[0:-1].split()
    i1=data[0]
    i2 = data[1]
    x.append(float(i1))
    y.append(float(i2))
xs = []
ys = []
image_list=[]
# plt.xlim((min(x))-10,max(x)+50)
# plt.ylim(min(y)-10,max(y)+75)

##########画图的画板处自动更新
# plt.xlim(0,300)
# plt.ylim(-100,300)
# plt.figure(figsize=(8, 6))

for i in range(len(x)):
    xs.append(x[i])
    ys.append(y[i])
    plt.axis([0, 300, -100, 300])
    plt.plot(xs, ys,color='blue',linestyle='-',marker='.',
        markeredgecolor='b',markersize='10')
    # plt.plot(x2, y2,color='blue',linestyle='-',marker='.',
    #      markeredgecolor='b',markersize='10')
    plt.savefig('temp.jpg')
    image_list.append(imageio.imread('temp.jpg'))
    plt.pause(1)
imageio.mimsave('pic.gif',image_list,duration=1130)






# def main():
#     date='0829'
#     label='0'
#     first='1'
#     camera='2'
#     graph(date,label,first,camera)
# if  __name__=="__main__":
#     main()

# with open('truth0706_1.txt', 'r', encoding='utf-8') as f:
#     data1=f.read()
#     data1=data1.split('\n')
#     data1=[list(filter(None, data1))]
#     # print(data)
#     data1=[a.split(',')for a in data1[0]]
#     # print(data)
#     new_data1=[]
#     for b in data1:
#         new_data1.append([float(a)for a in b])
#     # new_data1 = list(filter(None, new_data1))

# print(new_data)

# print(len(x1))

# for b in new_data1:
#     x2.append(float(b[0]))
#     y2.append(float(b[1])) 
# print(len(x2))


# plt.plot(x1, y1, color='red',linestyle='-',marker='.',
#          markeredgecolor='r',markersize='10',label='Estimation(cm)')
# plt.plot(x2, y2,color='blue',linestyle='-',marker='.',
#          markeredgecolor='b',markersize='10',label='True(cm)')
# plt.legend()



# ax.spines["bottom"].set_axisline_style("->", size = 1.5)
# ax.spines["left"].set_axisline_style("->", size = 1.5)

# ani = animation.ArtistAnimation(fig=fig, artists=artists, repeat=False, interval=10)
# # plt.show()
# ani.save('2.gif',writer='pillow', fps=30)
    # plt.show()
    # plt.figure()
# plt.ioff()
# plt.show()

# #计算误差
# error=[]
# # error_sum=[]
# # # print(len(new_x1))
# # # print(len(x1))
# for i in range(len(x1)):
#     error.append(np.sqrt((x1[i]-x2[i])**2+(y1[i]-y2[i])**2))

# print('max:',np.max(error),'min:',np.min(error))
# print('avarage:',sum(error)/len(x1))
# print(error)
