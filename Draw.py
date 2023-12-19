#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xlrd
#定义图像和三维格式坐标轴
from scipy.interpolate import griddata

fig=plt.figure()
ax2 = Axes3D(fig)
fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
wb = xlrd.open_workbook('覆盖率.xls')
#按工作簿定位工作表
sh = wb.sheet_by_name('Sheet1')
print(sh.nrows)#有效数据行数
print(sh.ncols)#有效数据列数
print(sh.cell(0,0).value)#输出第一行第一列的值
x=sh.col_values(0)#输出第一行的所有值
y=sh.col_values(1)#输出第一行的所有值
z=sh.col_values(15)#输出第一行的所有值

X, Y= np.meshgrid(x, y)
Z = griddata((x,y), z, (X,Y), method='linear')
# Z = 0*np.sin(X)+0*np.cos(Y)+z
# X,Y,Z= griddata(x,y,z,X,Y);
#将数据和标题组合成字典
# print(dict(zip(sh.row_values(0),sh.row_values(1))))
#遍历excel，打印所有数据

# xx = np.arange(-5,5,0.5)
# yy = np.arange(-5,5,0.5)
# X, Y = np.meshgrid(xx, yy)
# Z = np.sin(X)+np.cos(Y)z


#作图
#ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()
