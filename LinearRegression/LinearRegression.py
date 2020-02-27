import xlrd
import matplotlib.pyplot as plt
import numpy as np

#open file
data = xlrd.open_workbook('pneumonia.xlsx')

# get the table

table = data.sheet_by_index(0)

# get x and y
X = table.col_values(0)[1:table.nrows]
Y = table.col_values(1)[1:table.nrows]
X = np.array(X)
Y = np.array(Y)

# preprocessing
# X = X/abs(X.max())
# Y = Y/abs(Y.max())

#build model
Xsum = 0
X2sum = 0
Ysum = 0
XY = 0

n = table.nrows - 1

for i in range(n):
    Xsum += X[i]
    Ysum += Y[i]
    X2sum += X[i]**2
    XY = X[i] * Y[i]

k = (XY - Xsum * Ysum / n) / (X2sum - Xsum**2/n)
b = (Ysum - k * Xsum) / n


days = range(0,20)

Y_pred = k*days + b

#plot
plt.plot(X,Y,'go', label='Original Data')
plt.plot(days,Y_pred,'r-', label='Fitted Line')
plt.title('pneumonia prediction')
plt.xlabel('Date')
plt.ylabel('Increment')
plt.legend()
plt.show()
