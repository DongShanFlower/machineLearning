import pandas as pd

from io import StringIO

from sklearn import linear_model

import matplotlib.pyplot as plt
import numpy as np 


# ���������۸���ʷ����(csv�ļ�)
csv_data = 'square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n'
print(csv_data)

# ����dataframe
df = pd.read_csv(StringIO(csv_data))
print(df)


# �������Իع�ģ��
regr = linear_model.LinearRegression()
c = df['square_feet'].values.shape
print(c[0])
a = df['square_feet'].values
print(df['square_feet'].values)
b = a.reshape(-1,1)
# ���
regr.fit(df['square_feet'].values.reshape(-1, 1), df['price']) # ע��˴�.reshape(-1, 1)����ΪX��һά�ģ�

# ���ѵõ�ֱ�ߵ�б�ʡ��ؾ�
a, b = regr.coef_, regr.intercept_

# ������Ԥ�����
area = 238.5

# ��ʽ1������ֱ�߷��̼���ļ۸�
print(a * area + b)

# ��ʽ2������predict����Ԥ��ļ۸� 
list_a = np.array([area])
print(regr.predict(list_a.reshape(1,-1)))

# ��ͼ
# 1.��ʵ�ĵ�
plt.scatter(df['square_feet'], df['price'], color='blue')

# 2.��ϵ�ֱ��
plt.plot(df['square_feet'], regr.predict(df['square_feet'].values.reshape(-1,1)), color='red', linewidth=4)

plt.show()

