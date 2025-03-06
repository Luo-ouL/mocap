import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# 查找值为 30 的元素的索引
index = np.where(arr == 30)
print(index[0][0])  # 输出: (array([2]),)
