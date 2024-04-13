import numpy as np  # 导入所需包并将其命名为np


def ConsisTest(X):  # 函数接收一个如上述A似的矩阵
    # 计算权重
    # 方法一：算术平均法
    ## 第一步：将判断矩阵按照列归一化（每个元素除以其所在列的和）
    X = np.array(X)  # 将X转换为np.array对象
    sum_X = X.sum(axis=0)  # 计算X每列的和
    (n, n) = X.shape  # X为方阵，行和列相同，所以用一个n来接收
    sum_X = np.tile(sum_X, (n, 1))  # 将和向量重复n行组成新的矩阵
    stand_X = X / sum_X  # 标准化X（X中每个元素除以其所在列的和）

    ## 第二步：将归一化矩阵每一行求和
    sum_row = stand_X.sum(axis=1)

    ## 第三步：将相加后得到的向量中每个元素除以n即可得到权重向量
    print("算数平均法求权重的结果为：")
    print(sum_row / n)

    # 方法二：特征值法
    ## 第一步：找出矩阵X的最大特征值以及其对应的特征向量
    V, E = np.linalg.eig(X)  # V是特征值，E是特征值对应的特征向量
    max_value = np.max(V)  # 最大特征值
    print("最大特征值是：",max_value)
    max_v_index = np.argmax(V)  # 返回最大特征值所在位置
    max_eiv = E[:, max_v_index]  # 最大特征值对应的特征向量

    ## 第二步：对求出的特征向量进行归一化处理即可得到权重
    stand_eiv = max_eiv / max_eiv.sum()
    print("特征值法求权重的结果为：")
    print(stand_eiv)
    print("———————————————————————————————")
    # 一致性检验
    ## 第一步：计算一致性指标CI
    CI = (max_value - n) / (n - 1)
    ## 第二步：查找对应的平均随机一致性指标RI
    RI = np.array([0,0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54,1.56,1.58,1.59])
    ## 第三步：计算一致性比例CR
    print('CI=',CI,'RI=',RI[n])
    CR = CI / RI[n]
    if CR < 0.1:
        print("CR=", CR, "，小于0.1，通过一致性检验")
    else:
        print("CR=", CR, "，大于等于0.1，没有通过一致性检验，请修改判断矩阵")
    return None

A = [[1,1/6,3,1,4,4],[6,1,8,7,9,9],[1/3,1/8,1,1/2,2,1],[1,1/7,2,1,3,3],[1/4,1/9,1/2,1/3,1,1],[1/4,1/9,1,1/3,1,1]]


ConsisTest(A)
