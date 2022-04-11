from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
# 读取数据
def read_xls():
    # df = pd.DataFrame(pd.read_excel(r'../data/有功无功.xls'))
    df = pd.DataFrame(pd.read_excel(r'../data/用户电费累积电量.xls'))

    # for x in range(len(df['实际有功值'])):
    for x in range(len(df['累积电量'])):
        # tmp_data = [df['实际有功值'][x], df['实际无功值'][x]]
        tmp_data = [df['累积电量'][x], df['用户电费'][x]]
        if (x == 0):
            data = array([tmp_data])
        else:
            data = append(data, [tmp_data], axis=0)

    return data


# 计算距离
def cout_dist(vec1, vec2):
    return sqrt(sum(power(vec1 - vec2, 2)))


# 创建初始质心，需要根据main函数中的k对应更改
def creat_centers(data, k):
    centroids = zeros((k, n))
    # centroids[0, 0] = 12.5
    # centroids[0, 1] = 2.6
    # centroids[1, 0] = 18.2
    # centroids[1, 1] = 2.3
    # centroids[2, 0] = 17.3
    # centroids[2, 1] = 16.9
    # centroids[3, 0] = 20
    # centroids[3, 1] = 1.5
    centroids[0, 0] = 7.8
    centroids[0, 1] = 23.3
    centroids[1, 0] = 19.6
    centroids[1, 1] = 36.5
    centroids[2, 0] = 11.6
    centroids[2, 1] = 46
    centroids[3, 0] = 28.7
    centroids[3, 1] = 46.6
    print(centroids)
    return centroids


# kmeans聚类
def Kmeans(data, k, times, dist=cout_dist, creat_center=creat_centers):
    m = shape(data)[0]
    init = zeros((m, 2))
    cluster_assment = mat(init)
    # 初始化聚类中心矩阵
    centroids = creat_centers(data, k)
    for epoch in range(times):
        # 对数据集合中每个样本点进行计算
        for i in range(m):
            min_dist = inf
            min_index = -1
            # 对每个样本点到每个中心的距离进行计算
            for j in range(k):
                dist_ij = cout_dist(centroids[j, :], data[i, :])
                # 找到距离最近的中心的距离和索引
                if dist_ij < min_dist:
                    min_dist = dist_ij
                    min_index = j
                    cluster_assment[i, :] = min_index, min_dist
        # 对所有节点聚类之后,重新更新中心
        for i in range(k):
            # A把矩阵转为数组
            pts_in_cluster = data[nonzero(cluster_assment[:, 0].A == i)[0]]
            if len(pts_in_cluster != 0):
                centroids[i, :] = mean(pts_in_cluster, axis=0)
    return centroids, cluster_assment


if __name__ == '__main__':
    # 创建数据集
    # data = array([[2, 10], [2, 5], [8, 4], [5, 8],
    #               [7, 5], [6, 4], [1, 2], [4, 9]])
    # 调用读取数据函数
    data = read_xls()
    # print(data)
    k = 4  # 为聚类个数
    n = 2  # n为特征个数, 数据所拥有的特征
    times = 0  # 迭代次数
    while (True):
        times = times + 1
        centroids, cluster_assment = Kmeans(data, k, times, cout_dist, creat_centers)

        if (times > 1):
            if (centroids[:, 0].tolist() == last_center0 and centroids[:, 1].tolist() == last_center1):
                print('迭代 ', times - 1, ' 次后收敛')
                print(centroids)
                break

        last_center0 = centroids[:, 0].tolist()
        last_center1 = centroids[:, 1].tolist()
        # last_center2 = centroids[:, 1].tolist()

        # print(cluster_assment)
        predict_label = cluster_assment[:, 0]
        # print(predict_label)
        data_and_pred = column_stack((data, predict_label))
        print(data_and_pred)

        df = pd.DataFrame(data_and_pred, columns=['data1', 'data2', 'pred'])
        df0 = df[df.pred == 0].values
        df1 = df[df.pred == 1].values
        df2 = df[df.pred == 2].values
        df3 = df[df.pred == 3].values  # 离散的点

        plt.scatter(df0[:, 0], df0[:, 1], c='turquoise', marker='o', label='normal')
        plt.scatter(df1[:, 0], df1[:, 1], c='orange', marker='*', label='abnormal')
        # plt.scatter(df1[:, 0], df1[:, 1], c='orange', marker='*', label='normal')
        plt.scatter(df2[:, 0], df2[:, 1], c='purple', marker='+', label='normal')
        # plt.scatter(df2[:, 0], df2[:, 1], c='purple', marker='+', label='abnormal')
        plt.scatter(df3[:, 0], df3[:, 1], c='blue', marker='^', label='normal')
        plt.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), c='red')
        plt.xlabel('用户电量', fontproperties=font_set)
        plt.ylabel('累计电费', fontproperties=font_set)
        # plt.xlabel('有功', fontproperties=font_set)
        # plt.ylabel('无功', fontproperties=font_set)


        plt.legend(loc=1)

        # save_path = '../res/' + str(times) + '.png'
        save_path = '../res2/' + str(times) + '.png'
        print(save_path)
        plt.savefig(save_path)

        plt.show()
