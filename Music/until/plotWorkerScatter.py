import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS

colors_9 = np.array([0.7614285714285715, 0.74, 0.10714285714285714, 0.77, 0.77, 0.73, 0.7557142857142857, 0.77, 0.7371428571428571, 0.77, 0.7742857142857142, 0.7271428571428571, 0.7285714285714285, 0.7414285714285714, 0.6114285714285714, 0.6771428571428572, 0.6342857142857142, 0.7085714285714285, 0.6457142857142857, 0.6914285714285714, 0.5385714285714286, 0.6542857142857142, 0.7042857142857143, 0.6828571428571428, 0.6742857142857143, 0.7385714285714285, 0.6142857142857143, 0.6071428571428571, 0.7342857142857143, 0.7685714285714286, 0.5914285714285714, 0.6271428571428571, 0.6085714285714285, 0.58, 0.3914285714285714, 0.6142857142857143, 0.7257142857142858, 0.7528571428571429, 0.6471428571428571, 0.5242857142857142, 0.6257142857142857, 0.66, 0.22857142857142856, 0.11714285714285715])  # 每个点的颜色值，范围为0到1

def workerScatter(data):
    colors = np.array([0.7614285714285715, 0.74, 0.10714285714285714, 0.77, 0.77, 0.73,
                       0.7557142857142857, 0.77, 0.7371428571428571, 0.77, 0.7742857142857142,
                       0.7271428571428571, 0.7285714285714285, 0.7414285714285714, 0.6114285714285714,
                       0.6771428571428572, 0.6342857142857142, 0.7085714285714285, 0.6457142857142857,
                       0.6914285714285714, 0.5385714285714286, 0.6542857142857142, 0.7042857142857143,
                       0.6828571428571428, 0.6742857142857143, 0.7385714285714285, 0.6142857142857143,
                       0.6071428571428571, 0.7342857142857143, 0.7685714285714286, 0.5914285714285714,
                       0.6271428571428571, 0.6085714285714285, 0.58, 0.3914285714285714, 0.6142857142857143,
                       0.7257142857142858, 0.7528571428571429, 0.6471428571428571,
                       0.5242857142857142, 0.6257142857142857, 0.66, 0.22857142857142856,
                       0.11714285714285715])  # 每个点的颜色值，范围为0到1

    # 使用PCA进行降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    # 提取降维后的数据
    x = reduced_data[:, 0]  # 第一维度作为 x 轴数据
    y = reduced_data[:, 1]  # 第二维度作为 y 轴数据

    # 绘制散点图
    plt.scatter(x, y, c=colors, cmap='RdYlGn')
    plt.colorbar()

    # 在显示图形之前调用tight_layout()函数
    plt.tight_layout()
    # 显示图形
    plt.show()

def workerScatter_t_SNE(data, colors=colors_9):
    # 实例化t-SNE模型
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=200)
    # tsne = TSNE(n_components=2)
    # 拟合数据
    X_tsne = tsne.fit_transform(data)
    # 可视化结果
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='RdYlGn')
    plt.show()

def workerScatter_KPCA(data, colors=colors_9):
    # 实例化KPCA模型
    kpca = KernelPCA(n_components=2, kernel='rbf')

    # 拟合数据
    X_kpca = kpca.fit_transform(data)

    # 可视化结果
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def workerScatter_ICA(data, colors=colors_9):
    # # 实例化ICA模型
    # ica = FastICA(n_components=2)
    #
    # # 拟合数据
    # X_ica = ica.fit_transform(data)
    #
    # # 可视化结果
    # plt.scatter(X_ica[:, 0], X_ica[:, 1], c=colors_9, cmap='RdYlGn')
    # plt.colorbar()
    # plt.show()

    # 实例化KPCA模型
    kpca = KernelPCA(n_components=3, kernel='rbf')

    # 拟合数据
    X_kpca = kpca.fit_transform(data)

    # 可视化结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=colors, cmap='RdYlGn')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # plt.colorbar()


    plt.tight_layout()
    plt.show()

def workerScatter_IOSMAP(data, colors=colors_9):
    # 实例化ISOMAP模型 34
    isomap = Isomap(n_components=2, n_neighbors=29)

    # 拟合数据
    X_isomap = isomap.fit_transform(data)

    # 可视化结果
    plt.scatter(X_isomap[:-1, 0], X_isomap[:-1, 1], c=colors, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.scatter(X_isomap[-1, 0], X_isomap[-1, 1], marker='*', color='blue', label='oracle worker')  # 绘制最后一组点为星星形状蓝色
    y_ticks = [-0.3, -0.1, 0.1, 0.3, 0.5]
    x_ticks = [-0.4, -0.1, 0.2, 0.5, 0.8]
    plt.xticks(fontsize=21, ticks=x_ticks)
    plt.yticks(fontsize=21, ticks=y_ticks)
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.show()

def workerScatterFactor(data, colors = colors_9):
    # 实例化因子分析模型
    fa = FactorAnalysis(n_components=2)

    # 拟合数据
    X_fa = fa.fit_transform(data)
    plt.scatter(X_fa[:, 0], X_fa[:, 1], c=colors, cmap='RdYlGn')
    plt.colorbar()
    plt.show()

def workerScatterLLE(data):
    # 实例化LLE模型
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=18)

    # 拟合数据
    X_lle = lle.fit_transform(data)
    plt.scatter(X_lle[:, 0], X_lle[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.show()

def workerScatter_umap(data):
    # 实例化UMAP模型
    umap_model = umap.UMAP(n_components=2)

    # 拟合数据
    X_umap = umap_model.fit_transform(data)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.show()

def workerScatter_mds(data):
    # 实例化MDS模型
    mds_model = MDS(n_components=2, dissimilarity='precomputed')

    # 拟合数据
    X_mds = mds_model.fit_transform(data)

    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.show()
if __name__ == '__main__':
    oracle_worker = [0.13820427203489055, 0.17980818216228567, 0.10761719211584557, 0.038788771730331156, 0.029946872563708732,
                   0.2463344734515049, 0.08802976167845389, 0.06770221475833456, 0.09193783730082672, 0.01163042220381843]
    numpy_array = np.load('../getOracleWorker/numpy_array.npy')
    print(numpy_array.shape)
    numpy_array = np.vstack((numpy_array, oracle_worker))
    # plotWorkerScatter.workerScatter(numpy_array)
    workerScatter_IOSMAP(numpy_array)
