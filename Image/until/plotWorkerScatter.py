import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA, KernelPCA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

# colors_9 = np.array([0.7471852610030706, 0.7584442169907881, 0.7707267144319345, 0.7256908904810645, 0.7359263050153532, 0.7604912998976459, 0.7789150460593655, 0.7461617195496417, 0.7676560900716479, 0.7737973387922211, 0.7502558853633572, 0.7676560900716479, 0.797338792221085, 0.77482088024565, 0.6980552712384852, 0.7553735926305015, 0.7881269191402251, 0.7717502558853634, 0.7420675537359263, 0.7031729785056294, 0.7707267144319345, 0.7338792221084954, 0.7635619242579325, 0.7727737973387923, 0.7328556806550666, 0.7645854657113613, 0.7359263050153532, 0.7502558853633572, 0.7185261003070624, 0.7881269191402251, 0.7349027635619243, 0.7308085977482088, 0.7563971340839304, 0.7236438075742068, 0.8085977482088025, 0.7645854657113613, 0.7021494370522006, 0.72978505629478, 0.7911975435005117, 0.7420675537359263, 0.6560900716479018, 0.7758444216990789, 0.6683725690890481, 0.7164790174002047, 0.7318321392016377, 0.6264073694984647, 0.7737973387922211, 0.631525076765609, 0.7983623336745138, 0.7349027635619243, 0.7778915046059366, 0.7584442169907881, 0.7400204708290685, 0.7410440122824974, 0.7082906857727738, 0.7758444216990789, 0.8075742067553736, 0.6581371545547595, 0.6847492323439099])
colors_9 = np.array([0.7554, 0.7554, 0.7613, 0.7453, 0.7945, 0.7653, 0.8144, 0.7337, 0.7842, 0.7143, 0.7599, 0.8039, 0.8206, 0.784, 0.6897, 0.7296, 0.8146, 0.6499, 0.719, 0.7367, 0.7912, 0.7294, 0.7684, 0.8064, 0.7509, 0.7598, 0.7217, 0.7542, 0.6898, 0.7791, 0.7636, 0.7658, 0.7214, 0.7305, 0.7114, 0.7937, 0.6568, 0.7573, 0.7905, 0.7304, 0.6784, 0.7917, 0.6733, 0.7164, 0.7284, 0.7074, 0.7652, 0.7029, 0.79, 0.7568, 0.8026, 0.7837, 0.7387, 0.7926, 0.7629, 0.6922, 0.783, 0.6687, 0.7032])
def workerScatter(data):
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    # 提取降维后的数据
    x = reduced_data[:, 0]  # 第一维度作为 x 轴数据
    y = reduced_data[:, 1]  # 第二维度作为 y 轴数据

    # 绘制散点图
    plt.scatter(x, y)

    # 设置图形标题和标签

    # plt.xlabel('Dimension')
    # plt.ylabel('Dimension')
    # 在显示图形之前调用tight_layout()函数
    plt.tight_layout()
    # 显示图形
    plt.show()


def workerScatter_t_SNE(data, colors=colors_9):
    # 实例化t-SNE模型 50 10
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=10)
    # 拟合数据
    X_tsne = tsne.fit_transform(data)
    plt.scatter(X_tsne[:-1, 0], X_tsne[:-1, 1], c=colors, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.scatter(X_tsne[-1, 0], X_tsne[-1, 1], marker='*', color='blue', label='oracle worker')  # 绘制最后一组点为星星形状蓝色

    # 将图例放置在colorbar下方
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    # 将图例放置在colorbar下方
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.show()

def workerScatter_ICA(data, colors =colors_9):
    # 实例化ICA模型
    ica = FastICA(n_components=2)

    # 拟合数据
    X_ica = ica.fit_transform(data)


    # 可视化结果
    plt.scatter(X_ica[:, 0], X_ica[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.tight_layout()
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
    # # 实例化KPCA模型
    # kpca = KernelPCA(n_components=3, kernel='rbf')
    #
    # # 拟合数据
    # X_kpca = kpca.fit_transform(data)
    #
    # # 可视化结果
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=colors, cmap='RdYlGn')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    #
    # # plt.colorbar()
    #
    #
    # plt.tight_layout()
    # plt.show()


def workerScatter_IOSMAP(data, colors=colors_9):
    # 实例化ISOMAP模型
    isomap = Isomap(n_components=2, n_neighbors=36)

    # 拟合数据
    X_isomap = isomap.fit_transform(data)
    print(X_isomap)
    # 可视化结果
    plt.scatter(X_isomap[:-1, 0], X_isomap[:-1, 1], c=colors, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    plt.colorbar()
    plt.scatter(X_isomap[-1, 0], X_isomap[-1, 1], marker='*', color='blue', label='oracle')  # 绘制最后一组点为星星形状蓝色

    # 将图例放置在colorbar下方
    plt.legend()
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
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=30)

    # 拟合数据
    X_lle = lle.fit_transform(data)
    plt.scatter(X_lle[:, 0], X_lle[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.show()
if __name__ == '__main__':
    # 加载保存的NumPy数组
    oracle_worker = [0.1274182411880429, 0.10120543142093887, 0.10566259603864457, 0.11069937425046995,
                     0.12472410703131681, 0.04037496737383016,
                     0.07515931281861461, 0.12534738671061416, 0.11218173187960483, 0.07722685128792318]
    numpy_array = np.load('../getOracleWorker/numpy_array.npy')
    print(numpy_array.shape)
    numpy_array = np.vstack((numpy_array, oracle_worker))
    # plotWorkerScatter.workerScatter(numpy_array)
    workerScatter_IOSMAP(numpy_array)
