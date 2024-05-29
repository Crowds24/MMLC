import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA, FastICA, KernelPCA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap

colors_9 = np.array([0.7383939774153074, 0.644918444165621, 0.6041405269761606, 0.6618569636135508, 0.6079046424090339, 0.7252195734002509, 0.6932245922208281, 0.6047678795483061, 0.6838143036386449, 0.6838143036386449, 0.7239648682559598, 0.6838143036386449, 0.705771643663739, 0.6599749058971142, 0.7063989962358845, 0.6863237139272271, 0.7296110414052698, 0.7164366373902133, 0.7308657465495608, 0.7258469259723965, 0.698870765370138, 0.7202007528230866, 0.7220828105395232, 0.7358845671267252, 0.7271016311166876, 0.7559598494353826, 0.6085319949811794, 0.7340025094102886, 0.7214554579673776, 0.6825595984943539, 0.7183186951066499, 0.6141781681304893, 0.5633626097867002, 0.726474278544542, 0.7233375156838143, 0.6204516938519448, 0.6718946047678795, 0.6555834378920954, 0.6668757841907151, 0.7484316185696361, 0.7107904642409034, 0.7170639899623589, 0.7358845671267252, 0.7421580928481807, 0.7001254705144291, 0.7465495608531995, 0.5388958594730239, 0.7409033877038896, 0.7032622333751568, 0.6907151819322459, 0.7346298619824341, 0.7365119196988708, 0.6329987452948557, 0.7202007528230866, 0.698870765370138, 0.6637390213299874, 0.7302383939774153, 0.6913425345043914, 0.6499372647427855, 0.7371392722710163, 0.7358845671267252, 0.6185696361355082, 0.7070263488080301, 0.7120451693851945, 0.739021329987453, 0.5821831869510665, 0.7346298619824341, 0.7459222082810539, 0.698870765370138, 0.7434127979924717, 0.7421580928481807, 0.7202007528230866, 0.6323713927227101, 0.6593475533249686, 0.7132998745294856, 0.7314930991217063, 0.7352572145545797, 0.7132998745294856, 0.6919698870765371, 0.7051442910915935, 0.733375156838143, 0.7277289836888331, 0.7465495608531995, 0.7271016311166876, 0.7302383939774153, 0.6442910915934755, 0.7578419071518193, 0.6016311166875784, 0.6863237139272271, 0.7509410288582183, 0.7365119196988708, 0.6279799247176914, 0.733375156838143, 0.740276035131744, 0.6386449184441656, 0.733375156838143, 0.7208281053952321, 0.7038895859473023, 0.6762860727728983, 0.7095357590966123, 0.5865746549560853, 0.6706398996235885, 0.678168130489335, 0.679422835633626, 0.6706398996235885, 0.7283563362609786, 0.7521957340025094, 0.7314930991217063, 0.676913425345044, 0.6668757841907151, 0.7095357590966123, 0.6938519447929736, 0.7220828105395232, 0.7139272271016311, 0.6587202007528231, 0.7383939774153074, 0.6543287327478042, 0.7239648682559598, 0.7314930991217063, 0.6700125470514429, 0.7346298619824341, 0.622961104140527, 0.7327478042659975, 0.6725219573400251, 0.7051442910915935, 0.7484316185696361, 0.6869510664993727, 0.6543287327478042, 0.6831869510664994, 0.7308657465495608, 0.7346298619824341, 0.7327478042659975, 0.7427854454203262, 0.7383939774153074, 0.7352572145545797, 0.7358845671267252, 0.7503136762860728, 0.740276035131744, 0.6010037641154329, 0.7070263488080301, 0.7308657465495608, 0.7296110414052698, 0.6963613550815558, 0.7189460476787954, 0.7371392722710163, 0.6888331242158093, 0.7427854454203262, 0.7415307402760352, 0.6969887076537014, 0.7371392722710163, 0.7277289836888331, 0.7239648682559598, 0.7503136762860728])

def workerScatter(data, colors=colors_9):


    colors_9 = colors
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    # 提取降维后的数据
    x = reduced_data[:, 0]  # 第一维度作为 x 轴数据
    y = reduced_data[:, 1]  # 第二维度作为 y 轴数据
    # 绘制散点图
    plt.scatter(x, y, c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.tight_layout()
    # 显示图形
    plt.show()

def workerScatter_t_SNE(data):
    # 实例化t-SNE模 50-500 均匀散点
    tsne = TSNE(n_components=2, perplexity=100, learning_rate=300)
    # 拟合数据
    X_tsne = tsne.fit_transform(data)
    # 可视化结果
    plt.scatter(X_tsne[:-1, 0], X_tsne[:-1, 1], c=colors_9, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    plt.scatter(X_tsne[-1, 0], X_tsne[-1, 1], marker='*', color='blue')  # 绘制最后一组点为星星形状蓝色
    plt.show()

def workerScatter_ICA(data, colors =colors_9):
    # 实例化ICA模型
    ica = FastICA(n_components=2)

    # 拟合数据
    X_ica = ica.fit_transform(data)

    # 可视化结果
    plt.scatter(X_ica[:-1, 0], X_ica[:-1, 1], c=colors_9, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    plt.scatter(X_ica[-1, 0], X_ica[-1, 1], marker='*', color='blue')  # 绘制最后一组点为星星形状蓝色
    plt.colorbar()
    plt.show()


def workerScatter_KPCA(data, colors = colors_9):
    # # 实例化KPCA模型
    # kpca = KernelPCA(n_components=2, kernel='rbf')
    #
    # # 拟合数据
    # X_kpca = kpca.fit_transform(data)
    #
    # # 可视化结果
    # # 绘制散点图
    # plt.scatter(X_kpca[:-1, 0], X_kpca[:-1, 1], c=colors_9, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    # plt.scatter(X_kpca[-1, 0], X_kpca[-1, 1], marker='*', color='blue')  # 绘制最后一组点为星星形状蓝色
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # 实例化KPCA模型
    kpca = KernelPCA(n_components=3, kernel='rbf')

    # 拟合数据
    X_kpca = kpca.fit_transform(data)

    # 可视化结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=colors, cmap='RdYlGn')

    # plt.colorbar()

    plt.tight_layout()
    plt.show()


def workerScatterFactor(data, colors = colors_9):
    # 实例化因子分析模型
    fa = FactorAnalysis(n_components=2)

    # 拟合数据
    X_fa = fa.fit_transform(data)
    plt.scatter(X_fa[:, 0], X_fa[:, 1], c=colors_9, cmap='RdYlGn')
    plt.colorbar()
    plt.show()

def workerScatter_IOSMAP(data, colors=colors_9):
    # 实例化ISOMAP模型 12
    isomap = Isomap(n_components=2, n_neighbors=12)

    # 拟合数据
    X_isomap = isomap.fit_transform(data)
    x_ticks = [-0.6, -0.3, 0.0, 0.3 ,0.6]
    # 可视化结果
    plt.scatter(X_isomap[:-1, 0], X_isomap[:-1, 1], c=colors, cmap='RdYlGn')  # 绘制除了最后一组点的散点
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.scatter(X_isomap[-1, 0], X_isomap[-1, 1], marker='*', color='blue', label='oracle worker')  # 绘制最后一组点为星星形状蓝色
    plt.xticks(fontsize=21, ticks=x_ticks)
    plt.yticks(fontsize=21)
    # 将图例放置在colorbar下方
    plt.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.show()

def arrow3D(ax, x, y, z, dx, dy, dz, arrow_length=1.0, arrow_color="black"):
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color=arrow_color)
    arrow_x = arrow_length * dx
    arrow_y = arrow_length * dy
    arrow_z = arrow_length * dz
    ax.quiver(x, y, z, arrow_x, arrow_y, arrow_z, **arrow_prop_dict)


def workerScatter_IOSMAP_3D(data, colors=colors_9):
    # 实例化KPCA模型
    isomap = Isomap(n_components=3, n_neighbors=12)

    # 拟合数据
    X_isomap = isomap.fit_transform(data)

    # 可视化结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 设置超出范围的点为NaN
    x = X_isomap[:, 0] + 0.75
    y = X_isomap[:, 1] + 0.7
    z = X_isomap[:, 2] + 0.35
    x_range = (x > X_isomap[:, 0].min() + 0.6) & (x < X_isomap[:, 0].max() + 0.2)
    y_range = (y > X_isomap[:, 1].min()+0.2) & (y < X_isomap[:, 1].max() - 0.1)
    z_range = (z > X_isomap[:, 2].min() + 0.3) & (z < X_isomap[:, 2].max() +0.07)
    valid_points = x_range & y_range & z_range
    x_valid = np.where(valid_points, x, np.nan)
    y_valid = np.where(valid_points, y, np.nan)
    z_valid = np.where(valid_points, z, np.nan)

    # 设置3D坐标轴的背景颜色为白色
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.invert_yaxis()  # 改变 y 轴方向

    ax.scatter(X_isomap[:, 0]+0.75, X_isomap[:, 1]+0.7, X_isomap[:, 2]+0.35, c=colors, cmap='RdYlGn')
    # ax.scatter(x_valid, y_valid, z_valid, c=colors, cmap='RdYlGn')
    ax.quiver(-0.1, 0, 0, X_isomap[:, 0].max() - X_isomap[:, 0].min() - 0.25, 0, 0,
              color='black', arrow_length_ratio=0.1)
    ax.quiver(0, -0.15, 0, 0, X_isomap[:, 1].max() - X_isomap[:, 1].min()- 0.3, 0,
              color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, -0.1, 0, 0, X_isomap[:, 2].max() - X_isomap[:, 2].min() - 0.15,
              color='black', arrow_length_ratio=0.1)
    # 设置坐标轴范围
    ax.set_xlim3d(X_isomap[:, 0].min() + 0.6, X_isomap[:, 0].max() + 0.2)
    ax.set_ylim3d(X_isomap[:, 1].max() - 0.1, X_isomap[:, 1].min()+0.2)
    ax.set_zlim3d(X_isomap[:, 2].min() + 0.3, X_isomap[:, 2].max() +0.07)

    # 隐藏坐标数字
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.grid(True)
    # plt.tight_layout()
    plt.show()

    plt.savefig('save_path')
if __name__ == '__main__':
    oracle_worker = [0.053305521494511406, 0.07019503908514073, 0.031879648603875224, 0.07694784924387932,
                     0.04160289917461047, 0.04587434254812472,
                     0.09807082265615463, 0.049070558915234576, 0.08418243420733647, 0.11319732726222337]
    numpy_array = np.load('../getOracleWorker/numpy_array.npy')
    print(numpy_array.shape)
    numpy_array = np.vstack((numpy_array, oracle_worker))
    # plotWorkerScatter.workerScatter(numpy_array)
    workerScatter_IOSMAP(numpy_array)
