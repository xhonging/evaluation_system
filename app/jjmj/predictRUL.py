# coding:utf-8

import os
import pandas as pd
import numpy as np
import joblib     # 保存读取模型
from sklearn.model_selection import train_test_split, cross_val_score, \
    cross_validate, KFold, LeaveOneOut, cross_val_predict, GroupKFold
from sklearn import preprocessing  # 预处理模块
from sklearn.metrics import explained_variance_score  # 模型度量

# 模型
from sklearn.linear_model import LinearRegression   # 线性多元回归
from sklearn.svm import SVR                         # 支持向量回归
from sklearn.tree import DecisionTreeRegressor      # 决策树回归
from sklearn.neighbors import KNeighborsRegressor   # knn回归
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
import xgboost as xgb                               # XGBoost回归
from sklearn.neural_network import MLPRegressor     # 神经网络MLP回归
import lightgbm as lgb                              # LightGBM回归

from .config import parameters as pt
from .basic.battery import BatteryCluster
from .common.function import get_rated_value

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pandas.plotting import register_matplotlib_converters

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
register_matplotlib_converters()
sns.set_style('white', {'font.sans-serif': ['simhei', 'Arial']})
# import warnings
# warnings.filterwarnings("ignore")


def get_data(cluster_name):
    cluster = BatteryCluster(cluster_name, pt.start, pt.end)
    col = ['充电量', '充电时长', '充电倍率', 'SOC极差',
           '电压极差', '平均电压', '温度极差', '平均温度']
    # col = ['充电量', '充电时长', '充电倍率', 'SOC极差', '最高电压', '最低电压',
    #        '电压极差', '平均电压', '温度极差', '平均温度', '最高温度', '最低温度']
    # , '累计充电次数'
    data = pd.DataFrame(columns=col)
    # 充电电量
    chg_quantity = cluster.get_chg_detail()
    # 充电时长
    chg_time = cluster.get_chg_span()
    # 充电倍率
    chg_elec = cluster.get_chg_elec()
    # SOC极差
    chg_diff_soc = cluster.get_chg_diff_soc()
    # 最低、最高电压极差
    chg_vol = cluster.get_chg_vol()
    chg_max_vol = chg_vol['max']
    chg_min_vol = chg_vol['min']
    chg_diff_vol = chg_vol['max'] - chg_vol['min']
    # 电压均值
    chg_mean_vol = cluster.get_chg_mean_vol()
    # 最低、最高温度极差
    chg_temp = cluster.get_chg_temp()
    chg_max_temp = chg_temp['max']
    chg_min_temp = chg_temp['min']
    chg_diff_temp = chg_temp['max'] - chg_temp['min']
    # 温度均值
    chg_mean_temp = cluster.get_chg_mean_temp()
    # 充电次数
    chg_number = cluster.get_chg_number()

    data['充电量'] = chg_quantity
    data['充电时长'] = chg_time
    data['充电倍率'] = chg_elec
    data['SOC极差'] = chg_diff_soc
    # data['最高电压'] = chg_max_vol
    # data['最低电压'] = chg_min_vol
    data['电压极差'] = chg_diff_vol
    data['平均电压'] = chg_mean_vol
    # data['最高温度'] = chg_max_temp
    # data['最低温度'] = chg_min_temp
    data['温度极差'] = chg_diff_temp
    data['平均温度'] = chg_mean_temp

    # print(data.head())
    # data = data.loc[data['SOC极差'] > 85]
    data = data.dropna()
    # print(data.head())

    data['累计充电次数'] = None
    # df_corr = data.corr()
    for d in data.index:
        data.loc[d, '累计充电次数'] = chg_number[d]
    # print(data.head())
    data = data.apply(lambda x: x.astype(float))
    return data


def get_data_en(cluster_name, start, end):
    cluster = BatteryCluster(cluster_name, start, end)
    col = ['quantity', 'span', 'elec', 'diff_SOC',
           'diff_vol', 'mean_vol', 'diff_temp', 'mean_temp']
    data = pd.DataFrame(columns=col)
    # 充电电量
    chg_quantity = cluster.get_chg_detail()
    # 充电时长
    chg_time = cluster.get_chg_span()
    # 充电倍率
    chg_elec = cluster.get_chg_elec()
    # SOC极差
    chg_diff_soc = cluster.get_chg_diff_soc()
    # 最低、最高电压极差
    chg_vol = cluster.get_chg_vol()
    chg_diff_vol = chg_vol['max'] - chg_vol['min']
    # 电压均值
    chg_mean_vol = cluster.get_chg_mean_vol()
    # 最低、最高温度极差
    chg_temp = cluster.get_chg_temp()
    chg_diff_temp = chg_temp['max'] - chg_temp['min']
    # 温度均值
    chg_mean_temp = cluster.get_chg_mean_temp()
    # 充电次数
    chg_number = cluster.get_chg_number()

    data['quantity'] = chg_quantity
    data['span'] = chg_time
    data['elec'] = chg_elec
    data['diff_SOC'] = chg_diff_soc
    data['diff_vol'] = chg_diff_vol
    data['mean_vol'] = chg_mean_vol
    data['diff_temp'] = chg_diff_temp
    data['mean_temp'] = chg_mean_temp
    data = data.dropna()

    data['num'] = None
    for d in data.index:
        data.loc[d, 'num'] = chg_number[d]
    data = data.apply(lambda x: x.astype(float))
    return data


def ana_data(cluster_name):
    data = get_data(cluster_name)

    df_corr = data.corr()
    df_corr.to_excel(os.path.join(os.getcwd(), 'excel', 'corr.xlsx'))
    # 数据描述
    print(data.describe())
    # 缺失值检验
    print(data[data.isnull() == True].count())

    # data.boxplot()
    # plt.savefig("boxplot.jpg")
    # plt.show()
    ##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
    # 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
    print(data.corr())

    sns.pairplot(data, x_vars=['充电时长', '充电倍率', 'SOC极差', '累计充电次数',
           '电压极差', '平均电压', '温度极差', '平均温度'],
                 y_vars='充电量', size=7, aspect=0.8, kind='reg')
    # plt.savefig("pairplot.jpg")
    plt.show()


def model_KF(cluster_name):
    data = get_data(cluster_name)
    data = data.drop('温度极差', axis=1)
    X = data.iloc[:, 1:]
    scaler = preprocessing.StandardScaler().fit(X)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
    X_stand = scaler.transform(X)
    # X_stand = np.array(X)
    y = data['充电量']

    # print('K-折交叉验证')
    list_a = []
    list_b = []
    list_score = []
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    for train_index, test_index in kf.split(X_stand):
        # print("k折划分：%s %s" % (train_index, test_index))
        model = LinearRegression()
        X_train, y_train = X_stand[train_index], y[train_index]
        X_test, y_test = X_stand[test_index], y[test_index]
        # print('训练集大小：', X_train.shape, y_train.shape)  # 训练集样本大小
        # print('测试集大小：', X_test.shape, y_test.shape)  # 测试集样本大小

        model.fit(X_train, y_train)
        a = model.intercept_  # 截距
        b = model.coef_  # 回归系数
        score = model.score(X_test, y_test)

        print("最佳拟合线:截距", a, ",回归系数：", b)
        print('得分：', score)
        print('****')
        # if b[-1] < 0.0:
        list_a.append(a)
        list_b.append(b)
        list_score.append(score)

    pt_a = np.mean(list_a)
    pt_b = np.array(list_b).mean(axis=0)
    pt_score = np.mean(list_score)
    # print(pt_a, pt_b, pt_score)

    # y_pred = pt_a + np.dot(X_stand, pt_b)
    # col = ['充电时长', '充电倍率', 'SOC极差',
    #        '电压极差', '平均电压', '平均温度', '累计充电次数']
    #
    # # print(np.around(pt_b, decimals=4))
    # s = '充电量'+'='+str(pt_a)+'+'+'+'.join('('+str(i[1])+'*'+i[0]+')' for i in zip(col, np.around(pt_b, decimals=4)))
    # print(s)
    #
    # plt.figure()
    # plt.plot(y.index, y_pred, 'b', label="predict")
    # plt.plot(y.index, y, 'r', label="actual")
    # plt.legend(loc="best")  # 显示图中的标签
    # plt.xticks(y.index, rotation=90)
    # plt.xlabel("日期", fontproperties=font)
    # plt.ylabel('充电量', fontproperties=font)
    # plt.title(cluster_name + 'ROC曲线', fontproperties=font)
    # # plt.savefig("ROC.jpg")
    # plt.show()

    # print(X.iloc[-1:, :])
    # print(y[-1])
    if pt_score > 0:
        pred_x = X_stand[-1]
        rul = cal_rul(cluster_name, pred_x, y[-1], pt_a, pt_b)
    else:
        rul = 0

    return np.around(pt_score, decimals=4), rul


def cal_rul(name, pred_x, y, a, b):
    # d = pred_x.index[0]
    rated_chg = get_rated_value()[name]
    pred_num = (rated_chg * 0.8 - a - sum(i[0]*i[1] for i in zip(pred_x[:-1], b[:-1]))) / b[-1]
    rul = pred_num - y
    # print(rul)
    # num = pred_x['累计充电次数'][d]

    # for i in zip(pred_x, b):
    #     print(i)

    # while y > rated_chg*0.8:
    #     num = num + 1
    #     pred_x['累计充电次数'] = pred_x['累计充电次数'].map(lambda x: x+1)
    #     # print(pred_x)
    #     x_stand = scaler.transform(pred_x)
    #     y = a + np.dot(x_stand, b)
        # print(y)

    # print(num)

    # y_pred = a + np.dot(X_stand, b)
    return rul


def model_selected(name):
    # data = get_data(name)
    # data = data.drop('温度极差', axis=1)
    data = get_data_en(name, pt.start, pt.end)
    data = data.drop('diff_temp', axis=1)
    X = data.iloc[:, 1:]
    # scaler = preprocessing.StandardScaler().fit(X)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
    # X_stand = scaler.transform(X)
    # y = data['充电量']
    y = data['quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=241)

    models = {'线性多元回归': LinearRegression(),
              '支持向量回归': SVR(gamma='scale'),
              '决策树回归': DecisionTreeRegressor(),
              'KNN回归': KNeighborsRegressor(n_neighbors=5, weights='distance'),
              '随机森林回归': RandomForestRegressor(max_depth=3, random_state=0, n_estimators=200),
              'XGBoost回归': xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100,
                                            objective='reg:squarederror', n_jobs=-1),
              'LightGBM回归': lgb.LGBMRegressor(num_leaves=50, learning_rate=0.1, n_estimators=100)
              }

    col = pd.MultiIndex.from_product([[name], ['score', 'model']])
    df_model = pd.DataFrame(columns=col, index=models.keys())
    for k in models.keys():
        # print(k)
        model = models[k]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # print(score)
        df_model.loc[k][name] = [score, model]

    return df_model


def predict_url_KF():
    pcs_names = ['PCS' + str(i + 1) for i in range(pt.battery['PCS'])]
    for pcs in pcs_names:
        cluster_names = ['{}_{}'.format(pcs, str(i + 1)) for i in range(pt.battery['modules'])]
        df_model_score = pd.DataFrame(columns=cluster_names, index=[0])
        df_rul = pd.DataFrame(columns=cluster_names, index=[0])
        for cluster in cluster_names:
            dir_ = os.path.join(os.getcwd(), 'models', pcs, cluster)
            print(cluster)
            model_select = model_selected(cluster)
            # df_model_score[cluster][0] = score
            # df_rul[cluster][0] = rul

            break
        # print(df_model_score)
        # print(df_rul)
        # path = os.path.join(os.getcwd(), 'excel', 'model_score_{}.xlsx'.format(pcs))
        # rul_path = os.path.join(os.getcwd(), 'excel', 'rul_{}.xlsx'.format(pcs))
        # if os.path.exists(path):
        #     os.remove(path)
        # df_model_score.to_excel(path)
        # if os.path.exists(rul_path):
        #     os.remove(rul_path)
        # df_rul.to_excel(rul_path)
        break
    # ana_data('PCS1_1')


def train_model():
    pcs_names = ['PCS' + str(i + 1) for i in range(pt.battery['PCS'])]
    df_score = pd.DataFrame()
    for pcs in pcs_names:
        cluster_names = ['{}_{}'.format(pcs, str(i + 1)) for i in range(pt.battery['modules'])]
        for cluster in cluster_names:
            # print(cluster)
            df_model = model_selected(cluster)

            ds = df_model[cluster]['score'].apply(lambda x: x.astype(float))

            index = ds.idxmax()
            model_best = df_model.loc[index][cluster]['model']
            joblib.dump(model_best, os.path.join(os.getcwd(), 'models', cluster + '.model'))

            df_score[cluster] = ds
            # print(df_score)

        #     break
        # break
    df_score.to_excel(os.path.join(os.getcwd(), 'excel', '模型训练得分.xlsx'))


def test_model():
    pcs_names = ['PCS' + str(i + 1) for i in range(pt.battery['PCS'])]
    df_test = pd.DataFrame(index=['actual', 'predict'])
    for pcs in pcs_names:
        cluster_names = ['{}_{}'.format(pcs, str(i + 1)) for i in range(pt.battery['modules'])]
        for cluster in cluster_names:
            print(cluster)
            model = joblib.load(os.path.join(os.getcwd(), 'models', cluster + '.model'))

            data = get_data_en(cluster, pt.test_start, pt.test_end)
            data = data.drop('diff_temp', axis=1)
            X = data.iloc[:, 1:]
            # scaler = preprocessing.StandardScaler().fit(X)  # 通过训练集获得归一化函数模型。（也就是先减几，再除以几的函数）。在训练集和测试集上都使用这个归一化函数
            # X_stand = scaler.transform(X)
            y = data['quantity']
            y_prec = model.predict(X)
            print(y_prec)
            print(y)
            score = explained_variance_score(y, y_prec)
            print(score)
            df_test[cluster] = [y.values[0], y_prec[0]]
            # print(df_test)

            # break
        # break
    print(df_test)
    df_test.to_excel(os.path.join(os.getcwd(), 'excel', '04-30预测结果.xlsx'))





