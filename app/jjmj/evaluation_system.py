import os
import pandas as pd
import numpy as np
from datetime import timedelta, date

from .config import parameters as pt
from .basic.station import Station

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties
from pandas.plotting import register_matplotlib_converters

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
register_matplotlib_converters()


def evaluationSystem():
    date_range = pd.date_range(pt.start, pt.end, freq='m')
    col = ['实际可放电量', '电站综合效率', '电站储能损耗率',
           '站用电率', '电站等效利用系数', '电站可用系数']
    df_station = pd.DataFrame(columns=col, index=date_range)
    dir_ = os.path.join(os.getcwd(), 'excel')
    df_history = pd.DataFrame()
    if os.path.exists(os.path.join(dir_, 'station_data.xlsx')):
        df_history = pd.read_excel(os.path.join(dir_, 'station_data.xlsx'), index_col=0, header=0)

    for d in date_range:
        start = pd.to_datetime(date(d.year, d.month, 1))
        end = pd.to_datetime(date(d.year, d.month+1, 1))

        if d in df_history.index:
            df_station.loc[d] = df_history.loc[d].values
        else:
            df_station.loc[d] = get_station_indicator(start, end)

    print(df_station)
    # df_station.to_excel(os.path.join(dir_, 'station_data.xlsx'))

    # cal_score(df_station)

    # plt_comparison(df_station)


def get_station_indicator(start, end):
    station = Station(pt.project, start, end)
    # station.get_actual_work_hour()

    # 指标层：充放电能力
    # 1.实际可充放电效率
    # TODO

    # 2.实际可放电量
    dis_efficiency = station.get_dis_efficiency()

    # 指标层：能效水平
    # 1.电站综合效率
    combined_efficiency = station.get_combined_efficiency()

    # 2.电站储能损耗率
    loss_efficiency = station.get_loss_efficiency()

    # 3.站用电率
    self_efficiency = station.get_self_efficiency()

    # 指标层：设备运行状态
    # 1.电站等效利用系数
    coff_equa = station.get_coff_equa_use()

    # 2.电站可用系数
    coff_use = station.get_coff_use()

    return [dis_efficiency, combined_efficiency, loss_efficiency, self_efficiency, coff_equa, coff_use]


def set_weight(df, idx):
    col = df.columns
    df_norm = pd.DataFrame(columns=col)
    df_weight = pd.DataFrame(columns=col)
    df_p = pd.DataFrame(columns=col)
    for i in range(df.shape[-1]):
        df_norm[col[i]] = normalization_factor(df[col[i]], idx[i])
        # print(df_norm)
        df_p[col[i]] = df_norm[col[i]].apply(lambda x: x/df_norm[col[i]].sum())

    k = 1 / np.log(df.shape[0])
    e = []
    for i in range(df.shape[-1]):
        e_i = -k * sum(p * np.log(p) for p in df_p[col[i]])
        e.append(e_i)

    d = [1 - e_i for e_i in e]

    df_weight.loc[len(df_weight)] = [d_i/np.sum(d) for d_i in d]

    print(df_weight)
    return df_weight


def normalization_factor(df, type_):
    y_min = 0.002
    y_max = 0.998
    if type_ == 1:
        df_norm = df.apply(lambda x: (y_max - y_min)*(x - df.min()) / (df.max() - df.min()) + y_min)
    elif type_ == -1:
        df_norm = df.apply(lambda x: (y_max - y_min)*(df.max() - x) / (df.max() - df.min()) + y_min)
    else:
        raise ValueError("标准化类型值错误！应为1或-1")

    return df_norm


def cal_score(df):
    idx = [1, 1, -1, -1, 1, 1]
    for idx in df.index:
        df_month = df.loc[idx]
        score_combined = set_combined_efficiency_score(self.get_combined_efficiency())
        score_loss = set_loss_efficiency_score(self.get_loss_efficiency())
        score_self = set_station_efficiency_score(self.get_self_efficiency())
        score_ac = set_combined_efficiency_score(self.get_ac_efficiency())


def plt_comparison(df):
    """
    画出每月各项指标对比图
    :param df:
    :return:
    """
    fig = plt.figure(figsize=(12, 8))
    bar_width = 0.3
    N = np.arange(df.shape[-1])

    for i in range(df.shape[0]):
        y = df.iloc[i]
        plt.bar(N + bar_width * i, y, bar_width, label=df.index[i].strftime('%Y-%m-%d')[:7])
        for a, b in zip(N + bar_width*i, y):
            plt.text(a, b + 0.0005, '%.4f' % b, ha='center', va='bottom', fontsize=15)

    plt.xticks(N + bar_width/2, df.columns)
    title = '电站评价指标'
    plt.title(title, fontproperties=font)
    plt.xlabel('指标', fontproperties=font)
    plt.ylabel('值/%', fontproperties=font)
    plt.legend(prop=font, loc='upper right')
    dir_ = os.path.join(os.getcwd(), 'figure', title + '.jpg')
    plt.savefig(dir_)
    plt.show()


def set_combined_efficiency_score(value):
    if value >= 90:
        val = 100
    elif value >= 80:
        val = 90
    elif value >= 70:
        val = 80
    elif value >= 60:
        val = 70
    else:
        val = 60

    return val


def set_loss_efficiency_score(value):
    if value <= 10:
        val = 100
    elif value <= 20:
        val = 95
    elif value <= 30:
        val = 90
    elif value <= 40:
        val = 85
    else:
        val = 80

    return val


def set_station_efficiency_score(value):
    if value <= 5:
        val = 100
    elif value <= 10:
        val = 90
    elif value <= 15:
        val = 80
    elif value <= 20:
        val = 70
    else:
        val = 60

    return val