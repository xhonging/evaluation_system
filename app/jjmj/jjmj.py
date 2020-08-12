import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.font_manager import FontProperties
from pandas.plotting import register_matplotlib_converters

from .config import parameters as pt
from .dataset import vol
from .basic.battery import BatteryCluster

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
register_matplotlib_converters()


def min_vol():
    ids_min_vol = []
    for m in range(1, pt.battery['modules'] + 1):
        s = '南通通州积美$南通通州积美$PCS1_{}#电池组最低电压'.format(m)
        id = ['analoginput', s]
        ids_min_vol.append(id)

    for start in pd.date_range(pt.start, pt.end, freq='D', closed='left'):
        end = start + timedelta(days=1)
        min_vol = vol(ids_min_vol, start, end)

        plt_vol(min_vol)
        # break


def max_vol():
    ids_max_vol = []

    for m in range(1, pt.battery['modules'] + 1):
        s = '南通通州积美$南通通州积美$PCS1_{}#电池组最高电压'.format(m)
        id = ['analoginput', s]
        ids_max_vol.append(id)

    for start in pd.date_range(pt.start, pt.end, freq='D', closed='left'):
        end = start + timedelta(days=1)
        max_vol = vol(ids_max_vol, start, end)

        plt_vol(max_vol)
        # break


def vol_diff():
    ids_min_vol = []
    for m in range(1, pt.battery['modules'] + 1):
        s = '南通通州积美$南通通州积美$PCS1_{}#电池组最低电压'.format(m)
        id = ['analoginput', s]
        ids_min_vol.append(id)

    ids_max_vol = []
    for m in range(1, pt.battery['modules'] + 1):
        s = '南通通州积美$南通通州积美$PCS1_{}#电池组最高电压'.format(m)
        id = ['analoginput', s]
        ids_max_vol.append(id)

    for start in pd.date_range(pt.start, pt.end, freq='D', closed='left'):
        end = start + timedelta(days=1)
        min_vol = vol(ids_min_vol, start, end)
        max_vol = vol(ids_max_vol, start, end)

        col = range(len(max_vol.columns))
        vol_diff = pd.DataFrame(columns=col)
        for i in col:
            vol_diff[i] = max_vol[max_vol.columns[i]] - min_vol[min_vol.columns[i]]

        print(vol_diff)
        break





def temp():
    pass


def plt_vol(df):
    fig = plt.figure(figsize=(12, 8))
    for i in df.columns:
        plt.plot(df[i], label=i)
    plt.title(df.index[0].strftime('%Y-%m-%d'))
    plt.legend(prop=font, loc='best')
    plt.show()
    plt.clf()


def plot_chg_soc():
    name = 'PCS1_1'
    PCS1_1 = BatteryCluster(name, pt.start, pt.end)
    df_chg = PCS1_1.get_chg_detail()
    df_soc = PCS1_1.get_chg_soc()

    df_chg = df_chg.dropna()
    df_soc = df_soc.dropna()

    fig = plt.figure(figsize=(12, 8))
    y1_major_locator = MultipleLocator(2)   # 把x轴的刻度间隔设置为1，并存在变量里
    y2_major_locator = MultipleLocator(10)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax1 = fig.add_subplot(111)
    # ax1.scatter(x=df_chg.index, y=df_chg, marker='v', label='chg')
    ax1.plot(df_chg, c='yellow')
    ax1.plot(df_chg, marker='s', c='blue', label='chg')

    ax1.set_ylabel('充电量', fontproperties=font)
    ax1.set_xlim(datetime.strptime(pt.start, '%Y-%m-%d') - timedelta(days=1),
                 datetime.strptime(pt.end, '%Y-%m-%d') + timedelta(days=1))
    ax1.yaxis.set_major_locator(y1_major_locator)
    ax1.set_ylim((10, 45))
    ax1.legend(prop=font, loc='upper left')
    ax1.set_title(name, fontproperties=font)

    ax2 = ax1.twinx()  # 双Y轴
    ax2.scatter(x=df_soc['max'].index, y=df_soc['max'], color='red', label='max')
    ax2.scatter(x=df_soc['min'].index, y=df_soc['min'], color='green', label='min')
    ax2.set_ylabel('SOC', fontproperties=font)
    ax2.yaxis.set_major_locator(y2_major_locator)
    ax2.set_ylim((0, 105))
    ax2.legend(prop=font, loc='upper right')

    plt.show()
    # print(df_soc)


def plot_chg_vol():
    name = 'PCS1_1'
    PCS1_1 = BatteryCluster(name, pt.start, pt.end)
    df_vol = PCS1_1.get_chg_vol()
    df_soc = PCS1_1.get_chg_soc()

    df_vol = df_vol.dropna()
    df_soc = df_soc.dropna()

    fig = plt.figure(figsize=(12, 8))
    y1_major_locator = MultipleLocator(0.1)  # 把x轴的刻度间隔设置为1，并存在变量里
    y2_major_locator = MultipleLocator(10)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax1 = fig.add_subplot(111)
    # ax1.scatter(x=df_vol['max'].index, y=df_vol['max'], marker='v', color='black', label='max')
    # ax1.scatter(x=df_vol['min'].index, y=df_vol['min'], marker='v', color='blue', label='min')
    ax1.plot(df_vol['max'], marker='s')
    ax1.plot(df_vol['min'], marker='s')
    ax1.plot(df_vol['max'], color='cyan', label='max')
    ax1.plot(df_vol['min'], color='blue', label='min')
    ax1.set_ylabel('电压', fontproperties=font)
    ax1.set_xlim(datetime.strptime(pt.start, '%Y-%m-%d') - timedelta(days=1),
                 datetime.strptime(pt.end, '%Y-%m-%d') + timedelta(days=1))
    ax1.yaxis.set_major_locator(y1_major_locator)
    ax1.set_ylim((2.9, 4.3))
    ax1.legend(prop=font, loc='upper left')
    ax1.set_title(name, fontproperties=font)

    ax2 = ax1.twinx()  # 双Y轴
    ax2.scatter(x=df_soc['max'].index, y=df_soc['max'],  color='red', label='max')
    ax2.scatter(x=df_soc['min'].index, y=df_soc['min'], color='green', label='min')
    ax2.set_ylabel('SOC', fontproperties=font)
    ax2.yaxis.set_major_locator(y2_major_locator)
    ax2.set_ylim((0, 105))
    ax2.legend(prop=font, loc='upper right')

    plt.show()


def ana_corr():
    name = 'PCS1_1'
    PCS1_1 = BatteryCluster(name, pt.start, pt.end)
    df_vol = PCS1_1.get_chg_vol()
    df_soc = PCS1_1.get_chg_soc()
    df_chg = PCS1_1.get_chg_detail()

    col = ['SOC', '充电量', '电压']
    df = pd.DataFrame(columns=col)

    # print(df_vol)
    # print(df_soc)
    # print(df_chg)

    df[col[0]] = (df_soc['max'] - df_soc['min']).values
    df[col[1]] = df_chg.values
    df[col[2]] = (df_vol['max'] - df_vol['min']).values
    df = df.apply(lambda x: x.astype(float))
    print(df.info())

    df_corr = df.corr()
    df_corr.to_excel('D:/a.xlsx')

