import os
import numpy as np
import pandas as pd
from datetime import timedelta
from app.common.function import get_url_data
from ..config import parameters as pt


def get_max_count(df0):
    df = df0[-12:]
    df1 = df.value_counts()
    df2 = df1.sort_index(ascending=False)
    max_value = df2.index[0]
    return max_value


def cal_chg_number():
    cluster_names = []
    for i in range(pt.battery['PCS']):
        for j in range(pt.battery['modules']):
            cluster_names.append('PCS{}_{}'.format(str(i + 1), str(j + 1)))

    date_range = [d for d in pd.date_range(pt.first_start, pt.last_end, freq='d', closed='left')]
    df_chg_num = pd.DataFrame(columns=cluster_names, index=date_range)
    for name in cluster_names:
        ids = [['analoginput', pt.prefix + name + '#电池组日充电量']]

        modules_data = get_url_data(pt.url, ids, pt.func, pt.first_start, pt.last_end, pt.interval)
        modules_data.columns = set_columns(modules_data)
        modules_data.index = set_index(modules_data)

        dc_chg = pd.Series(index=date_range)

        df_chg = modules_data['{}#电池组日充电量'.format(name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            chg_value = df.max() - df.min()
            dc_chg[d] = chg_value

        dc_chg = dc_chg.fillna(0.0)
        df_rated_chg = get_rated_value()

        # df_num_detail = round((dc_chg / (df_rated_chg[name]*0.9)), 2)
        df_num_detail = dc_chg.apply(lambda x: np.ceil(10 * (x / (df_rated_chg[name] * 0.9))) / 10 \
            if np.ceil(10 * (x / (df_rated_chg[name] * 0.9))) / 10 < 1.0 \
            else 1.0)
        # print(df_num_detail)

        idx = df_num_detail.index

        df_chg_num[name][idx[0]] = df_num_detail[idx[0]]
        for i in range(1, len(idx)):
            df_chg_num[name][idx[i]] = df_chg_num[name][idx[i - 1]] + df_num_detail[i]

        # break
    dir_ = os.path.join(os.getcwd(), 'excel', '累计充放电次数.xlsx')
    df_chg_num.to_excel(dir_)


def get_rated_value():
    path = os.path.join(os.getcwd(), 'excel', '南通积简美居.xlsx')
    df = pd.read_excel(path, index_col=0)
    # print(df)
    names = df[df.columns[0]].apply(lambda x: x.replace('-', '_'))
    names = names.apply(lambda x: 'PCS' + x)

    df_rated_chg = df[df.columns[1]]
    df_rated_chg.index = names

    # print(df_rated_chg)

    return df_rated_chg


def delete_outliers(df):
    """
    去掉df中异常点
    :param df: 以Series形式输入
    :return: 去掉异常点的df
    """
    # df = df.apply(lambda x: x.astype(np.float64))
    df = df.astype(np.float64)

    df_describe = df.describe()
    print(df_describe)
    Q1 = df_describe['25%']
    Q3 = df_describe['75%']
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    df_norm = df[df.apply(lambda x:  lower_limit < x < upper_limit)]

    return df_norm
