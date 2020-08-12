import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.common.function import get_url_data, set_index, set_columns
# from common.function import set_combined_efficiency_score, set_loss_efficiency_score, set_station_efficiency_score
from ..config import parameters as pt
from .pcs import EnergyStorageUnit


class Station(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.pcs = pt.battery['PCS']
        self.data = self.get_data()
        self.on_grid_energy = None
        self.on_f_grid_energy = None
        self.on_p_grid_energy = None
        self.on_g_grid_energy = None
        self.off_grid_energy = None
        self.off_f_grid_energy = None
        self.off_p_grid_energy = None
        self.off_g_grid_energy = None
        self.dc_chg = None
        self.dc_dis = None
        self.ac_chg = None
        self.ac_dis = None

    def get_data(self):
        ids = [["accumulatorinput", "南通通州积美$南通通州积美$储能电表反向有功总"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表反向有功峰"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表反向有功谷"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表反向有功平"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表正向有功总"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表正向有功峰"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表正向有功谷"],
               ["accumulatorinput", "南通通州积美$南通通州积美$储能电表正向有功平"]]

        df = get_url_data(pt.url, ids, pt.func, self.start, self.end, pt.interval)
        df.index = set_index(df)
        df.columns = set_columns(df, pt.prefix)

        return df.dropna()

    def get_on_grid_energy(self):
        """
        上网总电量
        :return:
        """
        if self.on_grid_energy:
            return self.on_grid_energy
        else:
            df_data = self.data['储能电表反向有功总']
            sum_ = df_data[-1] - df_data[0]
            self.on_grid_energy = round(sum_, 2)
            return self.on_grid_energy

    def get_on_f_grid_energy(self):
        """
        上网峰段电量
        :return:
        """
        if self.on_f_grid_energy:
            return self.on_f_grid_energy
        else:
            df_data = self.data['储能电表反向有功峰']
            sum_ = df_data[-1] - df_data[0]
            self.on_f_grid_energy = round(sum_, 2)
            return self.on_f_grid_energy

    def get_on_g_grid_energy(self):
        """
        上网谷电量
        :return:
        """
        if self.on_g_grid_energy:
            return self.on_g_grid_energy
        else:
            df_data = self.data['储能电表反向有功谷']
            sum_ = df_data[-1] - df_data[0]
            self.on_g_grid_energy = round(sum_, 2)
            return self.on_g_grid_energy

    def get_on_p_grid_energy(self):
        """
        上网平段电量
        :return:
        """
        if self.on_p_grid_energy:
            return self.on_p_grid_energy
        else:
            df_data = self.data['储能电表反向有功平']
            sum_ = df_data[-1] - df_data[0]
            self.on_p_grid_energy = round(sum_, 2)
            return self.on_p_grid_energy

    def get_off_grid_energy(self):
        """
        下网电量
        :return:
        """
        if self.off_grid_energy:
            return self.off_grid_energy
        else:
            df_data = self.data['储能电表正向有功总']
            sum_ = df_data[-1] - df_data[0]
            self.off_grid_energy = round(sum_, 2)
            return self.off_grid_energy

    def get_off_f_grid_energy(self):
        """
        下网峰段电量
        :return:
        """
        if self.off_f_grid_energy:
            return self.off_f_grid_energy
        else:
            df_data = self.data['储能电表正向有功峰']
            sum_ = df_data[-1] - df_data[0]
            self.off_f_grid_energy = round(sum_, 2)
            return self.off_f_grid_energy

    def get_off_g_grid_energy(self):
        """
        下网谷段电量
        :return:
        """
        if self.off_g_grid_energy:
            return self.off_g_grid_energy
        else:
            df_data = self.data['储能电表正向有功谷']
            sum_ = df_data[-1] - df_data[0]
            self.off_g_grid_energy = round(sum_, 2)
            return self.off_g_grid_energy

    def get_off_p_grid_energy(self):
        """
        下网平段电量
        :return:
        """
        if self.off_p_grid_energy:
            return self.off_p_grid_energy
        else:
            df_data = self.data['储能电表正向有功平']
            sum_ = df_data[-1] - df_data[0]
            self.off_p_grid_energy = round(sum_, 2)
            return self.off_p_grid_energy

    def get_combined_efficiency(self):
        """
        电站综合效率
        :return:
        """
        on_grid_energy = self.get_on_grid_energy()
        off_grid_energy = self.get_off_grid_energy()
        if off_grid_energy == 0:
            efficiency = 0
        else:
            efficiency = on_grid_energy / off_grid_energy * 100

        return round(efficiency, 2)

    def get_ac_chg(self):
        """
        该电站的所有PCS交流侧充电量之和
        :return:
        """
        if self.ac_chg:
            return self.ac_chg
        else:
            names = ['PCS' + str(i + 1) for i in range(self.pcs)]
            sum_ = 0.0
            for name in names:
                pcs = EnergyStorageUnit(name, self.start, self.end)
                value = pcs.get_ac_chg().sum()
                sum_ = sum_ + value

            self.ac_chg = round(sum_, 2)
            return self.ac_chg

    def get_ac_dis(self):
        """
        该电站的所有PCS交流侧放电量之和
        :return:
        """
        if self.ac_dis:
            return self.ac_dis
        else:
            names = ['PCS' + str(i + 1) for i in range(self.pcs)]
            sum_ = 0.0
            for name in names:
                pcs = EnergyStorageUnit(name, self.start, self.end)
                value = pcs.get_ac_dis().sum()
                sum_ = sum_ + value

            self.ac_dis = round(sum_, 2)
            return self.ac_dis

    def get_dc_chg(self):
        """
        该电站PCS直流侧充电电量
        :return:
        """
        if self.dc_chg:
            return self.dc_chg
        else:
            names = ['PCS' + str(i + 1) for i in range(self.pcs)]
            sum_ = 0.0
            for name in names:
                pcs = EnergyStorageUnit(name, self.start, self.end)
                sum_ = sum_ + pcs.get_dc_chg().sum()

            self.dc_chg = round(sum_, 2)
            return self.dc_chg

    def get_dc_dis(self):
        """
        该电站PCS直流侧放电电量
        :return:
        """
        if self.dc_dis:
            return self.dc_dis
        else:
            names = ['PCS' + str(i + 1) for i in range(self.pcs)]
            sum_ = 0.0
            for name in names:
                pcs = EnergyStorageUnit(name, self.start, self.end)
                sum_ = sum_ + pcs.get_dc_dis().sum()

            self.dc_dis = round(sum_, 2)
            return self.dc_dis

    def get_dis_efficiency(self):
        """
        该电站实际可放点量：直流侧放电量总和/额定装机能量
        :return:
        """
        period = self.get_period_hours() / 24
        discharge = self.get_dc_dis()

        efficiency = discharge / (pt.rated_energy*period) * 100

        return round(efficiency, 2)

    def get_loss_efficiency(self):
        """
        电站储能损耗率
        :return:
        """
        loss = self.get_ac_chg() - self.get_ac_dis()
        if self.get_off_grid_energy() == 0:
            efficiency = 0
        else:
            efficiency = loss / self.get_off_grid_energy() * 100

        return round(efficiency, 2)

    def get_self_energy(self):
        """
        站用电量
        :return:
        """
        off_grid_energy = self.get_off_grid_energy()
        on_grid_energy = self.get_on_grid_energy()
        ac_chg = self.get_ac_chg()
        ac_dis = self.get_ac_dis()

        self_1 = off_grid_energy - ac_chg
        self_2 = ac_dis - on_grid_energy

        self_energy = self_1 + self_2
        return round(self_energy, 2)

    def get_self_efficiency(self):
        """
        站用电率
        :return:
        """
        self_energy = self.get_self_energy()
        off_energy = self.get_off_grid_energy()

        if off_energy == 0:
            efficiency = 0
        else:
            efficiency = self_energy / off_energy * 100
        return round(efficiency, 2)

    def get_ac_efficiency(self):
        """
        该电站PCS交流侧充放电效率
        :return:
        """
        if self.get_ac_chg() == 0:
            efficiency = 0
        else:
            efficiency = self.get_ac_dis() / self.get_ac_chg() * 100
        return round(efficiency, 2)

    def get_dc_efficiency(self):
        """
        该电站PCS直流侧充放电效率
        :return:
        """
        chg = self.get_dc_chg()
        dis = self.get_dc_dis()
        if chg == 0:
            efficiency = 0
        else:
            efficiency = dis / chg * 100
        return round(efficiency, 2)

    def get_work_hours(self):
        """
        获取电站运行小时数
        :return:
        """
        names = ['PCS' + str(i + 1) for i in range(self.pcs)]
        period_all = []
        for name in names:
            pcs = EnergyStorageUnit(name, self.start, self.end)
            period = pcs.get_work_hour()
            period_all.append(period)

        hours = float(np.mean(period_all))
        return round(hours, 1)

    def get_work_days(self):
        """
        获取电站运行天数
        :return:
        """
        df_on_energy = self.get_on_grid_energy()
        df_data = self.data['储能电表反向有功总']
        df_detail = pd.Series()
        for d, df in df_data.groupby(df_data.index.day):
            df_detail.loc[len(df_detail)] = df[-1] - df[0]
        df_selected = df_detail[df_detail > 500]
        if df_selected.empty:
            days = 0
        else:
            days = int(df_on_energy / np.mean(df_selected))

        return days

    def get_period_hours(self):
        """
        获取评价周期小时数
        :return:
        """
        start = self.start
        end = self.end
        if not isinstance(self.start, datetime):
            start = pd.to_datetime(self.start)
        if not isinstance(self.end, datetime):
            end = pd.to_datetime(self.end)

        period = (end - start).days * 24

        return period

    def get_coff_equa_use(self):
        """
        获取电站等效利用系数：直流侧充放电量之和/(额定功率*统计时间小时数)
        :return:
        """
        period = self.get_period_hours()
        names = ['PCS' + str(i + 1) for i in range(self.pcs)]
        coff_all = []
        for name in names:
            pcs = EnergyStorageUnit(name, self.start, self.end)
            sum_dis = pcs.get_dc_chg().sum() + pcs.get_dc_dis().sum()
            coff = sum_dis / (pt.rated_power*period)
            coff_all.append(coff)

        efficiency = np.mean(coff_all) * 100
        return round(efficiency, 2)

    def get_coff_use(self):
        """
        获取电站可用系数：实际使用时间小时数/统计时间小时数
        :return:
        """
        hours = self.get_period_hours()
        coff = self.get_work_hours()

        efficiency = coff / hours * 100
        return round(efficiency, 2)

    def get_earnings(self):
        '''
        计算电站经济收益
        :return:
        '''
        f_dis = self.get_on_f_grid_energy()
        p_dis = self.get_on_p_grid_energy()
        g_dis = self.get_on_g_grid_energy()

        f_chg = self.get_off_f_grid_energy()
        p_chg = self.get_off_p_grid_energy()
        g_chg = self.get_off_g_grid_energy()

        earnings = pt.f_price * (f_dis - f_chg) + pt.g_price * (g_dis - g_chg) + \
                   pt.p_dis_price * p_dis - pt.p_chg_price * p_chg

        return round(earnings, 2)

    def output(self):
        df = pd.DataFrame(columns=['值'], index=['下网电量', 'pcs充电量', '直流侧充电量',
                                                 '直流侧放电量', 'pcs放电量', '上网电量',
                                                 '综合效率', 'pcs效率', '直流侧效率'])
        df.loc['下网电量', '值'] = self.get_off_grid_energy()
        df.loc['pcs充电量', '值'] = self.get_ac_chg()
        df.loc['直流侧充电量', '值'] = self.get_dc_chg()
        df.loc['直流侧放电量', '值'] = self.get_dc_dis()
        df.loc['pcs放电量', '值'] = self.get_ac_dis()
        df.loc['上网电量', '值'] = self.get_on_grid_energy()
        df.loc['综合效率', '值'] = self.get_combined_efficiency()
        df.loc['pcs效率', '值'] = self.get_ac_efficiency()
        df.loc['直流侧效率', '值'] = self.get_dc_efficiency()

        df.to_excel(os.path.join(os.getcwd(), 'excel', '充放电数据.xlsx'))





    def run(self):
        df = self.get_on_grid_energy()
        print(df)


