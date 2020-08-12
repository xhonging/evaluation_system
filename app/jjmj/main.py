# from jjmj import min_vol, max_vol, vol_diff, plot_chg_soc, plot_chg_vol, ana_corr
# from test import test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_station
# from predictRUL import train_model, test_model
# from common.function import cal_chg_number
# from temp import ana_temp
# from evaluation_system import evaluationSystem
# from .report import generate_report
from .reduce_power import get_reduce_data
from time import *


if __name__ == '__main__':
    start = time()
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_5()
    # test_6()
    # test_7()
    # test_station()
    # plot_chg_vol()
    # plot_chg_soc()
    # ana_corr()
    # train_model()
    # test_model()
    # cal_chg_number()
    get_reduce_data('2020-04-01', '2020-05-01')
    # ana_temp()
    # evaluationSystem()
    # generate_docx()
    end = time()
    print('运行时长：', end - start)
    print('The end!')

