
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import pandas as pd

TOURNEY_DATA_M = pd.read_csv("tourney_data_m.csv")

def generate_predictions(col_list):
    return len(col_list)


def get_model(stats):
    test_year = int(stats[0])
    # metrics = ['Seed', 'FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']
    metrics = stats[1:]
    diff_metrics = [metric + 'DiffAdj' for metric in metrics]

    # weights = {'Seed': 5.0, 'FGM': 1.0, 'FGA': 0.5}
    # weights

    # for metric in metrics:
    #     TOURNEY_DATA_M[metric + 'DiffAdj'] *= weights[metric]

    X = TOURNEY_DATA_M[TOURNEY_DATA_M['Season'] != test_year][diff_metrics]
    y = TOURNEY_DATA_M[TOURNEY_DATA_M['Season'] != test_year]['ScoreDiffAdj']

    model_m_2019 = LinearRegression()
    model_m_2019.fit(X, y)

    # print('Score diff =', round(model_m_2019.intercept_, 3), '+')
    # for i in range(len(metrics)):
    #     print("%+.3f" % round(model_m_2019.coef_[i], 3), '*', metrics[i])

    return [round(item, 2) for item in model_m_2019.coef_]