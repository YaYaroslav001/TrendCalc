# Данный скрипт строит графики трендовых составляющих для нескольких
# точек наблюдения, выводит результат декомпозиции в файл,
# строит матрицу корреляции и группирует метеостанции.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.cluster.hierarchy as spc
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters

# Регистрация форматеров и конвертеров pandas в matplotlib.
register_matplotlib_converters()

# Сохранение значения стандартного вывода.
original_stdout = sys.stdout

# Создание папки для хранения результата работы скрипта.
if not os.path.exists("Result"):
    os.makedirs("Result")

# Удаление файлов с результатами декомпозиции, если они существуют.
if os.path.exists("Result/Temperature_Trend.txt"):
    os.remove("Result/Temperature_Trend.txt")
if os.path.exists("Result/Temperature_Trend_Groups.txt"):
    os.remove("Result/Temperature_Trend_Groups.txt")

# Стилизация графиков.
sns.set_style("darkgrid")
plt.rc("figure", figsize=(12, 9))
plt.rc("font", size=13)
plt.rc("lines", markersize=5)
plt.rc("lines", linewidth=3)
plt.xlabel("Дата")
plt.ylabel("Температура")
plt.title("График трендов температур")

data_file_str = ""

# Ввод данных.
while data_file_str == "":
    data_file_str = input("Введите один или несколько датасетов через пробел: ")
data_file_list = data_file_str.split()

# Считывание и обработка данных.
print("Считывание и обработка данных.")
data = pd.DataFrame(columns=range(9))

# Считывание датасетов.
for data_file in data_file_list:
    try:
        data_temp = pd.read_csv(data_file, dtype=object, sep=";", header=None)
    except FileNotFoundError:
        print("Файл " + data_file + " не обнаружен!")
        exit()
    data = pd.concat([data, data_temp])

# Присваивание имён столбцам датафрейма.
try:
    data.columns = ["index", "year", "month", "day",
                    "temp_quality", "temp_min", "temp_avg",
                    "temp_max", "precipitation"]
except ValueError:
    print("Количество столбцов одного из датасетов не равно 9!")
    exit()

# Заполнение датафрейма.
df = pd.DataFrame({'year': data["year"],
                   'month': data["month"],
                   'day': data["day"]})
df["date"] = pd.to_datetime(df)
df["temp_avg"] = pd.to_numeric(data["temp_avg"],
                               errors='coerce')
df["precipitation"] = pd.to_numeric(data["precipitation"],
                                    errors='coerce')
df["index"] = data["index"]
df['temp_avg'] = df['temp_avg'].interpolate(method='nearest')
df['precipitation'] = df['precipitation'].interpolate(method='nearest')
df.replace(r'^\s*$', np.nan, regex=True)
grouped = df.groupby('index')   # Группировка датафрейма по индексу метеостанции.
index_list = df['index'].unique().tolist()  # Сохранение всех индексов метеостанций в список.
df_cor = pd.DataFrame()     # Датафрейм, для которого будет считаться корреляция.
df_cor["date"] = pd.date_range("1950-1-1", "2020-12-31", freq="D")

for x in index_list:
    elem = grouped.get_group(x)
    dec = pd.DataFrame(columns=range(2))
    dec.columns = ['date', 'trend']
    dec["date"] = elem["date"]
    dec["trend"] = seasonal_decompose(elem["temp_avg"], model='additive', period=365).trend
    if elem["date"].iloc[0] != datetime(1950, 1, 1):    # Если станции не хватает данных до 1950-го года, добавляются пустые строки.
        missing = pd.date_range("1950-1-1", elem["date"].iloc[0] - timedelta(days=1), freq="D")
        missing_df = pd.DataFrame(columns=range(2))
        missing_df.columns = ['date', 'trend']
        missing_df["date"] = missing
        dec = pd.concat([missing_df, dec])
    plt.xlim([datetime(2000, 1, 1), datetime(2010, 1, 1)])
    plt.plot(dec["date"], dec["trend"], label=x)
    df_cor[x] = pd.Series(dec["trend"].to_numpy())
    a = {'index': x,
         'date': dec["date"],
         'trend': dec["trend"]
         }
    b = pd.DataFrame(a)
    b.to_csv("Result/Temperature_Trend.txt", header=None,
             index=None, sep=';', mode='a')

plt.legend()
plt.savefig('Result/Temperature_Trends_Decomposition.png')
plt.clf()

# Построение матрицы корреляции.
corr = df_cor.corr(method='spearman')
plot = sns.heatmap(corr,
                   xticklabels=corr.columns.values,
                   yticklabels=corr.columns.values, annot=True, fmt='.2f', annot_kws={"fontsize": 11})
plt.title("Матрица корреляции")
fig = plot.get_figure()
fig.savefig("Result/Temperature_Trends_Correlation.png")

# Кластеризация метеостанций на основе корреляции.
pdist = spc.distance.pdist(corr.values)
linkage = spc.linkage(pdist, method='complete')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
cluster = pd.DataFrame(columns=range(2))
cluster.columns = ["group", "index"]
cluster["group"] = idx
cluster["index"] = index_list
idx_list = list(set(idx))
cluster_grouped = cluster.groupby('group')
for x in idx_list:
    with open("Result/Temperature_Trend_Groups.txt", "a") as myfile:
        myfile.write("Group " + str(x) + ":\n")
    cluster_grouped.get_group(x)["index"].to_csv("Result/Temperature_Trend_Groups.txt", header=None,
                                                 index=None, sep=';', mode='a')

print("Результат сохранён в папку Result.")