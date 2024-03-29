# AlexeyALV == Lebedev A.N.

# Построение модели для прогноза индекса цен на товары и услуги
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from flask import Flask, render_template
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Загрузить статистику
iMon = datetime.now().month - 1 if datetime.now().month > 1 else 12
sMon = str(iMon) if datetime.now().month > 9 else "0" + str(iMon)
iYer = datetime.now().year if datetime.now().month > 1 else datetime.now().year - 1
sYer = str(iYer)
sLink = "https://rosstat.gov.ru/storage/mediabank/ipc_mes_" + sMon + "-" + sYer + ".xlsx"

# Прочитать файл excel
lData = pd.read_excel(sLink, "01")

lFullData = []
y_m = 4  # начальная строка
x_m = 1  # начальный столбец

while x_m < iYer - 1989:
    while y_m < 16:
        if x_m == iYer - 1990 and y_m == 4 + iMon:  # Для последнего года берем только до последнего отчетного месяца
            break
        lFullData.append(lData.iloc[y_m,x_m])
        y_m += 1
    y_m = 4
    x_m += 1

# Очистка данных
# Убираем значения за пределами 3 СКО
df = pd.DataFrame(lFullData)
z_scores = np.abs((df - df.mean()) / df.std())
df = df[z_scores < 3]
# Восстанавливаем пропущенные значения по среднему соседних
df.fillna(df.mean(), inplace=True)

# Проверка данных на нормальное распределение по Харке—Бера
jb_test = sm.stats.stattools.jarque_bera(df)

# Обучение модели ARIMA
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Прогноз на основе обученной модели
forecast = model_fit.forecast(steps=6)

# Расчет СКО и средней абсолютной ошибки
mse = mean_squared_error(df[-6:], forecast)
mae = mean_absolute_error(df[-6:], forecast)

# Присоединение прогноза к исходному DataFrame
df.append(forecast)

# Визуализация исходных данных и прогноза и
# сохранение в файл для отображения на странице
plt.plot(df.index[:-6], df[:-6], label='Исходные данные')
plt.plot(df.index[-6:], df[-6:], label='Прогноз')
plt.title('Статистика и прогноз на 6 месяцев')
plt.xlabel('Месяцы')
plt.ylabel('Индекс')
plt.legend()
plt.grid(True)
my_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(my_path, "static/diag.png"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", data_source = sLink,
                           data_list = lFullData, cleared_data = df, end_year = sYer,
                           norm = ("да" if jb_test[1] < 0.05 else "нет"),
                           quality = "MSE: " + str(mse) + "   MAE: " + str(mae))

if __name__ == '__main__':
    app.run(debug=True)
