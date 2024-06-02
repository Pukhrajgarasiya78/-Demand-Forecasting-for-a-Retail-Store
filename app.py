from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)


def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'Date' not in df.columns:
        raise KeyError("Date column not found in the DataFrame.")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    daily_sales = df.groupby(['Date', 'Product Category'])['Quantity'].sum().unstack().fillna(0)
    daily_sales = daily_sales.asfreq('D')
    daily_sales = daily_sales.interpolate(method='time')
    return daily_sales



def forecast_sales(daily_sales):
    train = daily_sales['Electronics'][:'2023-12-31']
    test = daily_sales['Electronics']['2024-01-01':]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))
    forecast = pd.Series(forecast, index=test.index)

    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    rmse = mse ** 0.5
    return forecast, mae, mse, rmse

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_result = None
    mae = None
    mse = None
    rmse = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            daily_sales = load_data(file)
            forecast_result, mae, mse, rmse = forecast_sales(daily_sales)
    if forecast_result is not None:
        forecast_dict = forecast_result.to_dict()
    else:
        forecast_dict = None

    return render_template('index.html', forecast_result=forecast_dict, mae=mae, mse=mse, rmse=rmse)

if __name__ == '__main__':
    app.run(debug=True)
