import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


file_path = "Orange_Telecom.csv"  
df = pd.read_csv(file_path)
df.drop(['phone_number', 'area_code', 'state', 'intl_plan', 'voice_mail_plan', 'churned'], axis='columns', inplace=True)


def solve_builtin_fixed(predictor, target):
    A = np.array(df[predictor]).reshape(-1, 1)
    b = np.array(df[target]).reshape(-1, 1)

  
    sorted_indices = np.argsort(A, axis=0).flatten()
    A_sorted = A[sorted_indices]
    b_sorted = b[sorted_indices]

    for i in range(1, 9): 
        start_time = time.time()
        poly = PolynomialFeatures(degree=i, include_bias=True)
        A_poly = poly.fit_transform(A_sorted)

     
        model = LinearRegression()
        model.fit(A_poly, b_sorted)
        pred = model.predict(A_poly)

        mse = mean_squared_error(b_sorted, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(b_sorted, pred)
        sse = np.sum(np.square(pred - b_sorted))
        r2 = r2_score(b_sorted, pred)

        print(f"Gradul {i}:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"SSE: {sse}")
        print(f"R2: {r2}")
        print(f"timp de executie: {time.time() - start_time}")

        
        plt.scatter(A_sorted, b_sorted, alpha=0.5, color='red', label='Date')  
        plt.plot(A_sorted, pred, color='blue', linewidth=2, label='linia de regresie')
        
        plt.xlabel(predictor)
        plt.ylabel(target)
        plt.legend()
        plt.title(f'regresie polinomiala (Gradul{i}) - builtin')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


solve_builtin_fixed('total_day_calls', 'total_intl_calls')
