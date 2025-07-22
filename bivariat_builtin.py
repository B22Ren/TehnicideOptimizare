import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('Orange_Telecom.csv', usecols=['total_day_calls', 'total_night_calls', 'total_intl_calls'])

#print(df.shape)
#print(df.describe())
#print()

t1 = df['total_day_calls'].values.reshape(-1,1)
t2 = df['total_night_calls'].values.reshape(-1,1)
b = df['total_intl_calls'].values.reshape(-1,1)
T = np.hstack((t1, t2))

x_range = np.linspace(t1.min(), t1.max(), 50)
y_range = np.linspace(t2.min(), t2.max(), 50)
X, Y = np.meshgrid(x_range, y_range)
T_grid = np.column_stack((X.ravel(), Y.ravel()))

for degree in range(1, 4):
    start_time = time.time()
    
    poly = PolynomialFeatures(degree=degree)
    T_poly = poly.fit_transform(T)
    T_grid_poly = poly.transform(T_grid)

    lin2 = LinearRegression()
    lin2.fit(T_poly, b)
    b_prediction_poly = lin2.predict(T_grid_poly)
    Z = b_prediction_poly.reshape(X.shape)
    
    execution_time = time.time() - start_time
    
    print(f"Pentru regresia polinomială de grad {degree}:")
    print('SSE:', metrics.mean_squared_error(b, lin2.predict(T_poly)) * len(b))
    print('MAE:', metrics.mean_absolute_error(b, lin2.predict(T_poly)))
    print('MSE:', metrics.mean_squared_error(b, lin2.predict(T_poly)))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(b, lin2.predict(T_poly))))
    print('R^2:', metrics.r2_score(b, lin2.predict(T_poly)))
    print(f'Timp de executie: {execution_time:.6f} secunde')
    print()
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t1, t2, b, color='blue', alpha=0.5, label="Date reale")
    ax.plot_surface(X, Y, Z, color='green', alpha=0.6, label="Plan de regresie")
    ax.set_xlabel("Total Day Calls")
    ax.set_ylabel("Total Night Calls")
    ax.set_zlabel("Total Intl Calls")
    ax.set_title(f"Regresie Polinomială - Grad {degree}")
    ax.legend(["Date reale", "Plan de regresie"])
    
    plt.show()
