import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


df = pd.read_csv('Orange_Telecom.csv', usecols=['total_day_calls', 'total_night_calls', 'total_intl_calls'])

print(df.shape)
print(df.describe())
print()

t1 = df['total_day_calls'].values.reshape(-1,1)
t2 = df['total_night_calls'].values.reshape(-1,1)
b = df['total_intl_calls'].values.reshape(-1,1)
T = np.hstack((t1, t2))

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corelație")
plt.show()


def polynomial_features(A, degree):
    m, n = A.shape
    features = [np.ones((m, 1))] 
    
    for d in range(1, degree + 1):
        for i in range(n):
            features.append(A[:, i:i+1] ** d)
            for j in range(i + 1, n):
                features.append((A[:, i:i+1] * A[:, j:j+1]) ** d)
    
    return np.hstack(features)

def manual_pinv(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag(1 / S)
    return np.dot(np.dot(Vt.T, S_inv), U.T)

for degree in range(1, 4):
    start_time = time.time()
    
    T_poly = polynomial_features(T, degree)
    x_pinv = manual_pinv(T_poly)
    theta = np.dot(x_pinv, b)
    
    x_range = np.linspace(t1.min(), t1.max(), 50)
    y_range = np.linspace(t2.min(), t2.max(), 50)
    X, Y = np.meshgrid(x_range, y_range)
    T_grid = np.column_stack((X.ravel(), Y.ravel()))
    T_grid_poly = polynomial_features(T_grid, degree)
    b_prediction_poly = np.dot(T_grid_poly, theta)
    Z = b_prediction_poly.reshape(X.shape)
    
    residuals = b - np.dot(T_poly, theta)
    mse = np.mean(residuals ** 2)
    sse = np.sum(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - (sse / np.sum((b - np.mean(b)) ** 2))
    execution_time = time.time() - start_time
    
    print(f"Pentru regresia polinomială de grad {degree}:")
    print('SSE:', sse)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R^2:', r2)
    print(f'Execution time: {execution_time:.6f} secunde')
    print()
    

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t1, t2, b, color='blue', alpha=0.5, label="Date reale")
    ax.plot_surface(X, Y, Z, color='green', alpha=0.6, label="Plan de regresie")
    ax.set_xlabel("Total Day Calls")
    ax.set_ylabel("Total Night Calls")
    ax.set_zlabel("Total Intl Calls")
    ax.set_title(f"Regresie Polinomială - Grad {degree}")
    ax.legend()
    
    plt.show()
