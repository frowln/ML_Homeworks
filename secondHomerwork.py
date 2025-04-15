import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def main():
    # 1. Загрузка и предобработка данных
    df = pd.read_csv('AmesHousing.csv').drop(columns=['Order'])
    numeric_df = df.select_dtypes(include=['number'])

    # Удаление высококоррелированных признаков
    corr_matrix = numeric_df.drop(columns=['SalePrice']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
    X = numeric_df.drop(columns=to_drop + ['SalePrice'])
    y = numeric_df['SalePrice']

    # Обработка пропусков
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # 2. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Визуализация с PCA (только тренировочные данные)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_train_pca[:, 0], 
        X_train_pca[:, 1], 
        y_train, 
        c=y_train, 
        cmap='plasma',
        alpha=0.6
    )
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('SalePrice')
    plt.title('3D визуализация тренировочных данных')
    plt.colorbar(scatter, label='SalePrice')
    plt.show()

    # 5. Обучение моделей
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(max_iter=10000)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name} RMSE: {rmse:.2f}")

    # 6. Подбор оптимального alpha для Lasso
    alphas = np.logspace(-4, 2, 100)
    rmse_values = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)
        rmse_values.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, rmse_values)
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('RMSE')
    plt.title('Зависимость ошибки от коэффициента регуляризации')
    plt.grid(True)
    plt.show()

    # 7. Определение важнейшего признака
    best_alpha = alphas[np.argmin(rmse_values)]
    final_lasso = Lasso(alpha=best_alpha, max_iter=10000)
    final_lasso.fit(X_train_scaled, y_train)

    feature_importance = pd.Series(
        np.abs(final_lasso.coef_), 
        index=X.columns
    )
    top_feature = feature_importance.idxmax()
    
    print(f"\nРезультаты анализа Lasso:")
    print(f"Оптимальный alpha: {best_alpha:.5f}")
    print(f"Самый важный признак: {top_feature}")
    print(f"Коэффициент: {final_lasso.coef_[feature_importance.argmax()]:.4f}")

if __name__ == '__main__':
    main()