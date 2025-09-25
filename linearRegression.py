import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('Housing.csv')

binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))


furnishing_dummies = pd.get_dummies(df['furnishingstatus'], drop_first=True, prefix='furnishing')
df = pd.concat([df, furnishing_dummies], axis=1)


y = df['price']


X_simple = df[['area']]


X_multi = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
              'mainroad', 'guestroom', 'basement', 'hotwaterheating',
              'airconditioning', 'prefarea',
              'furnishing_semi-furnished', 'furnishing_unfurnished']]


X_train_multi, X_test_multi, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.3, random_state=42)

X_train_simple = X_train_multi[['area']]
X_test_simple = X_test_multi[['area']]


slr_model = LinearRegression()
slr_model.fit(X_train_simple, y_train)
y_pred_slr = slr_model.predict(X_test_simple)


mlr_model = LinearRegression()
mlr_model.fit(X_train_multi, y_train)
y_pred_mlr = mlr_model.predict(X_test_multi)




def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")

evaluate_model(y_test, y_pred_slr, "Simple Linear Regression (Area)")
evaluate_model(y_test, y_pred_mlr, "Multiple Linear Regression (All Features)")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_simple['area'], y=y_test, color='blue', label='Actual Price')
sns.lineplot(x=X_test_simple['area'], y=y_pred_slr, color='red', label='Regression Line')
plt.title('Simple Linear Regression: Price vs. Area (Test Set)')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (INR)')
plt.grid(True)
plt.legend()
plt.show()


print("\n--- Multiple Linear Regression Model Coefficients ---")
coefficients = pd.Series(mlr_model.coef_, index=X_train_multi.columns)
intercept = mlr_model.intercept_

print(f"Intercept: ${intercept:,.2f}")
print("\nFeature Coefficients:")
print(coefficients.sort_values(ascending=False).apply(lambda x: f'${x:,.2f}'))