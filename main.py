import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
import numpy as np

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv("data/dataset.csv")

# Calculer le pourcentage de valeurs manquantes dans chaque colonne
missing_percentages = df.isnull().mean() * 100

# Identifier les colonnes avec des valeurs manquantes dépassant 5%
high_missing_columns = missing_percentages[missing_percentages > 5].index

# Supprimer les lignes avec des valeurs manquantes dépassant 5%
df = df.dropna(subset=high_missing_columns)

# Gérer les valeurs manquantes restantes (ne dépassant pas 5%) en utilisant l'imputation par mode
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Exporter le DataFrame dans un nouveau fichier CSV avec les valeurs imputées
df.to_csv("imputed_datadelivery.csv", index=False)
di = pd.read_csv("imputed_datadelivery.csv")
print(di.isna().values.any())

# Sélectionner les colonnes catégorielles dans le DataFrame
categorical_cols = di.columns[di.dtypes==object].tolist()

# Instancier LabelEncoder
encoder = LabelEncoder()

# Colonnes à encoder
columns_to_encode = ['created_at', 'actual_delivery_time', 'store_id', 'store_primary_category']

# Appliquer l'encodage à toutes les colonnes spécifiées dans le DataFrame di
di[columns_to_encode] = di[columns_to_encode].apply(lambda col: encoder.fit_transform(col.astype(str)))

print(di.head(5))

# Normaliser les fonctionnalités
X = di.drop('actual_delivery_time', axis=1)
y = di[['actual_delivery_time']]

sc = MinMaxScaler()
X_sc = sc.fit_transform(X)
X = pd.DataFrame(X_sc, columns=X.columns)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print(X_train.shape)
print(X_test.shape)

# Instancier le modèle de régression GradientBoosting
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
gbm = GradientBoostingRegressor(random_state=10)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
print("Mean Squared Error:", mse)

# Instancier le modèle SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R² Score:", r2)
print("Mean Squared Error:", mse)

# Instancier le modèle Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Instancier le modèle AdaBoost
adaboost = AdaBoostRegressor(n_estimators=50, random_state=0)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Instancier le modèle Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Instancier le modèle KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Instancier le modèle ElasticNet
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Afficher les métriques de performance supplémentaires
print('MSE = ',mean_squared_error(y_test, y_pred))
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('MAE = ', mean_absolute_error(y_test, y_pred))
print('R2 = ',r2_score(y_test, y_pred))
