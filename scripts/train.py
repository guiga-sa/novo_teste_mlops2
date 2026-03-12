import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def train_model():
    # 1. Carregar os dados limpos
    print("Carregando dados processados...")
    df = pd.read_csv('data/processed/weather_cleaned.csv')

    # 2. Separar X (features) e y (alvo)
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']

    # Transformar variáveis categóricas em numéricas (One-Hot Encoding)
    X = pd.get_dummies(X)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treinar o modelo XGBoost
    print("Treinando o modelo...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # 4. Avaliar e salvar métricas
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Acurácia do modelo: {accuracy:.4f}")

    with open('metrics.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)

    # 5. Salvar o modelo treinado
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print("Modelo salvo em models/model.pkl")

if __name__ == "__main__":
    train_model()