import pandas as pd
import numpy as np
import os

def preprocess_data():
    # Caminhos dos arquivos
    input_path = 'data/raw/weatherAUS.csv'
    output_dir = 'data/processed'
    output_path = os.path.join(output_dir, 'weather_cleaned.csv')

    # 1. Carregar os dados
    print(f"Lendo dados de: {input_path}...")
    df = pd.read_csv(input_path)

    # 2. Limpeza Básica: Remover colunas com excesso de nulos (> 40%)
    # Comum nesse dataset para colunas como 'Evaporation' e 'Sunshine'
    limit = len(df) * 0.6
    df = df.dropna(thresh=limit, axis=1)

    # 3. Remover linhas onde a variável alvo (RainTomorrow) é nula
    df = df.dropna(subset=['RainTomorrow'])

    # 4. Feature Engineering simples: Converter data e extrair o mês
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df.drop('Date', axis=1)

    # 5. Tratamento de Nulos (Imputação Simples)
    # Numéricas: preencher com a mediana / Categóricas: preencher com o mais frequente (moda)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Transformar Alvo (RainTomorrow) em binário (0 e 1)
    df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

    # 7. Salvar o resultado
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False)
    print(f"Dados processados salvos com sucesso em: {output_path}")
    print(f"ALERTA: O novo shape final da Austrália é: {df.shape}")

if __name__ == "__main__":
    preprocess_data()