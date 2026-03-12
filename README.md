# ⛈️ Machine Learning Pipeline: Previsão de Chuva na Austrália

Este repositório contém uma esteira completa de **MLOps** desenvolvida para o meu TCC. O objetivo é prever a ocorrência de chuva no dia seguinte na Austrália utilizando o dataset `weatherAUS`.

O projeto foca em **reprodutibilidade**, utilizando ferramentas de versionamento de dados e automação de experimentos.

---

## 🛠️ Arquitetura do Projeto

A infraestrutura foi desenhada para separar o código (GitHub) dos dados pesados (DagsHub), garantindo um repositório leve e organizado.

* **Linguagem:** Python 3.x
* **Modelo:** XGBoost Classifier
* **Versionamento de Dados:** DVC (Data Version Control)
* **Storage Remoto:** DagsHub (S3-Compatible)
* **CI/CD & Relatórios:** GitHub Actions + CML (Continuous Machine Learning)

---

## 🚀 Como o Projeto Funciona

### 1. Versionamento com DVC
Os dados brutos, processados e o modelo treinado não estão no GitHub. Eles são gerenciados pelo **DVC**. O GitHub armazena apenas os arquivos `.dvc` e o `dvc.lock`, que funcionam como "ponteiros" para os arquivos reais armazenados no DagsHub.

### 2. Pipeline de Dados
O projeto é dividido em estágios definidos no `dvc.yaml`:
1.  **Limpeza (Preprocess):** Trata valores nulos, codifica variáveis e prepara o CSV final.
2.  **Treino (Train):** Treina o XGBoost, salva o modelo (`.pkl`), gera métricas (`.json`) e uma matriz de confusão (`.png`).

### 3. Automação (CML)
A cada `git push`, o GitHub Actions:
* Instala as dependências.
* Conecta ao DagsHub e baixa o dataset bruto.
* Roda a pipeline completa (`dvc repro`).
* **Posta um relatório automático** no comentário do commit com a acurácia e o gráfico de performance do modelo.

---

## 💻 Como Reproduzir Localmente

### Pré-requisitos
* Python instalado.
* DVC instalado (`pip install dvc dvc-s3`).

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/guiga-sa/novo_teste_mlops2.git](https://github.com/guiga-sa/novo_teste_mlops2.git)
    cd novo_teste_mlops2
    ```

2.  **Baixe os dados (Acesso Público):**
    Como o storage no DagsHub é público, você não precisa de credenciais para baixar:
    ```bash
    dvc pull
    ```

3.  **Execute a Pipeline:**
    Para rodar todo o processo de limpeza e treino novamente:
    ```bash
    dvc repro
    ```

> **Nota para o Desenvolvedor:** Se você for o dono do projeto e desejar subir alterações (`dvc push`), lembre-se de configurar suas credenciais locais (`user` e `token`) do DagsHub.

---

## 📊 Performance Atual
* **Acurácia:** ~85.97%
* **Métricas detalhadas:** Podem ser encontradas no arquivo `metrics.json` ou nos relatórios de CI.
