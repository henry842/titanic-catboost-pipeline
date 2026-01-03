# Titanic CatBoost Pipeline

Pipeline completo para competição Titanic no Kaggle, com foco em evitar vazamento de dados e maximizar acurácia de validação.

---

## 🔹 Descrição

Este projeto implementa um modelo robusto para prever sobrevivência no Titanic utilizando CatBoost, incluindo:

- Feature engineering: títulos, sobrenomes, cabine, tickets, família
- Target Encoding seguro de variáveis de alta cardinalidade (K-Fold)
- Validação cruzada estratificada
- Threshold tuning para OOF
- Pseudo-labeling opcional (uma rodada segura)
- Winsorização suave e imputação inteligente de valores ausentes

O pipeline foi construído para ser **reprodutível** e pronto para submissão no Kaggle.

---

## 🔹 Estrutura do Notebook

1️⃣ **Imports e Configurações Globais**  
2️⃣ **Funções utilitárias**  
3️⃣ **Preprocessamento de dados**  
4️⃣ **K-Fold Target Encoding**  
5️⃣ **Treinamento CatBoost com CV**  
6️⃣ **Threshold tuning OOF**  
7️⃣ **Pseudo-labeling opcional**  
8️⃣ **Submission para Kaggle**  

---

## 🔹 Como usar

1. Clonar o repositório:
```bash
git clone https://github.com/SEU_USUARIO/titanic-catboost-pipeline.git
cd titanic-catboost-pipeline
