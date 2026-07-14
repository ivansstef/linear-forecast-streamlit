# Лінійний прогноз: інтерактивна демонстрація

Навчальний проєкт лінійної регресії зі Streamlit.

## Встановлення та запуск

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Тренування моделі

```powershell
python model.py
```

Виводить MSE і зберігає `linear_regression_model.joblib`, `X.joblib`, `y.joblib`.

### Запуск застосунку

```powershell
streamlit run app.py
```

### Тести

```powershell
pytest tests/ -v
```

## Структура

```
app.py          — Streamlit-застосунок
model.py        — навчання і збереження моделі
requirements.txt
tests/
    test_model.py
    test_app.py
```

## Деплой

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://linear-forecast-streamlit.streamlit.app)
