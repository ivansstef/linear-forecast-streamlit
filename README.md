# Лінійний прогноз: інтерактивна демонстрація

Навчальний проєкт курсу **EmpowerU AI Fundamentals**.
Демонструє повний цикл лінійної регресії — від генерації даних до інтерактивного прогнозу у вебзастосунку.

## Що демонструє проєкт

Дані → навчання → оцінювання → збереження → прогноз → візуалізація

Синтетичний набір даних (100 зразків, 1 ознака) генерується за допомогою `sklearn`, ознака X масштабується до `[-3, 3]`, модель `LinearRegression` навчається, оцінюється через MSE та зберігається. Streamlit-застосунок завантажує модель і дозволяє отримати прогноз для будь-якого X у діапазоні `[-3, 3]`.

## Основні функції

- `model.py` — навчання, оцінювання та серіалізація моделі
- `app.py` — Streamlit-інтерфейс із повзунком, прогнозом і порівняльним графіком

## Структура репозиторію

```
app.py                          Streamlit-застосунок
model.py                        Навчання та збереження моделі
requirements.txt                Залежності
linear_regression_model.joblib  Навчена модель
X.joblib                        Початкові значення ознаки
y.joblib                        Початкові цільові значення
tests/
    test_app.py
    test_model.py
```

## Встановлення та запуск

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Навчання моделі

```bash
python model.py
```

Виводить MSE у термінал і зберігає три файли: `linear_regression_model.joblib`, `X.joblib`, `y.joblib`.

Відтворюваний результат: **Mean Squared Error ≈ 416.81**

### Запуск застосунку

```bash
streamlit run app.py
```

### Тести

```bash
pytest tests/ -v
```

## Файли моделі

`linear_regression_model.joblib`, `X.joblib` та `y.joblib` закомічені безпосередньо в репозиторій для відтворюваного деплою.
**Завантажуйте joblib/pickle-артефакти лише з довірених джерел.**

## Посилання

- Застосунок: [https://finaltest.savytskyy.com.ua](https://finaltest.savytskyy.com.ua)
- Репозиторій: [https://github.com/ivansstef/linear-forecast-streamlit](https://github.com/ivansstef/linear-forecast-streamlit)
