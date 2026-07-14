from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Палітра ──────────────────────────────────────────────────────────────────
_BG        = "#1e2127"   # графітовий фон застосунку
_CARD_BG   = "#272b33"   # картки
_OLIVE     = "#6b7c3a"   # основний акцент
_OLIVE_LT  = "#8fa04a"   # hover / border акцент
_SAND      = "#c9a96e"   # другорядний теплий акцент
_TEXT      = "#e8e8e4"   # основний текст
_MUTED     = "#9a9a8e"   # другорядний текст
_GRID      = "#3a3f4a"   # сітка на графіку

_CSS = f"""
<style>
  /* ── сторінка ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {_BG};
      color: {_TEXT};
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stToolbar"] {{ display: none; }}

  /* ── центральний контейнер ── */
  .main .block-container {{
      max-width: 760px;
      padding: 2rem 1.5rem 3rem;
  }}

  /* ── заголовок ── */
  h1 {{ color: {_TEXT} !important; font-size: 1.55rem !important; font-weight: 600; }}
  .subtitle {{
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {_MUTED};
      margin-top: -0.6rem;
      margin-bottom: 1.2rem;
  }}

  /* ── картки ── */
  .ui-card {{
      background: {_CARD_BG};
      border: 1px solid {_GRID};
      border-radius: 6px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
  }}
  .ui-card p {{ margin: 0; }}

  /* ── слайдер ── */
  [data-testid="stSlider"] > div > div > div > div {{
      background: {_OLIVE} !important;
  }}

  /* ── кнопка ── */
  [data-testid="baseButton-secondary"],
  [data-testid="baseButton-primary"] {{
      background-color: {_OLIVE} !important;
      color: {_TEXT} !important;
      border: 1px solid {_OLIVE_LT} !important;
      border-radius: 4px !important;
      font-weight: 500 !important;
      transition: background 0.15s;
  }}
  [data-testid="baseButton-secondary"]:hover,
  [data-testid="baseButton-primary"]:hover {{
      background-color: {_OLIVE_LT} !important;
  }}

  /* ── результат ── */
  .result-card {{
      background: {_CARD_BG};
      border-left: 3px solid {_OLIVE};
      border-radius: 4px;
      padding: 0.9rem 1.2rem;
      margin-top: 0.8rem;
  }}
  .result-label {{ font-size: 0.78rem; color: {_MUTED}; text-transform: uppercase; letter-spacing: 0.07em; }}
  .result-value {{ font-size: 2rem; font-weight: 700; color: {_SAND}; }}

  /* ── повідомлення ── */
  [data-testid="stAlert"] {{ border-radius: 4px; }}

  /* ── скроллбар ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {_GRID}; border-radius: 3px; }}
</style>
"""

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """
    
    # TODO: your code here
    model = load(filename)
    y = model.predict(X)
    return y

def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature 
    value and making predictions using a pre-trained regression model. The app includes:
    
    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - A "Predict value" button that, when clicked, triggers the prediction process.
    - Upon clicking the "Predict value" button, the function:
        - Calls `load_and_predict`, passing the selected feature as input, to load the regression model 
          and make a prediction.
        - Displays the prediction result on the app.
        - Calls `visualize_difference`, passing the input feature and the prediction result, 
          to visualize the difference between the predicted value and the actual value in the original dataset.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by 
    writing content and rendering UI elements.
    """
    # TODO: your code here

    st.set_page_config(
        page_title="Лінійний прогноз: інтерактивна демонстрація",
        layout="centered",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Streamlit app title
    st.title("Лінійний прогноз: інтерактивна демонстрація")
    st.markdown('<p class="subtitle">Навчальний модуль лінійної регресії</p>',
                unsafe_allow_html=True)

    # Картка введення
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.8rem;color:{_MUTED};text-transform:uppercase;'
                f'letter-spacing:0.07em;margin-bottom:0.5rem">Значення ознаки X</p>',
                unsafe_allow_html=True)
    # User input for new prediction using a slider
    input_feature = st.slider(
        "Значення ознаки X",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.01,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Button to make a prediction
    if st.button("Передбачити значення", use_container_width=True):
        # 1. Call load_and_predict functions.
        # Make sure you convert the input_feature to a matrix before calling load_and_predict, e.g., load_and_predict([[input_feature]])
        try:
            prediction = load_and_predict([[input_feature]])
        except FileNotFoundError:
            st.error("Файл моделі не знайдено. Спочатку запустіть `python model.py`.")
            return

        # 2. Display the prediction.
        pred_scalar = float(np.asarray(prediction).flat[0])
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">Прогноз моделі</div>'
            f'<div class="result-value">{pred_scalar:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 4. Call visualize_difference to display a plot visualizing the difference between actual and perdicted value.
        visualize_difference(input_feature, pred_scalar)

def visualize_difference(input_feature: float, prediction: ArrayLike):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.

    """
    # Load the X and y datasets
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    try:
        X = load(X_filename)
        y = load(y_filename)
    except FileNotFoundError:
        st.error("Файли даних X.joblib або y.joblib не знайдено. Спочатку запустіть `python model.py`.")
        return

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Visualization
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD_BG)

    # Plot the entire dataset (X, y) as grey dots to visualize the data distribution.
    ax.scatter(X, y, color="#5a5f6a", alpha=0.55, s=28, label="Набір даних")

    # Plot the actual target value for a specific input feature as a blue dot.
    closest_x = float(np.asarray(X).flat[_index_of_closest(X, input_feature)])
    ax.scatter(closest_x, actual_target, color="#4da6d6", s=90, zorder=5,
               edgecolors="#7fc4e8", linewidths=0.8, label="Фактичне значення")

    # Plot the predicted target value for the same input feature as a red dot.
    ax.scatter(input_feature, prediction, color="#c97b4b", s=90, zorder=5,
               edgecolors="#e0a070", linewidths=0.8, label="Прогноз")

    # Display a legend on the plot to label the different scatter points (dataset, actual target, predicted target).
    legend = ax.legend(facecolor=_CARD_BG, edgecolor=_GRID,
                       labelcolor=_TEXT, fontsize=8.5)

    # Set the title of the plot, describing what is being visualized.
    ax.set_title("Порівняння фактичного і прогнозованого значення",
                 color=_TEXT, fontsize=10, pad=10)

    # Set the label for the x-axis to 'Feature', indicating that the x-axis represents the input features.
    ax.set_xlabel("Ознака (Feature)", color=_MUTED, fontsize=9)

    # Set the label for the y-axis to 'Target', indicating that the y-axis represents the target values (actual or predicted).
    ax.set_ylabel("Ціль (Target)", color=_MUTED, fontsize=9)

    # Enable a grid on the plot to improve readability.
    ax.grid(True, color=_GRID, linewidth=0.6, linestyle="--", alpha=0.7)

    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)

    # Draw a dashed line ('k--' for black dashed line) between the actual and predicted target values to visually represent the difference.
    ax.plot([closest_x, input_feature], [actual_target, prediction],
            linestyle="--", color="#c0bdb5", linewidth=1.2)

    # Annotate the plot with the difference between the actual and predicted target values, positioned halfway between them and offset slightly for visibility.
    mid_x = (closest_x + input_feature) / 2
    mid_y = (actual_target + prediction) / 2
    ax.annotate(f"Δ = {float(difference):.4f}",
                xy=(mid_x, mid_y),
                xytext=(mid_x + 0.12, mid_y + 0.12),
                fontsize=8.5, color=_SAND,
                arrowprops=dict(arrowstyle="-", color=_SAND, lw=0.8))

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the 
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array 
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates 
    the absolute difference between each element in `X` and `k`, identifies the minimum value 
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple, 
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()