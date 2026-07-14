from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

# ── Палітра ──────────────────────────────────────────────────────────────────
_BG     = "#F5F6F2"
_TEXT   = "#22252A"
_MUTED  = "#6B7280"
_BORDER = "#D9DCD3"
_OLIVE  = "#6B7435"
_SAND   = "#C9A96E"
_BLUE   = "#2C6FAC"
_ORANGE = "#C96B2C"

_CSS = """
<style>
  .main .block-container {
      max-width: 740px;
      padding-top: 1.6rem;
      padding-bottom: 2.5rem;
  }
  .ctrl-card {
      background: #ffffff;
      border: 1px solid #D9DCD3;
      border-radius: 8px;
      padding: 1.2rem 1.4rem 1rem;
      margin-bottom: 0.8rem;
  }
  .result-row {
      display: flex;
      gap: 1rem;
      margin-bottom: 0.8rem;
  }
  .metric-box {
      flex: 1;
      background: #ffffff;
      border: 1px solid #D9DCD3;
      border-radius: 8px;
      padding: 0.8rem 1.1rem;
  }
  .metric-label { font-size: 0.78rem; color: #6B7280; margin-bottom: 0.15rem; }
  .metric-value { font-size: 1.6rem; font-weight: 700; }
  .olive-val { color: #6B7435; }
  .blue-val  { color: #2C6FAC; }
  .orange-val{ color: #C96B2C; }
  .page-badge {
      display: inline-block;
      font-size: 0.75rem;
      color: #6B7435;
      border: 1px solid #6B7435;
      border-radius: 4px;
      padding: 0.1rem 0.55rem;
      margin-bottom: 0.4rem;
  }
  div[data-testid="stButton"] > button {
      background-color: #6B7435 !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 5px !important;
      padding: 0.45rem 1.6rem !important;
      font-weight: 600 !important;
      font-size: 0.95rem !important;
  }
  div[data-testid="stButton"] > button:hover {
      background-color: #7e8a3e !important;
  }
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
    st.set_page_config(
        page_title="Лінійний прогноз: інтерактивна демонстрація",
        layout="centered",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Streamlit app title
    st.markdown('<div class="page-badge">Навчальний модуль лінійної регресії</div>',
                unsafe_allow_html=True)
    st.title("Лінійний прогноз: інтерактивна демонстрація")
    st.caption("Оберіть значення ознаки та натисніть кнопку, щоб отримати прогноз лінійної регресії.")

    # Картка керування
    st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
    # User input for new prediction using a slider
    input_feature = st.slider(
        "Значення ознаки X",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.01,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Button to make a prediction
    if st.button("Передбачити значення"):
        try:
            prediction = load_and_predict([[input_feature]])
        except FileNotFoundError:
            st.error("Файл моделі не знайдено. Спочатку запустіть `python model.py`.")
            return

        pred_scalar = float(np.asarray(prediction).flat[0])

        # Картки метрик
        st.markdown(
            f'<div class="result-row">'
            f'<div class="metric-box">'
            f'<div class="metric-label">Прогноз моделі</div>'
            f'<div class="metric-value olive-val">{pred_scalar:.4f}</div>'
            f'</div>'
            f'<div class="metric-box">'
            f'<div class="metric-label">X (введене значення)</div>'
            f'<div class="metric-value" style="color:{_MUTED}">{input_feature:.2f}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Plot the entire dataset (X, y) as grey dots to visualize the data distribution.
    ax.scatter(X, y, color="#9ca3af", alpha=0.55, s=28, label="Набір даних")

    # Plot the actual target value for a specific input feature as a blue dot.
    closest_x = float(np.asarray(X).flat[_index_of_closest(X, input_feature)])
    ax.scatter(closest_x, actual_target, color=_BLUE, s=90, zorder=5,
               edgecolors="#5b9fd4", linewidths=0.8, label="Фактичне значення")

    # Plot the predicted target value for the same input feature as a red dot.
    ax.scatter(input_feature, prediction, color=_ORANGE, s=90, zorder=5,
               edgecolors="#d98a5a", linewidths=0.8, label="Прогноз")

    # Display a legend on the plot to label the different scatter points (dataset, actual target, predicted target).
    ax.legend(facecolor="#ffffff", edgecolor=_BORDER,
              labelcolor=_TEXT, fontsize=8.5)

    # Set the title of the plot, describing what is being visualized.
    ax.set_title("Порівняння фактичного і прогнозованого значення",
                 color=_TEXT, fontsize=10, pad=10)

    # Set the label for the x-axis to 'Feature', indicating that the x-axis represents the input features.
    ax.set_xlabel("Ознака (Feature)", color=_MUTED, fontsize=9)

    # Set the label for the y-axis to 'Target', indicating that the y-axis represents the target values (actual or predicted).
    ax.set_ylabel("Ціль (Target)", color=_MUTED, fontsize=9)

    # Enable a grid on the plot to improve readability.
    ax.grid(True, color=_BORDER, linewidth=0.7, linestyle="--", alpha=0.9)

    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)

    # Draw a dashed line ('k--' for black dashed line) between the actual and predicted target values to visually represent the difference.
    ax.plot([closest_x, input_feature], [actual_target, prediction],
            linestyle="--", color="#22252A", linewidth=1.1)

    # Annotate the plot with the difference between the actual and predicted target values, positioned halfway between them and offset slightly for visibility.
    mid_x = (closest_x + input_feature) / 2
    mid_y = (actual_target + prediction) / 2
    ax.annotate(
        f"Δ = {float(difference):.4f}",
        xy=(mid_x, mid_y),
        xytext=(mid_x + 0.15, mid_y + 0.15),
        fontsize=8.5, color=_TEXT,
        arrowprops=dict(arrowstyle="-", color=_MUTED, lw=0.8),
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    Знаходить індекс елементу в масиві ``X``, найближчого до значення ``k``.

    Args:
        X (ArrayLike): Масив числових значень.
        k (float): Значення, до якого шукається найближче.

    Returns:
        int: Індекс найближчого елементу.
    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()