# Machine Learning Educational Web App - AI Coding Guide

This is a Flask-based educational web application demonstrating supervised machine learning algorithms through interactive examples and theoretical concepts.

## Architecture Overview

**Core Pattern**: Single Flask app (`app.py`) with modular ML models as importable Python files
- Each ML algorithm lives in its own module (`LinearRegression601T.py`, `LogisticRegression601T.py`, `perceptron_senales_industriales_mejorado.py`)
- Models expose standardized interfaces: training, prediction, and evaluation functions
- Flask routes handle both concept pages (GET) and interactive predictions (POST)
- Matplotlib graphs are converted to base64 images for web display using `io.BytesIO()` pattern

## Key ML Model Patterns

**Model Structure**: Each ML module follows this pattern:
```python
# Data generation/loading
# Model training (sklearn-based)
# Prediction functions
# Evaluation/plotting functions
# Class wrapper for Flask integration
```

**Standard Model Interface**:
- `train()` - trains model with data splitting and preprocessing
- `predict_label(features)` - returns ("Sí"/"No", probability)
- `evaluate()` - returns metrics dict and generates plots
- `plot_confusion_matrix()`, `plot_roc_curve()` - visualization methods

## Flask Integration Patterns

**Image Generation**: Models generate matplotlib plots that Flask converts to base64:
```python
buffer = io.BytesIO()
model.plot_function()
plt.savefig(buffer, format='png')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()  # Always close to prevent memory leaks
```

**Route Structure**: Each ML topic has 2 routes:
- `/conceptos_[topic]` - Theory/concept pages (GET only)  
- `/[topic]` - Interactive prediction pages (GET/POST)

**Model Loading**: Models are instantiated at module level and imported into Flask:
```python
from LogisticRegression601T import modelo_logistico
from perceptron_senales_industriales_mejorado import PerceptronClassifier
```

## Development Workflow

**Running Locally**:
```bash
cd Proyecto
pip install -r requirements.txt
python app.py  # Starts on localhost:5000
```

**Adding New ML Models**:
1. Create `[ModelName]601T.py` with standard interface
2. Add routes in `app.py` following existing pattern
3. Create corresponding HTML templates in `templates/`
4. Update navigation in `templates/index.html`

**Dependencies**: Core stack is Flask + scikit-learn + matplotlib + pandas. Models use sklearn for preprocessing (StandardScaler, train_test_split) and Pipeline pattern to avoid data leakage.

## UI/Template Patterns

**Template Structure**: Bootstrap 5.3.8 + custom CSS (`static/style.css`)
- Dark theme with consistent color scheme (see CSS variables)
- Form inputs use `form-control form-control-lg` classes
- Results displayed in alert boxes with success/warning styling
- Responsive grid layout with visualization cards

**Navigation**: Dropdown-based menu structure in `index.html` with consistent URL routing pattern.

## Data Handling

**Synthetic Data**: Models generate synthetic datasets with controlled seeds for reproducibility
**File Persistence**: Some models save/load using joblib (e.g., `modelo_citas.joblib`)
**CSV Exports**: Test data exported as CSV files for web app integration

When working with this codebase, maintain the existing patterns for ML model interfaces and Flask route structures. Always use matplotlib's `Agg` backend and proper buffer management for web-compatible image generation.