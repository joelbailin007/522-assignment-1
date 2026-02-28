# MSIS 522 HW1 – Streamlit App

## Repository contents
- `joel_bailin_msis_522_hw1_analytics_for_covid_19_student_notebook.py`: original notebook export.
- `build_artifacts.py`: trains models once and saves reusable artifacts.
- `streamlit_app.py`: Streamlit deployment app with required tabs.
- `requirements.txt`: Python dependencies.
- `artifacts/` (generated): saved models, metrics, and plots loaded by the app.

## 1) Create venv and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Build pretrained artifacts
```bash
python build_artifacts.py --data covid.csv
```

### macOS note for LightGBM (`libomp`)
If you see an error like `Library not loaded: @rpath/libomp.dylib`, install OpenMP:

```bash
brew install libomp
```

Then re-run artifact generation.

If LightGBM is still unavailable, `build_artifacts.py` now automatically falls back to a scikit-learn boosting model so the pipeline can still complete.

## 3) Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Deployment
Deploy `streamlit_app.py` to Streamlit Community Cloud (or similar) after generating and committing `artifacts/` so the app loads pretrained models without retraining.
