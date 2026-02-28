# MSIS 522 HW1 – Streamlit App

## Repository contents
- `joel_bailin_msis_522_hw1_analytics_for_covid_19_student_notebook.py`: original notebook export.
- `build_artifacts.py`: trains models once and saves reusable artifacts.
- `streamlit_app.py`: Streamlit deployment app with required tabs.
- `requirements.txt`: Python dependencies.
- `artifacts/` (generated): saved models, metrics, and plots loaded by the app.

## 1) Build pretrained artifacts
```bash
python build_artifacts.py --data covid.csv
```

## 2) Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Deployment
Deploy `streamlit_app.py` to Streamlit Community Cloud (or similar) after generating and committing `artifacts/` so the app loads pretrained models without retraining.
