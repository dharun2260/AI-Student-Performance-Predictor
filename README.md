# AI Student Performance Predictor

A beginner-friendly AI Engineering project built with Python, Pandas, Scikit-learn and Streamlit.

## Features
- Loads a sample student-performance dataset.
- Trains a Random Forest classification model.
- Predicts **At Risk**, **Average**, or **Excellent** performance.
- Displays model confidence and a simple recommendation.
- Provides test-set accuracy.

## Project structure
```text
ai_student_performance_project/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── student_performance.csv
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will open in your browser.

## Publish with Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select `app.py` as the main file.
5. Deploy.

## Note
The dataset is synthetic and intended for educational demonstration only.
