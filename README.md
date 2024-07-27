# Unacceptable Product Prediction

- This project has already been deployed on `Streamlit` and can be accessed through this link: https://nc-prediction.streamlit.app/

- I've also included a `Dashboard` using Looker Studio: https://lookerstudio.google.com/u/0/reporting/af684925-6d5b-4b7a-9e5b-f5ed9e48b797/page/mVE7D for analysis and visualization

## Project Overview
This project aims to predict the Non-Conformance (NC) percentage or unacceptable product rate in a manufacturing process using machine learning techniques. The project involves data cleaning, exploratory data analysis, model training, and the development of a Streamlit web application for real-time predictions and model interpretability.

## Demo
Here is a demo of the NC or `Unacceptable Product` prediction application based on Machine Learning using the XGBoost algorithm. SHAP analysis is also used to provide an overview of the influence of each feature on the NC prediction.
![Alt Text](https://github.com/beemabee/Factory_Prediction/raw/main/documentation/demo.gif)

## Project Structure
- `data/`: Stores the dataset and new_data after preprocessing
- `documentation/`: Stores the documentation and demo
- `model/`: Stores trained models and scalers
- `src/`: Contains the source code
  - `training.ipynb`: Jupyter notebook for data preprocessing and model training
  - `streamlit_app.py`: Streamlit application for predictions and SHAP analysis
- `requirement.txt`: Stores the required package

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/beemabee/Factory_Prediction.git
   cd Factory_Prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   or

   ```
   conda create --name ENVIRONTMENT_NAME python=3.11
   conda activate ENVIRONTMENT_NAME
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Preprocessing and Model Training
1. Open and run the `src/training.ipynb` notebook in Jupyter or any compatible environment.
2. The notebook covers:
   - Data loading and cleaning
   - Exploratory Data Analysis (EDA)
   - Feature selection and engineering
   - Model training (Linear Regression, Random Forest, XGBoost)
   - Hyperparameter tuning
   - Model evaluation

## Running the Local Streamlit Application
1. Ensure you have completed the model training step.
2. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```
3. Open the provided URL in your web browser.

## Using the Streamlit Application
1. Input values for each feature (SamplingNC, SamplingChek, QTY, TimeProduce, Years).
2. Click the "Prediksi dan Analisis" button.
3. View the predicted NC percentage and SHAP analysis visualizations.

## Project Methodology
1. Data Cleaning:
   - Handled missing values and outliers
   - Standardized data formats

2. Feature Selection:
   - Used correlation analysis to identify relevant features
   - Applied domain knowledge to select meaningful predictors

3. Model Training:
   - Experimented with Linear Regression, Random Forest, and XGBoost
   - Performed hyperparameter tuning using GridSearchCV
   - Selected the best performing model based on evaluation metrics

4. Model Interpretation:
   - Utilized SHAP (SHapley Additive exPlanations) for model interpretability
   - Implemented visualizations to explain feature importance and impact

## Technologies Used
- Python
- Pandas for data manipulation
- Scikit-learn for model training and evaluation
- XGBoost, Random Forest, and Linear Regression for modeling
- Streamlit for web application development
- SHAP for model interpretation

## Future Improvements
- Incorporate more advanced feature engineering techniques
- Experiment with ensemble methods for improved predictions
- Enhance the Streamlit UI for better user experience

## Contributors
- Andika Atmanegara Putra

## License
This project is licensed under the [MIT License](LICENSE).