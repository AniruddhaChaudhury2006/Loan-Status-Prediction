## Loan Prediction Project

This notebook demonstrates a machine learning workflow to predict loan approval status based on various applicant features.

### Data Preprocessing

- **Loading Data**: The `loan.csv` dataset is loaded into a pandas DataFrame.
- **Handling Missing Values**: Missing values in the dataset are dropped.
- **Categorical Data Encoding**: Categorical features like 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', and 'Loan_Status' are converted into numerical representations using `replace()`.
- **Feature Engineering**: The 'Dependents' column's '3+' value is replaced with '4' to make it numerical.

### Model Training

- **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`.
- **Model Selection**: A Support Vector Machine (SVM) classifier with a linear kernel is used for prediction.
- **Model Training**: The SVM model is trained on the preprocessed training data.

### Prediction and Evaluation

- **Prediction**: The trained model makes predictions on both the training and testing datasets.
- **Accuracy Score**: The accuracy of the model is evaluated on both training and test data to assess its performance.
