# ML_Project33-CarPricePrediction

### Car Price Prediction with Machine Learning
This project explores car price prediction using a machine learning model based on a used car dataset.

### Dataset
The code assumes a CSV file named car_data.csv containing information about used cars, including:
Car Name
Year
Selling Price
Present Price
Kms Driven
Fuel Type
Seller Type
Transmission
Owner

### Note: 
You might need to modify the code to point to the location of your specific data file.

### Functionality
Data Loading and Preprocessing:

Reads the CSV data using pandas.

Explores the data (head, shape, data types, missing values).

Creates new features like the difference between the current year and the car's year.

Encodes categorical features (e.g., Fuel Type, Seller Type) using one-hot encoding.

Identifies important features using feature importance analysis with ExtraTreesRegressor (optional).

### Model Training and Evaluation:

Splits the data into training and testing sets.

Trains a Random Forest Regression model on the training data.

Hyperparameter tuning using RandomizedSearchCV (optional).

Evaluates the model performance on the testing data using mean squared error.

Visualizes the predicted prices vs actual prices.

### Model Saving:
Saves the trained model using pickle for future predictions (optional).

### Future Enhancements:

Explore different machine learning algorithms (e.g., Gradient Boosting Regressors, Support Vector Machines).

Feature engineering (e.g., combining features, creating polynomial features).

Model selection and hyperparameter tuning using GridSearchCV.

Saving and loading the trained model for deployment.

User interface for price prediction based on car details.

Running the Script

### Install required libraries:
pandas
numpy
seaborn
matplotlib
scikit-learn (specifically ExtraTreesRegressor, RandomForestRegressor, RandomizedSearchCV, pickle)

### Save the code:
Save the provided code as a Python file (e.g., car_price_prediction.py).

Update the path to your CSV file (car_data.csv) in the script (if necessary).

### Run the script:
Execute the script from your terminal: python car_price_prediction.py

This script will perform data loading, preprocessing, model training, evaluation, and potentially save the trained model.
