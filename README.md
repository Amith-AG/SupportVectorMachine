# SupportVectorMachine

The provided code demonstrates the usage of Support Vector Machine (SVM) for a classification task using the diabetes dataset. SVM is a supervised machine learning algorithm used for both regression and classification tasks. It finds an optimal hyperplane in a high-dimensional space to separate different classes of data points.

Here is a step-by-step explanation of the code:

1. Import necessary libraries: The code begins by importing the required libraries. `pandas` is imported as `pd` for data manipulation, and `numpy` is imported as `np` for numerical computations. The diabetes dataset is loaded using `pd.read_csv` from the 'diabetes.csv' file.

2. Split the data into training and test sets: The code uses `train_test_split` from `sklearn.model_selection` to split the data into training and test sets. It takes the input features (`diabetes.loc[:, diabetes.columns != 'Outcome']`) and the target variable (`diabetes['Outcome']`). The `stratify` parameter ensures that the class distribution is preserved in the train-test split. The `random_state` parameter is set to 66 for reproducibility. The resulting splits are stored in `X_train`, `X_test`, `y_train`, and `y_test`.

3. Create an instance of SVM classifier: The code creates an instance of the SVM classifier using `SVC()`. By default, it uses the radial basis function (RBF) kernel, which is commonly used for classification problems.

4. Fit the SVM model: The `fit` method is used to train the SVM model. It takes the training data (`X_train` and `y_train`) as input and fits the model to the data.

5. Print accuracy on training and test sets: The code calculates and prints the accuracy of the SVM model on the training and test sets using `svc.score(X_train, y_train)` and `svc.score(X_test, y_test)`, respectively. The `score` method returns the mean accuracy on the given data and labels.

In summary, this code loads the diabetes dataset, splits it into training and test sets, creates an instance of the SVM classifier, trains the model, and evaluates its accuracy on both the training and test sets. The SVM algorithm finds an optimal hyperplane to separate different classes in the data and is commonly used for classification tasks.
