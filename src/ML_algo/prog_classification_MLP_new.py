import pandas as pd
TestDatatset = "dataset_new.csv"
df = pd.read_csv(TestDatatset, sep=";")

#Définir les colonnes et le feature
X = df.iloc[:,0:170]
y = df.iloc[:,170]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3) Specify ML model configurations
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV # Create the parameter grid based on the results of random search

param_grid = {
    #'n_estimators': [50, 100]
}
# Define which metric will be used
score = 'precision'

# Create a based model
kn = MLPClassifier()# Instantiate the grid search model

# 4)  Train (Fit) the best model with training data
best_model_search = GridSearchCV(estimator = kn, param_grid = param_grid, cv = 4,
                                 scoring='%s_macro' % score, verbose = 2)
best_model_search.fit(X_train, y_train)

#  Show which is the best model
best_grid = best_model_search.best_estimator_
print("  ------------------------------------  ")
print ("BEST Configuration is  ==== ",best_grid )
print("  ------------------------------------  ")









# Predict the values using best_model
predictions = best_model_search.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions, y_test)
print("  ------------------------------------  ")
print(" Performance final = ", accuracy)
print("  ------------------------------------  ")


from sklearn.metrics import classification_report
# Display classification report - confusion matrix
print()
y_true, y_pred = y_test, best_grid.predict(X_test)
print(classification_report(y_true, y_pred))
print()
print("  ------------------------------------  ")

from sklearn.metrics import confusion_matrix
print("  ------------------------------------  ")
print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred))
print("  ------------------------------------  ")

from sklearn.metrics import confusion_matrix, plot_roc_curve
print("  ------------------------------------  ")
print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
plt.imshow(confusion_matrix(y_true, y_pred), cmap='binary', interpolation='None')
plt.show()
print("  ------------------------------------  ")

print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))