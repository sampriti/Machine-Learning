# PCA example
# author: SAMPRITI NEOG

#########################################################################################################
# A project to analyse how many PCA components required in a MLP classifier to dectect mines accurately #
# and find out the chances of survival from the mines                                                   #
#########################################################################################################

import pandas as pd  # data frame
import matplotlib.pyplot as plt  # modifying plot
from sklearn.model_selection import train_test_split  # splitting data
from sklearn.preprocessing import StandardScaler  # scaling data

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA  # PCA package
from sklearn.metrics import accuracy_score  # grading
import itertools
from sklearn.metrics import confusion_matrix
from warnings import filterwarnings
##################################################
# To suppress the warnings of convergence issues #
##################################################
filterwarnings('ignore')

# read the database.
sonar = pd.read_csv('sonar_all_data_2.csv', header=None)
# print(sonar.shape)

########################################################################
# To separate the feature and required classes from the dataset "sonar"#
########################################################################
X = sonar.iloc[:, 0:60].values  # features are in columns 1:(N-1)
y = sonar.iloc[:, 61].values  # classes are in column 0!

# now split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

stdsc = StandardScaler()  # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# Initialise a list to store accuracy and no. of components for PCA analysis
accuracy = []  # List of test accuracy
comp = []      # List of no of components used for PCA
##########################################################################
# A loop to find out the accuracy for different number of PCA components #
##########################################################################
for i in range(0, 60):
    pca = PCA(n_components=i + 1)  # only keep two "best" features!
    X_train_pca = pca.fit_transform(X_train_std)  # apply to the train data
    X_test_pca = pca.transform(X_test_std)  # do the same to the test data
    comp.append(i)
    # now create a Logistic Regression and train on it
    lr = MLPClassifier(hidden_layer_sizes=(200), activation='logistic', max_iter=2000, alpha=0.00001, solver='adam',
                       tol=0.00001, random_state=10)
    lr.fit(X_train_pca, y_train)

    y_pred = lr.predict(X_test_pca)  # how do we do on the test data?
    print('for ' + str(i + 1) + ' components, Test Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

# Finding the maximum test accuracy
maximum = "{:.2f}".format(max(accuracy))

# Finding the no of PCA components required for maximum test accuracy
for i in range(0, 60):

    if "{:.2f}".format(accuracy[i]) == maximum:
        print("\n")
        print("For " + str(i + 1) + " components, we get maximum test accuracy of %2f" % max(accuracy))

# Printing the confusion matrix
print("The confusion matrix is:")
print(confusion_matrix(y_test, y_pred))


# A plot showing test accuracy versus no of PCA components
plt.plot(comp, accuracy)
plt.title('components vs accuracy')
plt.xlabel('components')
plt.ylabel('accuracy')
plt.show()
print("\n")


