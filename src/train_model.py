import os
import pickle
from datetime import datetime

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# setup variables
INPUT_DIR = './img'
CATEGORIES = ['1','2','3','4','5','6']
CURRENT_DAY=datetime.now()
# Get and Set data
data = []
labels = []

for category_idx, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(INPUT_DIR, category)):
        img_path = os.path.join(INPUT_DIR, category, file)
        img = imread(img_path)
        img = resize(img, (20, 20))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))
# Save model
pickle.dump(best_estimator, open(f'./models/model_{CURRENT_DAY.day}-{CURRENT_DAY.month}-{CURRENT_DAY.year}.p', 'wb'))
