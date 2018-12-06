import numpy as np


PREDICTED_CLASSES = np.array(['C1', 'C2', 'C2', 'C1', 'C3', 'C3', 'C2'])
CORRECT_CLASSES   = np.array(['C1', 'C1', 'C2', 'C1', 'C3', 'C2', 'C2'])

print((PREDICTED_CLASSES == CORRECT_CLASSES).mean())
# 0.714285714286


from sklearn.metrics import accuracy_score
print(accuracy_score(CORRECT_CLASSES, PREDICTED_CLASSES))
# 0.714285714286