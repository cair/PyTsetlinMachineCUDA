from PyTsetlinMachineCUDA.tm import RegressionTsetlinMachine
from PyTsetlinMachineCUDA.tools import Booleanizer
import numpy as np
from time import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

california_housing = datasets.fetch_california_housing()
X = california_housing.data
Y = california_housing.target

b = Booleanizer(max_bits_per_feature = 25)
b.fit(X)
X_transformed = b.transform(X)

tm = RegressionTsetlinMachine(16*1000, 16*500*10, 10.0, max_weight=255)

X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y)

print("\nRMSD over 25 runs:\n")
tm_results = np.empty(0)
for i in range(25):
	X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y)

	start = time()
	tm.fit(X_train, Y_train, epochs=100, incremental=True)
	stop = time()
	tm_results = np.append(tm_results, np.sqrt(((tm.predict(X_test) - Y_test)**2).mean()))

	print("#%d RMSD: %.3f +/- %.3f (%.2fs)" % (i+1, tm_results.mean(), 1.96*tm_results.std()/np.sqrt(i+1), stop-start))
