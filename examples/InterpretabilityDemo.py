from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np 

number_of_features = 20
noise = 0.1

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

tm = MultiClassTsetlinMachine(10, 15, 3.0, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("\nClass 0 Positive Clauses:\n")
for j in range(0, 10, 2):
        print("Clause #%d: " % (j), end=' ')
        l = []
        for k in range(number_of_features*2):
                if tm.ta_action(0, j, k) == 1:
                        if k < number_of_features:
                                l.append(" x%d" % (k))
                        else:
                                l.append("¬x%d" % (k-number_of_features))
        print(" ∧ ".join(l))

print("\nClass 0 Negative Clauses:\n")
for j in range(1, 10, 2):
        print("Clause #%d: " % (j), end=' ')
        l = []
        for k in range(number_of_features*2):
                if tm.ta_action(0, j, k) == 1:
                        if k < number_of_features:
                                l.append(" x%d" % (k))
                        else:
                                l.append("¬x%d" % (k-number_of_features))
        print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")
for j in range(0, 10, 2):
        print("Clause #%d: " % (j), end=' ')
        l = []
        for k in range(number_of_features*2):
                if tm.ta_action(1, j, k) == 1:
                        if k < number_of_features:
                                l.append(" x%d" % (k))
                        else:
                                l.append("¬x%d" % (k-number_of_features))
        print(" ∧ ".join(l))

print("\nClass 1 Negative Clauses:\n")
for j in range(1, 10, 2):
        print("Clause #%d: " % (j), end=' ')
        l = []
        for k in range(number_of_features*2):
                if tm.ta_action(1, j, k) == 1:
                        if k < number_of_features:
                                l.append(" x%d" % (k))
                        else:
                                l.append("¬x%d" % (k-number_of_features))
        print(" ∧ ".join(l))
