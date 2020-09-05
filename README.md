# PyTsetlinMachineCUDA - Massively Parallel and Asynchronous Architecture for Logic-based AI 

Using logical clauses to represent patterns, Tsetlin machines (https://arxiv.org/abs/1804.01508) have obtained competitive performance in terms of accuracy, memory footprint, energy, and learning speed on several benchmarks (image classification, regression and natural language understanding).

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/MassiveParallel.png">
</p>

In the parallel and asynchronous architecture implemented here, each clause runs in its own thread for massive parallelism. The clauses access the training examples simultaneously, updating themselves and local voting tallies in parallel (see figure).

A team of Tsetlin Automata composes each clause. The Tsetlin Automata thus drive the entire learning process. These are rewarded/penalized according to three local rules that optimize global behaviour (see https://github.com/cair/TsetlinMachine).

There is no synchronization among the clause threads, apart from atomic adds to the local voting tallies. Hence, the speed up!

<p>
The architecture currently supports multi-class classification, multiple layers (https://arxiv.org/abs/1804.01508), integer clause weighting (https://arxiv.org/abs/2005.05131, https://arxiv.org/abs/2002.01245), regression (https://royalsocietypublishing.org/doi/full/10.1098/rsta.2019.0165) and convolution (https://arxiv.org/abs/1905.09688). Support for local and global interpretability and Boolean embedding coming soon.
</p>

## Installation

```bash
pip install PyTsetlinMachineCUDA
```

## Examples

#### Code: NoisyXORDemo.py

```python
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np 

train_data = np.loadtxt("NoisyXORTrainingData.txt")
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("NoisyXORTestData.txt")
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200,batch_size=100)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))))
```

#### Output

```bash
python3 ./NoisyXORDemo.py 

Accuracy: 100.00%

Prediction: x1 = 1, x2 = 0, ... -> y = 1
Prediction: x1 = 0, x2 = 1, ... -> y = 1
Prediction: x1 = 0, x2 = 0, ... -> y = 0
Prediction: x1 = 1, x2 = 1, ... -> y = 0
```

### Interpretability Demo

#### Code: InterpretabilityDemo.py

```python
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
```

#### Output

```bash
python ./InterpretabilityDemo.py

Accuracy: 100.0

Class 0 Positive Clauses:

Clause #0:  ¬x0 ∧ ¬x1
Clause #2:   x0 ∧  x1
Clause #4:   x0 ∧  x1
Clause #6:  ¬x0 ∧ ¬x1
Clause #8:  ¬x0 ∧ ¬x1

Class 0 Negative Clauses:

Clause #1:   x0 ∧ ¬x1
Clause #3:   x0 ∧ ¬x1
Clause #5:   x1 ∧ ¬x0
Clause #7:   x1 ∧ ¬x0
Clause #9:   x0 ∧ ¬x1

Class 1 Positive Clauses:

Clause #0:   x1 ∧ ¬x0
Clause #2:   x1 ∧ ¬x0
Clause #4:   x0 ∧ ¬x1
Clause #6:   x0 ∧ ¬x1
Clause #8:   x0 ∧ ¬x1

Class 1 Negative Clauses:

Clause #1:   x0 ∧  x1
Clause #3:  ¬x0 ∧ ¬x1
Clause #5:  ¬x0 ∧ ¬x1
Clause #7:  ¬x0 ∧ ¬x1
Clause #9:   x0 ∧  x1
```

### MNIST Demo w/Weighted Clauses

#### Code: MNISTDemoWeightedClauses.py

```python
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50*16, 10.0, max_weight=16)

print("\nAccuracy over 100 epochs:\n")
for i in range(100):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python ./MNISTDemoWeightedClauses.py

Accuracy over 100 epochs:

#1 Accuracy: 93.00% Training: 4.91s Testing: 0.63s
#2 Accuracy: 94.48% Training: 3.29s Testing: 0.42s
#3 Accuracy: 95.57% Training: 3.32s Testing: 0.42s
...

#98 Accuracy: 98.12% Training: 3.06s Testing: 0.42s
#99 Accuracy: 98.20% Training: 3.06s Testing: 0.42s
#100 Accuracy: 98.16% Training: 3.06s Testing: 0.42s
```

### MNIST 2D Convolution Demo w/Weighted Clauses

#### Code: MNISTDemo2DConvolutionWeightedClauses.py

```python
from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = MultiClassConvolutionalTsetlinMachine2D(2000, 50*15, 5.0, (10, 10), max_weight=16)

print("\nAccuracy over 50 epochs:\n")
for i in range(50):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python ./MNISTDemo2DConvolutionWeightedClauses.py 

Accuracy over 50 epochs:

#1 Accuracy: 97.14% Training: 13.83s Testing: 1.82s
#2 Accuracy: 98.22% Training: 12.17s Testing: 1.39s
#3 Accuracy: 98.57% Training: 11.88s Testing: 1.39s
...

#48 Accuracy: 99.13% Training: 9.74s Testing: 1.38s
#49 Accuracy: 99.14% Training: 9.93s Testing: 1.38s
#50 Accuracy: 99.10% Training: 9.14s Testing: 1.38s
```

### Fashion MNIST 2D Convolution Demo w/Weighted Clauses

#### Code: FashionMNISTDemo2DConvolutionWeightedClauses.py

```python
from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import cv2
from keras.datasets import fashion_mnist

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

tm = MultiClassConvolutionalTsetlinMachine2D(8000, 100*100, 10.0, (10, 10), max_weight=255)

print("\nAccuracy over 30 epochs:\n")
for i in range(30):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python ./FashionMNISTDemo2DConvolutionWeightedClauses.py 

Accuracy over 30 epochs:

#1 Accuracy: 82.64% Training: 49.47s Testing: 7.13s
#2 Accuracy: 85.86% Training: 47.00s Testing: 6.25s
#3 Accuracy: 87.11% Training: 47.22s Testing: 6.52s
#4 Accuracy: 88.09% Training: 46.47s Testing: 6.59s
...

#27 Accuracy: 90.83% Training: 44.06s Testing: 5.90s
#28 Accuracy: 90.70% Training: 44.26s Testing: 6.14s
#29 Accuracy: 90.75% Training: 44.04s Testing: 5.96s
#30 Accuracy: 90.94% Training: 44.02s Testing: 6.12s
```

### IMDb Text Categorization Demo

#### Code: IMDbTextCategorizationDemo.py

```python
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from time import time

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
                terms.append(id_to_word[word_id])
        
        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        
                        if phrase in vocabulary:
                                vocabulary[phrase] += 1
                        else:
                                vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
        if vocabulary[phrase] < 10:
                continue

        phrase_bit_nr[phrase] = bit_nr
        bit_nr_phrase[bit_nr] = phrase
        bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
                terms.append(id_to_word[word_id])

        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        if phrase in phrase_bit_nr:
                                X_train[i,phrase_bit_nr[phrase]] = 1

        Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
                terms.append(id_to_word[word_id])

        for N in range(1,MAX_NGRAM+1):
                grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
                for gram in grams:
                        phrase = " ".join(gram)
                        if phrase in phrase_bit_nr:
                                X_test[i,phrase_bit_nr[phrase]] = 1                             

        Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

tm = MultiClassTsetlinMachine(10000, 80*16, 27.0, max_weight=16)

print("\nAccuracy over 50 epochs:\n")
for i in range(50):
        start_training = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output:

```bash
python ./IMDbTextCategorizationDemo.py

Producing bit representation...
Selecting features...

Accuracy over 50 epochs:

#1 Accuracy: 85.15% Training: 18.19s Testing: 6.61s
#2 Accuracy: 87.10% Training: 13.54s Testing: 4.71s
#3 Accuracy: 87.72% Training: 13.12s Testing: 4.78s
...

#48 Accuracy: 89.47% Training: 10.49s Testing: 5.82s
#49 Accuracy: 89.70% Training: 10.44s Testing: 5.83s
#50 Accuracy: 89.57% Training: 10.41s Testing: 5.84s
```

### Regression Demo

#### Code: RegressionDemo.py

```python
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
```

#### Output

```bash
python3 ./RegressionDemo.py 

RMSD over 25 runs:

#1 RMSD: 0.501 +/- 0.000 (12.96s)
#2 RMSD: 0.497 +/- 0.005 (12.90s)
...

#24 RMSD: 0.497 +/- 0.003 (12.91s)
#25 RMSD: 0.497 +/- 0.003 (12.95s)
```

## Paper

Coming soon.

## Requirements

- Python 3.7.x, https://www.python.org/
- Numpy, http://www.numpy.org/
- PyCUDA, https://documen.tician.de/pycuda/

## Acknowledgements

I thank my colleagues from the Centre for Artificial Intelligence Research (CAIR), Lei Jiao, Xuan Zhang, Geir Thore Berge, Darshana Abeyrathna, Saeed Rahimi Gorji, Sondre Glimsdal, Rupsa Saha, Bimal Bhattarai, Rohan K. Yadev, Bernt Viggo Matheussen, Morten Goodwin, Christian Omlin, Vladimir Zadorozhny (University of Pittsburgh), Jivitesh Sharma, and Ahmed Abouzeid, for their contributions to the development of the Tsetlin machine family of techniques. I would also like to thank our House of CAIR partners, Alex Yakovlev, Rishad Shafik, Adrian Wheeldon, Jie Lei, Tousif Rahman (Newcastle University), Jonny Edwards (Temporal Computing), Marco Wiering (University of Groningen), Christian D. Blakely (PwC Switzerland), Adrian Phoulady, Anders Refsdal Olsen, Halvor Smørvik, and Erik Mathisen for their many contributions.

## Tsetlin Machine Papers

```bash
@InProceedings{saha2020causal,
  author = {Rupsa {Saha} and Ole-Christoffer {Granmo} and Morten {Goodwin}},
  title = "{Mining Interpretable Rules for Sentiment and Semantic Relation Analysis using Tsetlin Machines}",
  booktitle="Lecture Notes in Computer Science: Proceedings of the 40th International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2020)", year="2020",
  publisher="Springer International Publishing"
}
```

```bash
@InProceedings{abeyrathna2020deterministic,
  title="{A Novel Multi-Step Finite-State Automaton for Arbitrarily Deterministic Tsetlin Machine Learning}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Rishad Shafik and Alex Yakovlev and Adrian Wheeldon and Jie Lei and Morten Goodwin},
  booktitle="Lecture Notes in Computer Science: Proceedings of the 40th International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2020)", year="2020",
  publisher="Springer International Publishing"
}
```

```bash
@article{zhang2020convergence,
  title="{On the Convergence of Tsetlin Machines for the IDENTITY- and NOT Operators}",
  author={Xuan Zhang and Lei Jiao and Ole-Christoffer Granmo and Morten Goodwin},
  journal = {arXiv preprint arXiv:2007.14268}, year = {2020},
  url = {https://arxiv.org/abs/2007.14268}
}
```

```bash
@article{blakely2020closedform,
  title="{Closed-Form Expressions for Global and Local Interpretation of Tsetlin Machines with Applications to Explaining High-Dimensional Data}",
  author={Christian D. Blakely and Ole-Christoffer Granmo},
  journal = {arXiv preprint arXiv:2007.13885}, year = {2020},
  url = {https://arxiv.org/abs/2007.13885}
}
```

```bash
@article{wheeldon2020learning, 
  author={Adrian {Wheeldon} and Rishad {Shafik} and Tousif {Rahman} and Jie {Lei} and Alex {Yakovlev} and Ole-Christoffer {Granmo}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{Learning Automata based Energy-efficient AI Hardware Design for IoT}",
  year={2020}
}
```

```bash
@InProceedings{shafik2020explainability,
  title="{Explainability and Dependability Analysis of Learning Automata based AI Hardware}",
  author={Rishad {Shafik} and Adrian {Wheeldon} and Alex {Yakovlev}},
  booktitle={IEEE 26th International Symposium on On-Line Testing and Robust System Design (IOLTS)},
  year={2020},
  organization={IEEE}
}
```

```bash
@article{lavrova2020,
  author = {D. S. {Lavrova} and N. N. {Eliseev}},
  title = "{Network Attacks Detection based on Tsetlin Machine}",
  pages = {17-23},
  journal = {Information Security Problems. Computer Systems.}, year = {2020}
}
```

```bash
@article{abeyrathna2020integer,
  author = {Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  title = "{Extending the Tsetlin Machine With Integer-Weighted Clauses for Increased Interpretability}",
  journal = {arXiv preprint arXiv:2005.05131}, year = {2020}
}
```

```bash
@InProceedings{gorji2020indexing,
  title="{Increasing the Inference and Learning Speed of Tsetlin Machines with Clause Indexing}",
  author={Saeed {Gorji} and Ole Christoffer {Granmo} and Sondre {Glimsdal} and Jonathan {Edwards} and Morten {Goodwin}},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{abeyrathna2020integerregression,
  title="{A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation,}",
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{phoulady2020weighted, 
  author={Adrian {Phoulady} and Ole-Christoffer {Granmo} and Saeed Rahimi {Gorji} and Hady Ahmady {Phoulady}}, 
  booktitle={Proceedings of the Ninth International Workshop on Statistical Relational AI (StarAI 2020)}, 
  title="{The Weighted Tsetlin Machine: Compressed Representations with Clause Weighting}",
  year={2020}
}
```

```bash
@InProceedings{wheeldon2020pervasive, 
  author={Adrian {Wheeldon} and Rishad {Shafik} and Alex {Yakovlev} and Jonathan {Edwards} and Ibrahim {Haddadi} and Ole-Christoffer {Granmo}}, 
  booktitle={SCONA Workshop at Design, Automation and Test in Europe (DATE 2020)}, 
  title="{Tsetlin Machine: A New Paradigm for Pervasive AI}",
  year={2020}
}
```

```bash
@article{abeyrathna2019nonlinear, 
  author={K. Darshana {Abeyrathna} and Ole-Christoffer {Granmo} and Xuan {Zhang} and Lei {Jiao} and Morten {Goodwin}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{The Regression Tsetlin Machine - A Novel Approach to Interpretable Non-Linear Regression}",
  volume={378}, issue={2164},
  year={2019}
}
```

```bash
@InProceedings{gorji2019multigranular,
  author = {Saeed Rahimi {Gorji} and Ole-Christoffer {Granmo} and Adrian {Phoulady} and Morten {Goodwin}},
  title = "{A Tsetlin Machine with Multigranular Clauses}",
  booktitle="Lecture Notes in Computer Science: Proceedings of the Thirty-ninth International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2019)", year="2019",
  volume = {11927},
  publisher="Springer International Publishing"
}
```

```bash
@article{berge2019text, 
  author={Geir Thore {Berge} and Ole-Christoffer {Granmo} and Tor Oddbjørn {Tveit} and Morten {Goodwin} and Lei {Jiao} and Bernt Viggo {Matheussen}}, 
  journal={IEEE Access}, 
  title="{Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications}",
  volume={7},
  pages={115134-115146}, 
  year={2019}, 
  doi={10.1109/ACCESS.2019.2935416}, 
  ISSN={2169-3536}
}
```

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal = {arXiv preprint arXiv:1905.09688}, year = {2019}
}
```

```bash
@InProceedings{abeyrathna2019regressiontsetlin,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Jiao}, Lei and {Goodwin}, Morten},
  title = "{The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems}",
  editor="Moura Oliveira, Paulo and Novais, Paulo and Reis, Lu{\'i}s Paulo ",
  booktitle="Progress in Artificial Intelligence", year="2019",
  publisher="Springer International Publishing",
  pages="268--280"
}
```

```bash
@InProceedings{abeyrathna2019continuousinput,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Zhang}, Xuan and {Goodwin}, Morten},
  title = "{A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks}",
  booktitle = "{Advances and Trends in Artificial Intelligence. From Theory to Practice}", year = "2019",
  editor = "Wotawa, Franz and Friedrich, Gerhard and Pill, Ingo and Koitz-Hristov, Roxane and Ali, Moonis",
  publisher = "Springer International Publishing",
  pages = "564--578"
}
```

```bash
@article{granmo2018tsetlin,
  author = {{Granmo}, Ole-Christoffer},
  title = "{The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}",
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018},
  url={https://arxiv.org/abs/1804.01508}
}
```

## Licence

Copyright (c) 2020 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
