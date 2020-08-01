# PyTsetlinMachineCUDA - A Massively Parallel and Asynchronous Architecture for Logic-based AI 

Using logical clauses to represent patterns, Tsetlin machines (https://arxiv.org/abs/1804.01508) have obtained competitive performance in terms of accuracy, memory footprint, energy, and learning speed on several benchmarks (image classification, regression and natural language understanding).

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/MassiveParallel.png">
</p>

In the parallel and asynchronous architecture implemented here, each clause runs in its own thread for massive parallelism. The clauses access the training examples simultaneously, updating themselves and local voting tallies in parallel (see figure).

A team of Tsetlin Automata composes each clause. The Tsetlin Automata thus drive the entire learning process. These are rewarded/penalized according to three local rules that optimize global behaviour (see https://github.com/cair/TsetlinMachine).

There is no synchronization among the clause threads, apart from atomic adds to the local voting tallies. Hence, the speed up!

<p>
The architecture currently supports clause weighting, classification, regression and convolution (support for local and global interpretability, Boolean embedding and multiple layers coming soon).
</p>

## Examples

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
@article{abeyrathna2020deterministic,
  title="{A Novel Multi-Step Finite-State Automaton for Arbitrarily Deterministic Tsetlin Machine Learning}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Rishad Shafik and Alex Yakovlev and Adrian Wheeldon and Jie Lei and Morten Goodwin},
  journal = {arXiv preprint arXiv:2007.02114}, year = {2020}
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
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018}
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
