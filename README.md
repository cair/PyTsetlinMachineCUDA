# PyTsetlinMachineCUDA - A Massively Parallel and Asynchronous Architecture for Logic-based AI 

Using logical clauses to represent patterns, Tsetlin machines (https://arxiv.org/abs/1804.01508) have obtained competitive performance in terms of accuracy, memory footprint, energy, and learning speed on diverse benchmarks (image classification, regression and natural language understanding).

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/MassiveParallel.png">
</p>

In the parallel and asynchronous architecture implemented here, each conjunctive clause runs in its own thread for massive parallelism. The clauses access the training examples simultaneously, updating themselves and local voting tallies in the process (see figure).

A team of Tsetlin Automata composes each clause. The Tsetlin Automata thus drive the entire learning process. These are rewarded/penalized according to three local rules that optimize global behaviour (see https://github.com/cair/TsetlinMachine).

There is no synchronization among the clause threads, apart from atomic adds to the local voting tallies. Hence, the speed up!

<p>
The architecture currently supports clause weighting, classification, regression and convolution (support for local and global interpretability, Boolean embedding and multiple layers coming soon).
</p>

## Paper

Coming soon.
