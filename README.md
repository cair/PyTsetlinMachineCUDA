# PyTsetlinMachineCUDA - A Massively Parallel and Asynchronous Architecture for Logic-based AI 

Each logical clause runs in its own thread and accesses the training examples one at a time,  updating itself and a local voting tally in the process.
A team of Tsetlin Automata, which also can run efficiently in parallel with special-purpose hardware, composes the clause.

The Tsetlin Automata thus drive the entire learning process. These are rewarded/penalized according to three local rules that optimize global behaviour (see https://github.com/cair/TsetlinMachine).

There is no synchronization among the clause threads, apart from atomic adds to the local voting tallies. Hence, the speed up!

<p>
The architecture currently supports clause weighting, classification, regression and convolution (support for local and global interpretability, Boolean embedding and multiple layers coming soon).
</p>
