** SPiCe competition offline files **
* http://spice.lif.univ-mrs.fr/ *

This archive contains the files needed to use offline the data of the SPiCe competition.

The Sequence PredIction ChallengE (SPiCe) took place between April and July 2016: participants had to learn a model from a set of whole sequences and then to guess a rankings of 5 symbols that are the most likely to be the next symbol of some prefixes.

Each problem came with two prefixes sets: a public test set and a private one. Participant were allowed unlimited number of submissions on the public test set (validation sample) and only one on the private test set (the true test sample). Results of the competition relied only on the score on the private test set. The competition was on-line: only the first prefixes of the test sets were given beforehand: participants had to submit their ranking on a first prefix in order to be given the second one, and so on.

This archive contains all elements needed to be able to use the SPiCe data sets without the long iterative submission process.

SPiCe data consist in 15 problems (plus the early Problem 0, which is a toy one, used before the beginning of the competition to test the platform): 4 purely synthetic, 4 partly synthetic (i.e synthetic ones obtained from computation on real-world data), and 7 real-world data (NLP, biology, software engineering).

** Content of this archive **
This archive is made of 4 folders:

- 'train' contains the training files, i.e. the 15 samples of complete sequences. It is in PAutomaC/SPiCe format: first line contains the number of examples and the size of the alphabet (i.e. the number of potential different symbols), the other lines correspond each to one sequence, starting by its length and followed by its different elements (integers) separated pairwise by a space.

- 'prefixes' contains the public and private test sets. Contrary to the on-line version that were made of only one element, these files contains all the prefixes of the sample, in PAutomaC/SPiCe format.

- 'targets' contains the elements needed to compute the score on each test set. These files are made of as many lines as there are prefixes in the test set. Each line starts by the denominator of the score function (independent of the proposed ranking) and then for each possible next symbol with non-null probability, the symbol followed by its probability is given. This allows the computation of the score metric used during the competition: the normalized discounted cumulative gain at 5 (more details on the website).

- 'codes' contains useful python scripts:
3gram_baseline.py that computes rankings for all prefixes of a test set given a train set using the 3-gram baseline;
spectral_baseline.py that computes rankings for all prefixes of a test set given a train set using the spectral learning baseline;
score_computation.py that computes the score given the output of the baselines (or of any program that follows that format) and a target file (the one of the 'target' directory).
Use-cases are given for each of these scripts at the beginning of the files.

**Use and cite this benchmark**
These data sets are not licensed: you are free to use them the way you want.
If you are using them in the context of a scientific paper, we would like you to cite the paper presenting the results of the competition:

@inproceedings{spice2016,
    title = {Results of the Sequence PredIction ChallengE ({SPiCe}): a competition about learning the next symbol in a sequence},
    author = {Balle Pigem, Borja and Eyraud, R{\'}emi and Luque, Franco M. and Quattoni, Ariadna and Verwer, Sicco},
    booktitle = {Proceedings of the 13th International Conference on Grammatical Inference (ICGI)},
    year = {2016},
    pages = {to appear}
}



