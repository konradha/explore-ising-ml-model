A repo to reproduce [Explaining the Machine Learning Solution of the Ising Model](https://arxiv.org/abs/2402.11701)
by Alamino from 2024.

We explore the suggested methods:

- PCA has a shortcoming here (secondary literature suggests nice separability for different projections on eigenvectors
  of M -- here, the particle number is not conserved so this is a little less visible)
- elegant idea of using a "SLNN" (a single layer NN) to predict the model's phase -- the SLNN is a simple logistic
  regression and it works nicely on data which has positive magnetization!
- the straightforward idea to extend the ML model by a 2-unit hidden layer can immediately predict the nice properties
  of the Ising model
- the graph on "inferred probabilities" is, at times, not reproducible -- we strongly depend on the seed but I wager
  that this happens in less than 10% of cases
- overall a very illuminating paper giving us _some_ intuition on what's actually going on for simple NN architectures
  _and_ we have some theory on the data-generating process  
