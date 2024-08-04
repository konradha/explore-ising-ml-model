A repo to reproduce [Explaining the Machine Learning Solution of the Ising Model](https://arxiv.org/abs/2402.11701)
by Alamino from 2024.

We explore the author's suggestions.

- no theory work here.
- simulation employs [Wolff cluster updates](https://en.wikipedia.org/wiki/Wolff_algorithm) (I am lazy) -- just call `python generate_configs.py` 
- the simulation generates the right number of configs; it however is not aware of the sign such that the SLNN is
  trained on less configs than it should -- the work however still reproduces favorably (again, I am lazy) 
- PCA has a shortcoming here (secondary literature suggests nice separability for different projections on eigenvectors
  of M -- here, the particle number is not conserved so this is a little less visible)
- elegant idea of using a "SLNN" (a single layer NN) to predict the model's phase -- the SLNN is a simple logistic
  regression and it works nicely on data which has positive magnetization!
- SLNN args to activation function scale differently in this implementation -- might be due to other parameter samplers
  (maybe inherent in Pytorch as original work uses keras)
- the straightforward idea to extend the ML model by a 2-unit hidden layer can immediately predict the nice properties
  of the Ising model
- the graph on "inferred probabilities" is, at times, not reproducible -- we strongly depend on the seed;
  see below
- overall a very illuminating paper giving us _some_ intuition on what's actually going on for simple NN architectures
  _and_ we have some theory on the data-generating process  
- more implementation shortcomings: initialization for the Ising model and the paper's suggestions varies; "1HLNN" has
  no strictly correct way of initializing (needs work on that front to be fully explainable) 



Suggested way to run this repo:

```bash
micromamba env create -n repro-ising-ml 
micromamba activate repro-ising-ml
micromamba install -c conda-forge numpy tqdm pytorch scikit-learn matplotlib
# e.g.
python apply_model.py 20 500 100 20 mlnn
```


#### Comparing a little
The "explanatory" part for the network with a hidden layer is fuzzy. Results are not consistent. Trained on the
same data, same procedure, only the Pytorch-given seed differs -- the results are sometimes overwhelmingly different.


Compare

<table>
  <tr>
    <td><img src="./compare_mlnn_args/2ca847_hidden_final_units.png" alt="hidden final units 1"></td>
    <td><img src="./compare_mlnn_args/2ca847_preds.png" alt="preds 1"></td>
  </tr>
</table>

to

<table>
  <tr>
    <td><img src="./compare_mlnn_args/03fc00_hidden_final_units.png" alt="hidden final units 2"></td>
    <td><img src="./compare_mlnn_args/03fc00_preds.png" alt="preds 2"></td>
  </tr>
</table>

The test data here is now all of the data points that have been excluded from training, ie. all configs generated
for temperatures between 1 and 4.
