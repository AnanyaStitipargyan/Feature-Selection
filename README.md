# Feature-Selection
Feature selection is primarily focused on removing non-informative or redundant predictors from the model.

- Filter based: We specify some metric and based on that filter features.
     An example of such a metric could be correlation/chi-square.
- Wrapper-based: Wrapper methods consider the selection of a set of features as a search problem.
     Example: Recursive Feature Elimination
- Embedded: Embedded methods use algorithms that have built-in feature selection methods.
    For instance, Lasso and RF have their own feature selection methods.
