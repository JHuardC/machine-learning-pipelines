# Machine Learning Pipelines

The field of machine learning is huge, with novel techniques to modelling data being developed all the time. The pace at which advances are made, makes it impossible for any one Python Package to create objects for all techniques, as such the machine learning programmer will often find they have to combine tools from separate packages. Combining tools from different packages into one pipeline can slow development time, as syntaxes, inputs and outputs are not always consistent accross packages.

The Machine Learning Pipelines (MLP) repo has been developed to help coordinate any Machine Learning process where modelling and tranformation tools come from separate packages. MLP is based on the observatiom that applying machine learning techniques/tools to data tends to consist of a common set of processes:

1. **Saving** a model/transformer and it's outputs.
2. **Loading** a trained model/transformer state for reuse.
3. **Fitting** a transformer to data; borrowing from [scikit-learn's](https://scikit-learn.org/stable/data_transforms.html) terminology, where a fit method learns model parameters (e.g. mean and standard deviation for normalization).
4. **Transforming** data (e.g. normalization).
5. **Training** a model.
6. **Predicting** target variables.

**README Contents:**

- [Local Build](#Local-Build)
- [Description](#Description-1)
- [Usage Example](#Usage-Example)
- [Extending pipelines to handle new packages](#Extending-pipelines-to-handle-new-packages)
- [The ModellingPipeline environment](#The-ModellingPipeline-environment)