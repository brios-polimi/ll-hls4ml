# Lab Notebook

## Experiment 1
### May 12


## Experiment 2
### June 12
When leaving one kernel type out at a time, what is the inductive vs. transductive generalization? Also serves as an initial baseline.

### Config
```json
{
  "model": "rgcn, GaTv2",
  "epochs": 250,
  "batch_size": 8,
  "patience": 50,
  "seed": 42,
  "hidden_dim": 32,
  "num_layers": 4,
  "dropout": 0.3,
  "tensors": {
    "2layer": 200,
    "3layer": 200,
    "conv1d": 173,
    "conv2d": 106,
    "dense_latency": 140,
    "dense_resource": 107,
    "exemplar": 300,
    "rule4ml": 200
  }
}
```

### Results
