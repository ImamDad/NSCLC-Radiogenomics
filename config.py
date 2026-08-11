model:
  name: "MS-HGNN"
  embedding_dim: 64
  hidden_dim: 128
  num_heads: 8
  num_layers: 3
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 3e-4
  weight_decay: 1e-2
  epochs: 100
  early_stopping_patience: 50

data:
  modalities: ["ct", "pet", "clinical", "genomic"]
  test_year: 1996
  validation_year: 1995
