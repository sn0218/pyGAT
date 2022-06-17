# Pytorch Graph Attention Network (GAT)

## Description
The project is the PyTorch implementation of Graph Attention Network (GAT). 
The GAT model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903) and the official repository for the GAT is available in https://github.com/PetarV-/GAT 
In this project, I modify the code based on the Pytorch implementation presented by Diego999 (https://github.com/Diego999/pyGAT).
Visualizations are added in this project to demonstrate the feature maps and attention maps of the GAT model.

## Performance of the model
In the original paper, the result in terms of classification accuracies of Cora dataset in transductive learning task is about 83.0%.
This GAT model obtains the accuracy about 85% in a single run.


## Reference
```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```
