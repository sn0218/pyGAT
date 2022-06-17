# Pytorch Graph Attention Network (GAT)

## Description
The project is the PyTorch implementation of Graph Attention Network (GAT). 
The GAT model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903) and the official repository for the GAT is available in https://github.com/PetarV-/GAT 
In this project, I modify the code based on the Pytorch implementation presented by Diego999 (https://github.com/Diego999/pyGAT).
Visualizations are added in this project to demonstrate the meaningful representations such as feature maps, attention maps and t-SNE visualization of the model.

## Performance 
In the original paper, the result in terms of classification accuracies of Cora dataset in transductive learning task is about 83.0%.
This GAT model obtains the accuracy about 85% in a single run.

### Experiment result
![gat_result](https://user-images.githubusercontent.com/48129546/174257758-340d0cc6-f11f-4b68-bb67-35d4be4de573.JPG)

![gat_result2](https://user-images.githubusercontent.com/48129546/174257940-a23d77f2-3c93-482d-b710-21ad0700d939.JPG)

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
