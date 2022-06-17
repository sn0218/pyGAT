# Pytorch Graph Attention Network (GAT)

## Description
- The project is the PyTorch implementation of Graph Attention Network (GAT). 
- The GAT model is presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903) and the official repository for the GAT is available in https://github.com/PetarV-/GAT 
- In this project, I modify the code based on the Pytorch implementation presented by Diego999 (https://github.com/Diego999/pyGAT). Visualizations are added in this project to demonstrate the meaningful representations such as feature maps, attention maps and t-SNE visualization of the model.


## Performance 
In the original paper, the result in terms of classification accuracies of Cora dataset in transductive learning task is about 83.0%. This GAT model obtains the accuracy about 85% in a single run.

### Experiment result

<img src="https://user-images.githubusercontent.com/48129546/174257758-340d0cc6-f11f-4b68-bb67-35d4be4de573.JPG" width="600">

![gat_result2](https://user-images.githubusercontent.com/48129546/174257940-a23d77f2-3c93-482d-b710-21ad0700d939.JPG)


## Visualization

### Feature maps 
- In the forward pass of the first GAT layer, it transforms the input feature vectors of dimension (2708, 1433) into (2708, 8) for each attention head because the number of hiddent units is set to 8. 
- The visualization of a head zero is shown in the following figure how the node neighbouring information is learnt to update the hidden representation. 


<img src="https://user-images.githubusercontent.com/48129546/174258756-a614c07a-2c7e-49dc-af24-39767284dbc7.png" width="400">


### Heat maps
- The attention weights αij in our pre-trained model of Cora dataset to visualize the attention distribution. 
- In the figure, v the attention scores of the first 10 nodes is visualized for simplicity. 
- The visualization is a heatmap generated by Seaborn. The magnitude of the learned attention scores is shown in the grid square.


<img src="https://user-images.githubusercontent.com/48129546/174259004-5842d11d-c3f0-4cf9-90ee-695128d4e0cc.png" width="700">


### t-SNE visualization
- For the transductive learning tasks, the GAT model is set to be two-layer architecture. The first layer is used to learn the 
neighbourhood features, while the second layer is used for classification. 
- t-SNE library is used to visualize the output of node feature vectors in the second layer as illustrated in figure 3. The dimension of the output is (2708, 7) because Cora dataset has 2708 nodes and 7 classes. We leverage t-SNE to map the 7-diemnsional vectors into 2d vectors to plot the nodes in a 2d plane. 
Nodes classified in the same class with the same colour are clustered. 


<img src="https://user-images.githubusercontent.com/48129546/174259439-434b8232-d7f1-437b-b504-62c44037ce89.png" width="600">


## Reference
If you make advantage of the GAT model in your research, please cite the following in your manuscript:
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

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]([https://opensource.org/licenses/MIT](https://github.com/sn0218/pyGAT/blob/master/LICENSE.md))
