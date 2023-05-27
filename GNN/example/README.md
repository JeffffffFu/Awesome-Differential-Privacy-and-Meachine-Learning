# Example description
介绍：此目录下的所有脚本都可直接运行

| Script                                                                               | Description                         
|:-------------------------------------------------------------------------------------|:------------------------------------
| [node_classification_gcn_via_subgraph.py](node_classification_gcn_via_subgraph.py)   | 以node为单位切分为子图，进行node classification |
| [extract_node_subgraphs.py](extract_node_subgraphs.py)                               | 提取node的k-hop子图
| [dpgnn_node_classification.py](dpgnn_node_classification.py)                         | DPGNN实现，node classification任务，进行了degree限制之后sample subgraph，使用GCN；在cora数据集上可以在sigma=0.01时取得72%左右的acc，sigma=1.23时，0.56。                   