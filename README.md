<hr>

<h1> Refining Latent Representations: A Generative SSL Approach for Heterogeneous Graph Learning </h1>


Implementation for our paper submitted to AAAI'25. 


<h2>Dependencies </h2>

* Python >= 3.8
* [Pytorch](https://pytorch.org/) >= 1.9.0 
* [dgl](https://www.dgl.ai/) >= 0.7.2


<h2>Quick Start </h2>

For node classification task, you could run the scripts: 

**Node classification**

```bash
python main.py --dataset dblp
```

Run the scripts provided or add `--use_cfg` in command to reproduce the reported results.

Supported datasets: "dblp", "freebase", "acm", "aminer".

