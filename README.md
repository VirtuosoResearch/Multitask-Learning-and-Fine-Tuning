
## Multitask Learning and Fine-Tuning

### Surveys

- Zhang, Y., & Yang, Q. (2021). A survey on multi-task learning. IEEE transactions on knowledge and data engineering. 
- Jiang et al. (2022). Transferability in deep learning: A survey. arXiv. 

### Multitask Learning Basics

- Caruana, R. (1997). Multitask learning. *Machine learning*. [paper](https://link.springer.com/article/10.1023/a:1007379606734)
- Caruana, R. (1996). Algorithms and applications for multitask learning. In *ICML*. [Paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3980c955f95092e527c580f9cfe066a17f752c08)
- Duong et al. (2015). Low resource dependency parsing: Cross-lingual parameter sharing in a neural network parser. In *ACL*. 
- Yang, Y., & Hospedales, T. (2016). Deep multi-task representation learning: A tensor factorisation approach. *ICLR.* [Paper](https://arxiv.org/abs/1605.06391)
- GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR 2019. [paper](https://arxiv.org/pdf/1804.07461)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arVix 2018. [paper](https://arxiv.org/pdf/1810.04805.pdf)
- Multi-task Sequence to Sequence Learning. ICLR 2016. [paper](https://arxiv.org/pdf/1511.06114)
- The natural language decathlon: Multitask learning as question answering.  arXiv 2019. [paper](https://arxiv.org/pdf/1806.08730)
- Understanding and Improving Information Transfer in Multi-Task Learning. ICLR 2020. [paper](https://openreview.net/pdf?id=SylzhkBtDB)
- Multi-Task Deep Neural Networks for Natural Language Understanding. ACL 2019. [paper](https://arxiv.org/pdf/1901.11504)

### **Task Relatedness**

**Theoretical notions of task relatedness.** 

- Ben-David, S., & Schuller, R. (2003). Exploiting task relatedness for multiple task learning. In *Learning Theory and Kernel Machines*. [paper](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_41)
- Ben-David et al. (2010). A theory of learning from different domains. *Machine learning* [paper](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- Hanneke, S., & Kpotufe, S. (2019). On the value of target data in transfer learning. *Advances in Neural Information Processing Systems*. [Paper](https://proceedings.neurips.cc/paper/2019/hash/b91f4f4d36fa98a94ac5584af95594a0-Abstract.html)
- Du et al. (2020). Few-shot learning via learning the representation, provably. *ICLR*. [paper](https://arxiv.org/abs/2002.09434)

**Measurements in deep neural networks.**

Grdients

- Yu et al. (2020). Gradient surgery for multi-task learning. *NeurIPS.* [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- Dery et al.  (2021). Auxiliary task update decomposition: The good, the bad and the neutral. *ICLR.* [paper](https://arxiv.org/abs/2108.11346)
- Chen et al. (2021). Weighted training for cross-task learning. *ICLR.* [paper](https://arxiv.org/abs/2105.14095)

Predicted probabilities between tasks

- Nguyen et al (2020). Leep: A new measure to evaluate transferability of learned representations. *ICML.* [Paper](https://proceedings.mlr.press/v119/nguyen20b.html)

- Identifying beneficial task relations for multi-task learning in deep neural networks. EACL 2017. [paper](https://www.aclweb.org/anthology/E17-2026.pdf) 

Task affinity

- Standley et al. (2020). Which tasks should be learned together in multi-task learning? *ICML*. [paper](https://proceedings.mlr.press/v119/standley20a.html)
- Fifty et al. (2021). Efficiently identifying task groupings for multi-task learning. *NeuIPS.* [Paper](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)

### Multitask Learning Architectures

**Mixture-of-Experts**

- Ma et al. (2018). Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *KDD*. [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)

**Branching**

- Guo et al. (2020). Learning to branch for multi-task learning. In *ICML*. [paper](https://arxiv.org/abs/2006.01895)
- Huang, et al. (2018). Gnas: A greedy neural architecture search method for multi-attribute learning. In *ACM MM*. 
- Ruder et al. (2019). Latent multi-task architecture learning. In *AAAI*. 

**Soft-parameter sharing**

- Liu et al. (2019). End-to-end multi-task learning with attention. In CVPR. [paper](https://arxiv.org/abs/1803.10704)

- Cross-stitch Networks for Multi-task Learning. CVPR 2016. [paper](https://arxiv.org/abs/1604.03539)

- Gated multi-task network for text classification. NAACL 2018. [paper](https://www.aclweb.org/anthology/N18-2114.pdf)
- A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks. EMNLP 2017. [paper](https://arxiv.org/pdf/1611.01587)
- End-to-End Multi-Task Learning with Attention. CVPR 2019. [paper](https://arxiv.org/abs/1803.10704)
- Latent Multi-task Architecture Learning. AAAI 2019. [paper](https://arxiv.org/abs/1705.08142)
- Learning Multiple Tasks with Multilinear Relationship Networks. NIPS 2017. [paper](https://arxiv.org/abs/1506.02117)

### Optimization Methods for Multi-Task Learning

- Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR 2018. [paper](https://arxiv.org/pdf/1705.07115)

### Benchmarks

- [GLUE](https://gluebenchmark.com/): Natural Language Understanding
- [decaNLP](https://decanlp.com/): 10 NLP Tasks

### Softwares and Open-source Libraries 

- [LibMTL]( https://github.com/median-research-group/LibMTL): an open-source library built on PyTorch for mulitask learning. 

## Meta Learning

### Survey

Meta-Learning in Neural Networks: A Survey. [paper](https://arxiv.org/pdf/2004.05439.pdf)

### Black-Box Approaches

#### Recurrent Neural Network

(MANN) Meta-learning with memory-augmented neural networks. ICML 2016. [paper](http://proceedings.mlr.press/v48/santoro16.pdf)

#### Attention-Based Network

Matching Networks for One-Shot Learning. NIPS 2016. [paper](https://arxiv.org/abs/1606.04080)

(SNAIL)  A Simple Neural Attentive Meta-Learner. ICLR 2018. [paper](https://arxiv.org/pdf/1707.03141.pdf)

### Optimization-Based Methods

(MAML) Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017. [paper](https://arxiv.org/pdf/1703.03400.pdf) :star:

(Reptile; First-order method) On First-Order Meta-Learning Algorithms. arXiv 2018. [paper](https://arxiv.org/pdf/1803.02999.pdf)

#### Other Forms of Prior on MAML

(Implicit MAML) Meta-Learning with Implicit Gradients. NIPS 2019. [paper](https://arxiv.org/abs/1909.04630)

(Implicit Differentiation; SVM) Meta-Learning with Differentiable Convex Optimization. CVPR 2019. [paper](https://arxiv.org/abs/1904.03758)

(Bayesian linear regression) Meta-Learning Priors for Efficient Online Bayesian Regression. Workshop on the Algorithmic Foundations of Robotics 2018. [paper](https://arxiv.org/pdf/1807.08912)

(Ridge regression; Logistic regression) Meta-learning with Differentiable Closed-Form Solvers. ICLR 2019. [paper](https://openreview.net/pdf?id=HyxnZh0ct7) 

#### Understanding MAML

(MAML expressive power and university) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

(Map MAML to Bayes Framework) Recasting Gradient-Based Meta-Learning as Hierarchical Bayes. ICLR 2018. [paper](https://openreview.net/pdf?id=BJ_UL-k0b) 

#### Tricks to Optimize MAML

**Choose architecture that is effective for inner gradient-step**

Auto-Meta: Automated Gradient Based Meta Learner Search. NIPS 2018 Workshop on Meta-Learning. [paper](https://arxiv.org/pdf/1806.06927) 

**Automatically learn inner vector learning rate, tune outer learning rate**

Alpha MAML: Adaptive Model-Agnostic Meta-Learning. ICML 2019 Workshop on Automated Machine Learning. [paper](https://arxiv.org/pdf/1905.07435) 

Meta-SGD: Learning to Learn Quickly for Few-Shot Learning. arXiv 2017. [paper](https://arxiv.org/pdf/1707.09835.pdf)

**Optimize only a subset of the parameters in the inner loop**

(DEML) Deep Meta-Learning: Learning to Learn in the Concept Space. arXiv 2018. [paper](https://arxiv.org/pdf/1802.03596)

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

**Decouple inner learning rate, BN statistics per-step**

(MAML++) How to train your MAML. ICLR 2019. [paper](https://arxiv.org/pdf/1810.09502)

**Introduce context variables for increased expressive power**

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

(Bias transformation) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

## Non-Parametric Methods via Metric Learning

Siamese Neural Networks for One-shot Image Recognition. ICML 2015. [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Matching Networks for One Shot Learning. NIPS 2016. [paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

Prototypical Networks for Few-shot Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.05175)

**Learn non-linear relation module on embeddings**

Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018. [paper](https://arxiv.org/pdf/1711.06025)

**Learn infinite mixture of prototypes**

Infinite Mixture Prototypes for Few-Shot Learning. ICML 2019. [paper](https://arxiv.org/pdf/1902.04552.pdf)

**Perform message passing on embeddings**

Few-Shot Learning with Graph Neural Networks ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

## Bayesian Meta-Learning & Generative Models

#### Amortized Inference

Amortized Bayesian Meta-Learning. ICLR 2019. [paper](https://openreview.net/pdf?id=rkgpy3C5tX)

#### Ensemble Method

Bayesian Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.03836)

#### Sampling & Hybrid Inference

Probabilistic Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.02817)

Meta-Learning Probabilistic Inference for Prediction. ICLR 2019. [paper](https://openreview.net/pdf?id=HkxStoC5F7)

### Hybrid meta-learning approaches

Meta-Learning with Latent Embedding Optimization. ICLR 2019. [paper](https://arxiv.org/pdf/1807.05960)

Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples. ICLR 2020. [paper](https://arxiv.org/pdf/1903.03096)

Few-Shot Learning with Graph Neural Networks. ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

(CAML) Learning to Learn with Conditional Class Dependencies. ICLR 2019. [paper](https://openreview.net/pdf?id=BJfOXnActQ)

## Meta Reinforcement Learning

#### Policy Gradient RL

**MAML and Black-Box Meta Learning Approaches can be directly applied to Policy-Gradient RL methods**

#### Value-Based RL

**It is not easy to applied existing meta learning approaches to Value-Based RL because Value-Based RL is dynamic programming method** 

Meta-Q-Learning. ICLR 2020. [paper](https://openreview.net/pdf?id=SJeD3CEFPH)

(Goal-Conditioned RL with hindsight relabeling)/(Multi-Task RL) Hindsight Experience Replay. NIPS 2017. [paper](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf) 

(better learning) Learning Latent Plans from Play. CoRL 2019. [paper](https://arxiv.org/pdf/1903.01973)

(learn a better goal representation) 

Universal Planning Networks. ICML 2018. [paper](http://proceedings.mlr.press/v80/srinivas18b/srinivas18b.pdf)

Unsupervised Visuomotor Control through Distributional Planning Networks. RSS 2019. [paper](https://arxiv.org/pdf/1902.05542.pdf)

## Applications

Meta-Learning for Low-Resource Neural Machine Translation. EMNLP 2018. [paper](https://arxiv.org/pdf/1808.08437)

Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions. ICLR 2018. [paper](https://arxiv.org/pdf/1710.10304)

One-Shot Imitation Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.07326.pdf)

Massively Multitask Networks for Drug Discovery. ICML 2015. [paper](https://arxiv.org/pdf/1502.02072.pdf)

