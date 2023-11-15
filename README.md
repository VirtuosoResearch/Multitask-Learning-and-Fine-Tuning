## Benchmarks for Reasoning Abilities of Large Language Models

**Benchmarks:**

- GSM8k [(Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168): Arithmetic reasoning with grade school problems and natural language descripts. Problems include arithmic operations: addition, subtraction, multiplication, and division.

- MATH (Hendrycks et al., 2021): 
- MMLU (Hendrycks et al., 2020)
- Big-Bench-Hard ()
- HumanEval ()
- TheoremQA
- SummEdits

|           | Task Type                                                    | Construction     | # Problems | # Problem Types | Problems                                         | Prompt style                                                 |
| --------- | ------------------------------------------------------------ | ---------------- | ---------- | --------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| GSM8k     | Arithmic reasoning of math computation steps using language  | Manually written | 8,500      | 4               | addition, subtraction, multiplication,  division | Multi-step reasoing (Similar to chain-of-thoughts):  The problems take between 2 and 8 steps to solve, as described by natural languages. |
| MATH      | Math problems from mathematics competitions                  |                  | 12,000     | 7               |                                                  |                                                              |
| MMLU      | High-school and college-level common knowledge               |                  | 15,000     | 57              |                                                  |                                                              |
| BBH       | Language and symbolic reasoning                              |                  | 6,500      | 23              |                                                  |                                                              |
| HumanEval | Python programming problems with text comments and docstrings test cases | Manually written | 164        |                 |                                                  |                                                              |
| TheoremQA |                                                              |                  |            |                 |                                                  |                                                              |
| SummEdits |                                                              |                  |            |                 |                                                  |                                                              |

**Examples.**

|       | Example                                                      |
| ----- | ------------------------------------------------------------ |
| GSM8K | **Question:** Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?<br/>**Let's think step by step**<br/>Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.<br/>For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.<br/>Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.<br/>However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.<br/>They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.<br/>And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.<br/>So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.<br/>They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75<br/>They will need to plan to study 4 days to allow for all the time they need.<br/>**The answer** is 4 |



 

**Concepts:**

- Fine-tuning: Use the language modeling objective to further train a pretrained language model. 
- Verification: First train a generator by question-solution pairs. Then, sample multiple generated solutions, assign each solution a score (binary scores of whether the solution leads to the correct answer), and train a model by the scores. A model trained by the verification scores is called verifier. 
  - At test time, we sample solutions to each test problem, rank them with the verifier, and then return the one with the highest verifier score.





**Prompting methods:**

| Prompt strategy                                              |                           Prompts                            | In-context examples                 |            Prompt generation            |                GSM8k                 |      |
| ------------------------------------------------------------ | :----------------------------------------------------------: | ----------------------------------- | :-------------------------------------: | :----------------------------------: | ---- |
|                                                              |                                                              |                                     |                                         | Metric: Solve rate (%); Model: Codex |      |
| Scratchpad (Nye et al., 2021)                                | Break a code function down and ask the model to output all intermediate steps of the code | (input, intermediate steps, output) | Manually designed based on an algorithm |                  ?                   |      |
| Chain-of-though prompting ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)) | Prompt the model with the rationale in solving a multi-step reasoning problem. | (input, chain-of-thought, output)   |            Manually written             |                 63.1                 |      |
| Algotihmic prompting ([Zhou et al., 2022](https://arxiv.org/abs/2211.09066)) | Prompt the model with detailed rationales, including describing the steps within an algorithm. | (input, algorithmic prompt, output) |            Manually written             |                 82.7                 |      |
|                                                              |                                                              |                                     |                                         |                                      |      |



<!--**Concepts:** Verifier Prompting (In-context learning): prompt a model with a few input--output exemplars demonstrating the task.-->




## Multi-Task Learning

End-to-End Multi-Task Learning with Attention. CVPR 2019. [paper](https://arxiv.org/abs/1803.10704)

Latent Multi-task Architecture Learning. AAAI 2019. [paper](https://arxiv.org/abs/1705.08142)

Cross-stitch Networks for Multi-task Learning. CVPR 2016. [paper](https://arxiv.org/abs/1604.03539)

Learning Multiple Tasks with Multilinear Relationship Networks. NIPS 2017. [paper](https://arxiv.org/abs/1506.02117)

More multitask learning papers [here](https://github.com/lidongyue12138/Multi-Task-and-Meta-Learning/blob/master/Multitask%20Learning%20Papers%20Review.md) 

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

### Non-Parametric Methods via Metric Learning

Siamese Neural Networks for One-shot Image Recognition. ICML 2015. [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Matching Networks for One Shot Learning. NIPS 2016. [paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

Prototypical Networks for Few-shot Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.05175)

**Learn non-linear relation module on embeddings**

Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018. [paper](https://arxiv.org/pdf/1711.06025)

**Learn infinite mixture of prototypes**

Infinite Mixture Prototypes for Few-Shot Learning. ICML 2019. [paper](https://arxiv.org/pdf/1902.04552.pdf)

**Perform message passing on embeddings**

Few-Shot Learning with Graph Neural Networks ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

### Bayesian Meta-Learning & Generative Models

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

### Meta Reinforcement Learning

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

### Application

Meta-Learning for Low-Resource Neural Machine Translation. EMNLP 2018. [paper](https://arxiv.org/pdf/1808.08437)

Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions. ICLR 2018. [paper](https://arxiv.org/pdf/1710.10304)

One-Shot Imitation Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.07326.pdf)

Massively Multitask Networks for Drug Discovery. ICML 2015. [paper](https://arxiv.org/pdf/1502.02072.pdf)

