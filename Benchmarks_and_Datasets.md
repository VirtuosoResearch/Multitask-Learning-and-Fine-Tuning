# Benchmarks and Datasets

### Benchmarks for Reasoning Abilities of Large Language Models

- GSM8k [(Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168): Arithmetic reasoning with grade school problems and natural language descripts. Problems include arithmic operations: addition, subtraction, multiplication, and division.

- MATH [(Hendrycks et al., 2021)](https://arxiv.org/abs/2103.03874)
- MMLU [(Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300)
- Big-Bench-Hard [(Suzgun et al., 2022)](https://arxiv.org/abs/2210.09261)
- HumanEval [(Chen et al., 2021)](https://arxiv.org/abs/2107.03374)
- TheoremQA
- SummEdits

|           | Task Type                                                    | Construction                           | # Problems | # Problem Types | Problems                                         | Prompt style                                                 |
| --------- | ------------------------------------------------------------ | -------------------------------------- | ---------- | --------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| GSM8k     | Arithmic reasoning of math computation steps using language  | Manually written language descriptions | 8,500      | 4               | addition, subtraction, multiplication,  division | Multi-step reasoing (Similar to chain-of-thoughts):  The problems take between 2 and 8 steps to solve, as described by natural languages. |
| MATH      | Math problems from mathematics competitions                  |                                        | 12,000     | 7               |                                                  | Multi-step reasoing (Similar to chain-of-thoughts)           |
| MMLU      | High-school and college-level common knowledge               |                                        | 15,000     | 57              |                                                  | Multiple-choice; Few-show examples.                          |
| BBH       | Language and symbolic reasoning                              |                                        | 6,500      | 23              |                                                  | Few shot chain-of-thought exemplars.                         |
| HumanEval | Python programming problems with text comments and docstrings test cases | Manually written programs              | 164        |                 |                                                  |                                                              |
| TheoremQA |                                                              |                                        |            |                 |                                                  |                                                              |
| SummEdits |                                                              |                                        |            |                 |                                                  |                                                              |

**Examples.**

|           | Example                                                      |
| --------- | ------------------------------------------------------------ |
| GSM8K     | **Question:** Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?<br/>**Let's think step by step**<br/>Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.<br/>For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.<br/>Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.<br/>However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.<br/>They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.<br/>And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.<br/>So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.<br/>They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75<br/>They will need to plan to study 4 days to allow for all the time they need.<br/>**The answer** is 4 |
| MATH      | **Question:** The sum of two numbers is 6. The difference of their squares is 12. What is the positive difference of the two numbers?<br/>**Let's think step by step**<br/>Call the two numbers $x$ and $y$. <br/>We are given that $x+y = 6$ and $x^2 - y^2 = 12$. <br/>Because $x^2 - y^2$ factors into $(x+y)(x-y)$, <br/>we can substitute in for $x+y$, <br/>giving $6(x-y) = 12$, <br/>or $x-y = \boxed{2}$.<br/>**The answer** is 2 |
| MMLU      | The following are multiple choice questions (with answers) about  abstract algebra.<br/>Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field. <br/> A. 0<br/> B. 1<br/> C. 2<br/> D. 3<br/> **Answer:** B<br/> Statement 1: Every function from a finite set onto itself must be one to one. Statement 2: Every subgroup of an abelian group is abelian. <br/> A. True, True <br/> B. False, False <br/> C. True, False <br/>  D. False, True <br/> **Answer:** A <br/> Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.<br/> A. 0<br/> B. 4<br/> C. 2<br/> D. 6<br/> **Answer:** B |
| BBH       | **Input:** "If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right. Options: Yes or No",<br/>**Let's think step by step**<br/>**Target:** "No" |
| HumanEval | **Prompt:** ```def incr_list(l: list):<br/>	"""Return list with elements incremented by 1. <br/>	>>> incr_list([1, 2, 3])<br/>	[2, 3, 4]<br/>	>>> incr_list([5, 3, 5, 2, 3, 3, 9, 0])<br/>	[6, 4, 6, 3, 4, 4, 10, 1]<br/>	""" ```<br />**Output:** ```return [i+1 for i in l]``` |
|           |                                                              |
|           |                                                              |

 

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

