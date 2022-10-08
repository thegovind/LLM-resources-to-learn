# LLM resources to learn

---
## Overview

  - [Applications](#applications)
    - [Text related](#generative-text)
  - [Robotics](#robotics)
    - [Reasoning](#reasoning)
    - [Planning](#planning)
    - [Manipulation](#manipulation)
    - [Instructions and Navigation](#instructions-and-navigation)
    - [Simulation Frameworks](#simulation-frameworks)
  - [Citation](#citation)
 

----
## Applications
### Text related
* **Copywriting**, **Summarization**, **Parsing**, **Classification**, **Translation**: GPT-3 - "Language Models are Few-Shot Learners" [[Paper](https://arxiv.org/abs/2005.14165v4)][[Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/openai/Fine_tune_GPT_3_with_Weights_%26_Biases.ipynb)][[Website](https://beta.openai.com/docs/introduction)]
----
## Robotics
### Reasoning

 * **Code-As-Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sept 2022*. [[Paper](https://arxiv.org/abs/2209.07753)]  [[Colab](https://github.com/google-research/google-research/tree/master/code_as_policies)] [[Website](https://code-as-policies.github.io/)] 

 * **Say-Can**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.01691)]  [[Colab](https://say-can.github.io/#open-source)] [[Website](https://say-can.github.io/)] 

 * **Socratic**: "Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.00598)] [[Pytorch Code](https://socraticmodels.github.io/#code)] [[Website](https://socraticmodels.github.io/)]

 * **PIGLeT**: "PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World", *ACL, Jun 2021*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](http://github.com/rowanz/piglet)] [[Website](https://rowanzellers.com/piglet/)]

---
### Planning

 * **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Pytorch Code](https://github.com/blazejosinski/lm_nav)] [[Website](https://sites.google.com/view/lmnav)]

 * **InnerMonlogue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Website](https://innermonologue.github.io/)]

 * **Housekeep**: "Housekeep: Tidying Virtual Households using Commonsense Reasoning", *arXiv, May 2022*. [[Paper](https://arxiv.org/abs/2205.10712)] [[Pytorch Code](https://github.com/yashkant/housekeep)] [[Website](https://yashkant.github.io/housekeep/)]

 * **LID**: "Pre-Trained Language Models for Interactive Decision-Making", *arXiv, Feb 2022*. [[Paper](https://arxiv.org/abs/2202.01771)] [[Pytorch Code](https://github.com/ShuangLI59/Language-Model-Pre-training-Improves-Generalization-in-Policy-Learning)] [[Website](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)]

 * **ZSP**: "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents", *ICML, Jan 2022*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](https://github.com/huangwl18/language-planner)] [[Website](https://wenlong.page/language-planner/)]

---
### Manipulation

* **VIMA**:"VIMA: General Robot Manipulation with Multimodal Prompts", "arXiv, Oct 2022", [[Paper](https://arxiv.org/abs/2210.03094)] [[Pytorch Code](https://github.com/vimalabs/VIMA)] [[Website](https://vimalabs.github.io/)]

* **Perceiver-Actor**:"A Multi-Task Transformer for Robotic Manipulation", *CoRL, Sep 2022*. [[Paper](https://peract.github.io/paper/peract_corl2022.pdf)] [[Pytorch Code](https://github.com/peract/peract)] [[Website](https://peract.github.io/)]

 * **LaTTe**: "LaTTe: Language Trajectory TransformEr", *arXiv, Aug 2022*. [[Paper](https://arxiv.org/abs/2208.02918)] [[TensorFlow Code](https://github.com/arthurfenderbucker/NL_trajectory_reshaper)] [[Website](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/robot-language/)]

 * **ATLA**: "Leveraging Language for Accelerated Learning of Tool Manipulation", *CoRL, Jun 2022*. [[Paper](https://arxiv.org/abs/2206.13074)]

 * **ZeST**: "Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?", *L4DC, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.11134)]

 * **LSE-NGU**: "Semantic Exploration from Language Abstractions and Pretrained Representations", *arXiv, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.05080)]

 * **Embodied-CLIP**: "Simple but Effective: CLIP Embeddings for Embodied AI ", *CVPR, Nov 2021*. [[Paper](https://arxiv.org/abs/2111.09888)] [[Pytorch Code](https://github.com/allenai/embodied-clip)]

 * **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL, Sept 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Pytorch Code](https://github.com/cliport/cliport)] [[Website](https://cliport.github.io/)]

---
### Instructions and Navigation

 * **ADAPT**: "ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts", *CVPR, May 2022*. [[Paper](https://arxiv.org/abs/2205.15509)] 

 * "The Unsurprising Effectiveness of Pre-Trained Vision Models for Control", *ICML, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.03580)] [[Pytorch Code](https://github.com/sparisi/pvr_habitat)] [[Website](https://sites.google.com/view/pvr-control)]

 * **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.10421)] 
 
 * **Recurrent VLN-BERT**: "A Recurrent Vision-and-Language BERT for Navigation", *CVPR, Jun 2021* [[Paper](https://arxiv.org/abs/2011.13922)] [[Pytorch Code](https://github.com/YicongHong/Recurrent-VLN-BERT)]
 
 * **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV, Apr 2020* [[Paper](https://arxiv.org/abs/2004.14973)] [[Pytorch Code](https://github.com/arjunmajum/vln-bert)]

---
### Simulation Frameworks

 * **MineDojo**: "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge", *arXiv, Jun 2022*. [[Paper](https://arxiv.org/abs/2206.08853)] [[Code](https://github.com/MineDojo/MineDojo)] [[Website](https://minedojo.org/)] [[Open Database](https://minedojo.org/knowledge_base.html)]
 * **Habitat 2.0**: "Habitat 2.0: Training Home Assistants to Rearrange their Habitat", *NeurIPS, Dec 2021*. [[Paper](https://arxiv.org/abs/2106.14405)] [[Code](https://github.com/facebookresearch/habitat-sim)] [[Website](https://aihabitat.org/)]
 * **BEHAVIOR**: "BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments", *CoRL, Nov 2021*. [[Paper](https://arxiv.org/abs/2108.03332)] [[Code](https://github.com/StanfordVL/behavior)] [[Website](https://behavior.stanford.edu/)]
 * **iGibson 1.0**: "iGibson 1.0: a Simulation Environment for Interactive Tasks in Large Realistic Scenes", *IROS, Sep 2021*. [[Paper](https://arxiv.org/abs/2012.02924)] [[Code](https://github.com/StanfordVL/iGibson)] [[Website](https://svl.stanford.edu/igibson/)]
 * **ALFRED**: "ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks", *CVPR, Jun 2020*. [[Paper](https://arxiv.org/abs/1912.01734)] [[Code](https://github.com/askforalfred/alfred)] [[Website](https://askforalfred.com/)]

 

----
### Citation
```
@misc{kira2022llmroboticspaperslist,
    title = {Awesome-LLM-Robotics},
    author = {Zsolt Kira},
    journal = {GitHub repository},
    url = {https://github.com/GT-RIPL/Awesome-LLM-Robotics},
    year = {2022},
}
```
