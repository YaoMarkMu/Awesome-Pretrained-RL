# Pretrained Model for Decision Making and Control

This is a list of papers for **Decision Making Pretrained Model**.
And the repository will be continuously updated to track the frontier researches.

Welcome to follow and star!



## Learn Transferable Policies and Representations with Transformer

- [CtrlFormer: Learning Transferable State Representation for Visual Control via Transformer](https://arxiv.org/abs/2206.08883?context=cs.LG)
  - Yao(Mark) Mu, Shoufa Chen, Mingyu Ding, Jianyu Chen, Runjian Chen, Ping Luo
  - Publisher: ICLM 2022 (Spotlight)
  - Key: Reinforcement Learning, Transfer Learning, Representation Learning
  - Code: [official](https://github.com/YaoMarkMu/CtrlFormer_robotic), [simplified single task version](https://github.com/YaoMarkMu/ViT4RL)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py)
  
- [AnyMorph: Learning Transferable Polices By Inferring Agent Morphology](https://arxiv.org/abs/2206.12279)
  - Brandon Trabucco, Mariano Phielipp, Glen Berseth
  - Publisher: ICML 2022 (Spotlight)
  - Key: Morphology, Transfer Learning, Zero Shot
  - ExpEnv: [Modular-RL](https://github.com/huangwl18/modular-rl)

- [Silver-Bullet-3D at ManiSkill 2021: Learning-from-Demonstrations and Heuristic Rule-based Methods for Object Manipulation](https://arxiv.org/abs/2206.06289)
  - Yingwei Pan, Yehao Li, Yiheng Zhang, Qi Cai, Fuchen Long, Zhaofan Qiu, Ting Yao, Tao Mei
  - Publisher: ICLR 2022 (GPL Workshop Poster)
  - Key: Object Manipulation
  - Code: [official](https://github.com/caiqi/Silver-Bullet-3D/)
  - ExpEnv: [ManiSkill](https://github.com/haosulab/ManiSkill)

- [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/abs/2203.07413)
  - Qinjie Lin, Han Liu, Biswa Sengupta
  - Publisher: Arxiv(preprint)
  - Key: Multi-Task RL, Sparse Reward
  - ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid)



## Decision Transformer
### Single Agent DT

- [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)
  - Michael Janner, Qiyang Li, Sergey Levine
  - Publisher: NeurIPS 2021 (Spotlight)
  - Key: Conditional sequence modeling, Discretization
  - Code: [official](https://github.com/JannerM/trajectory-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
  - Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch
  - Publisher: NeurIPS 2021 (Poster)
  - Key: Conditional sequence modeling
  - Code: [official](https://github.com/kzl/decision-transformer), [DI-engine](https://github.com/kzl/decision-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)
   
- [Generalized Decision Transformer for Offline Hindsight Information Matching](https://arxiv.org/abs/2111.10364)
  - Hiroki Furuta, Yutaka Matsuo, Shixiang Shane Gu
  - Publisher: ICLR 2021 (Spotlight)
  - Key: HIM, SMM
  - Code: [official](https://github.com/frt03/generalized_dt)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)



- [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2206.08569)
  - Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, Dongsheng Li
  - Publisher: Arxiv(preprint)
  - Key:  Generation model
  - Code: [official](https://seqml.github.io/bootorl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Adroit](https://github.com/aravindr93/hand_dapg)

- [Online Decision Transformer](https://arxiv.org/abs/2202.05607)
  - Qinqing Zheng, Amy Zhang, Aditya Grover
  - Publisher:  ICML 2022 (Oral)
  - Key: Online finetuning,  Max-entropy, Exploration
  - Code: [unofficial](https://github.com/daniellawson9999/online-decision-transformer)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl)

- [Prompting Decision Transformer for Few-Shot Policy Generalization](https://arxiv.org/abs/2206.13499)
  - Mengdi Xu, Yikang Shen, Shun Zhang, Yuchen Lu, Ding Zhao, Joshua B. Tenenbaum, Chuang Gan
  - Publisher:  ICML 2022 (Poster)
  - Key: Prompt, Few-shot, Generalization
  - Code: [official](https://mxu34.github.io/PromptDT/) (released soon)
  - ExpEnv: [DMC](https://github.com/deepmind/dm_control)

- [Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning](https://proceedings.mlr.press/v162/villaflor22a.html)
  - Adam R Villaflor, Zhe Huang, Swapnil Pande, John M Dolan, Jeff Schneider
  - Publisher:  ICML 2022 (Poster)
  - Key: World model
  - Code: [official](https://mxu34.github.io/PromptDT/) (released soon)
  - ExpEnv: [CARLA](https://leaderboard.carla.org/)

- [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241)
  - Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, Igor Mordatch
  - Publisher: Arxiv(preprint)
  - Key: Multi-Task,  Finetuning
  - Code: [official](https://sites.google.com/view/multi-game-transformers)
  - ExpEnv: [Atari](https://github.com/openai/gym), [REM](https://github.com/google-research/batch_rl)

- [Transfer learning with causal counterfactual reasoning in Decision Transformers](https://arxiv.org/abs/2110.14355)
  - Ayman Boustati, Hana Chockler, Daniel C. McNamee
  - Publisher: Arxiv(preprint)
  - Key: Causal reasoning, Transfer Learning
  - ExpEnv: [MINIGRID](https://github.com/Farama-Foundation/gym-minigrid)

- [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/abs/2201.12122)
  - Machel Reid, Yutaro Yamada, Shixiang Shane Gu
  - Key: VLN, Transfer Learning
  - Code: [official](https://github.com/machelreid/can-wikipedia-help-offline-rl)
  - ExpEnv: [MuJoco](https://github.com/openai/mujoco-py), [D4RL](https://github.com/rail-berkeley/d4rl), [Atari](https://github.com/openai/gym)

### Multi-Agent DT
- [Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)
  - Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang
  - Publisher: Arxiv(preprint)
  - Key: Multi-Agent RL
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac), [MA MuJoco](https://github.com/schroederdewitt/multiagent_mujoco)

- [Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks](https://arxiv.org/abs/2112.02845)
  - Linghui Meng, Muning Wen, Yaodong Yang, Chenyang Le, Xiyun Li, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, Bo Xu
  - Publisher: Arxiv(preprint)
  - Key: Multi-Agent RL
  - Code: [official](https://github.com/reinholdm/offline-pre-trained-multi-agent-decision-transformer)
  - ExpEnv: [SMAC](https://github.com/oxwhirl/smac)



## Transformer as a world model

- [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/abs/2202.09481)
  - Chang Chen, Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn
  - Publisher: NeurIPS 2021 (Deep RL Workshop)
  - Key: Dreamer, World Model
  - ExpEnv: Hidden Order Discovery, [DMC](https://github.com/deepmind/dm_control), [Atari](https://github.com/openai/gym)

- [Dreaming with Transformers](http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_24.pdf)
  - Catherine Zeng, Jordan Docter, Alexander Amini, Igor Gilitschenski, Ramin Hasani, Daniela Rus
  - Publisher: AAAI 2022 (RLG Workshop)
  - Key: Dreamer, World Model
  - ExpEnv: [Deepmind Lab](https://github.com/deepmind/lab), [VISTA](https://github.com/vista-simulator/vista)

  
- [Transformers are Sample Efficient World Models](https://arxiv.org/abs/2209.00588)
    - Vincent Micheli, Eloi Alonso, François Fleuret
    - Publisher: Arxiv (Preprint)
    - Key: World Model
    - ExpEnv: [Atari](https://github.com/openai/gym)

## Language-Conditioned Decision Making and Control
- [Pretraining for Language Conditioned Imitation with Transformers](https://openreview.net/forum?id=eCPCn25gat)
  - Aaron L Putterman, Kevin Lu, Igor Mordatch, Pieter Abbeel
  - Key: Text-Conditioned Decision
  - ExpEnv: Text-Conditioned Frostbite (MultiModal Benchmark)
  
- [BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning](https://openreview.net/pdf?id=8kbp23tSGYv)
    - Eric Jang1, Alex Irpan1, Mohi Khansari
    - Publisher: CORL 2021
    - Code: [official](https://sites.google.com/view/bc-z/home)

- [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/pdf/2203.12601.pdf)
    - Suraj Nair, Aravind Rajeswaran, Vikash Kumar 
    - Publisher: CORL 2022
    - Code: [official](https://tinyurl.com/robotr3m)
    
- [Language-Conditioned Imitation Learning for Robot Manipulation Tasks](https://proceedings.neurips.cc/paper/2020/file/9909794d52985cbc5d95c26e31125d1a-Paper.pdf)
    - Simon Stepputtis, Joseph Campbell, Mariano Phielipp
    - Publisher: NeurIPS 2020
    - Key: Language-Conditioned Imitation Learning
    
- [Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation](https://arxiv.org/pdf/2109.01115.pdf)
    - Suraj Nair, Eric Mitchell, Kevin Chen
    - Publisher: CORL 2021


    



## Direct Language Planning via Pretrained Language Model
- [Transformers are Adaptable Task Planners](https://arxiv.org/abs/2207.02442)
  - Vidhi Jain, Yixin Lin, Eric Undersander, Yonatan Bisk, Akshara Rai
  - Key: Task Planning, Prompt, Control, Generalization
  - Code: [official](https://anonymous.4open.science/r/temporal_task_planner-Paper148/README.md)
  - ExpEnv: Dishwasher Loading
  
- [Pre-Trained Language Models for Interactive Decision-Making](https://arxiv.org/pdf/2202.01771.pdf)
     - Shuang Li, Xavier Puig, Chris Paxton, Yilun Du
     - Key: using LMs to scaffold learning and generalization in general sequential decision-making problems 

- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/pdf/2204.01691.pdf)
  - Michael, AhnAnthony BrohanNoah, Brown 
  - Publisher: Arixiv(preprint)


- [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/pdf/2207.04429.pdf)
  - Publisher: Arixiv(preprint)
  - Code: [official](https://sites.google.com/view/lmnav)

- [LaTTe: Language Trajectory Transformer](https://arxiv.org/pdf/2208.02918.pdf)
    - Arthur Bucker, Luis Figueredo, Sami Haddadin
    - Publisher: Arixiv(preprint)
    - Code: [official](https://github.com/arthurfenderbucker/nl_trajectory_reshaper)
    
This repo is based on the [OpenDILab](https://github.com/opendilab)


