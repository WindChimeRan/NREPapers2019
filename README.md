# Relation Extraction in 2019

This repo covers almost all the papers related to **Neural Relation Extraction** in ACL, EMNLP, COLING, NAACL, AAAI, IJCAI in 2019.

:star: is the recommended papers.

Use tags to search papers you like.

**tags: |NRC | DSRE | PGM | Combining Direct Supervision | GNN | new perspective | new dataset | joint extraction of relations and entities | few shot | BERT | path | imbalance | trick | KBE | RL | cross bag | ML | GAN | false negative | ... |**

DSRE: Distant Supervised Relation Extraction

NRC: Neural Relation Classification

KBE: Knowledge Base Embedding

RL: Reinforcement Learning

## NAACL 2019

1. **Structured Minimally Supervised Learning for Neural Relation Extraction**
   _Fan Bai and Alan Ritter_
   NAACL 2019
   [paper](https://arxiv.org/pdf/1904.00118.pdf)
   [code](https://github.com/bflashcp3f/PCNN-NMAR)

   | PGM | DSRE |

   > This paper adds a PGM inference into training stage.

2. **Combining Distant and Direct Supervision for Neural Relation Extraction**
   _Iz Beltagy, Kyle Lo and Waleed Ammar_
   NAACL 2019
   [paper](https://arxiv.org/pdf/1810.12956.pdf)
   [code](https://github.com/allenai/comb_dist_direct_relex)

   | Combining Direct Supervision | DSRE |

   > This paper combines direct supervision and distant supervision. It innovatively uses direct supervision for training sigmoid attention in a multi-task way. Further, when applying to the CNN backbone with different filter sizes, adding entity embedding as additional inputs is a useful trick, which performs comparable to RESIDE and better than PCNN-ATT. After combining the supervised sigmoid attention, this paper become a new sota.   

3. **Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions**
    _Ye, Zhi-Xiu  and Ling, Zhen-Hua_
    NAACL 2019
    [paper](https://www.aclweb.org/anthology/N19-1288)
    [code](https://github.com/ZhixiuYe/Intra-Bag-and-Inter-Bag-Attentions)

    | DSRE | cross bag |

    >In this paper, we have proposed a neural network
    with intra-bag and inter-bag attentions to cope
    with the noisy sentence and noisy bag problems
    in distant supervision relation extraction. First,
    relation-aware bag representations are calculated
    by a weighted sum of sentence embeddings where
    the noisy sentences are expected to have smaller
    weights. Further, an inter-bag attention module is
    designed to deal with the noisy bag problem by
    2818
    calculating the bag-level attention weights dynamically
    during model training. 

4. **A Richer-but-Smarter Shortest Dependency Path  with Attentive Augmentation for Relation Extraction**
    _Duy-Cat Can, Hoang-Quynh Le, Quang-Thuy Ha, Nigel Collier_
    NAACL 2019 [paper](https://www.aclweb.org/anthology/N19-1298) [code](https://github.com/catcd/RbSP) (code not available on 2019/06/28)

    | path | NRC |

    >In this paper, we have presented RbSP, a novel representation
    of relation between two nominals in a
    sentence that overcomes the disadvantages of traditional
    SDP. Our RbSP is created by using multilayer
    attention to choose relevant information to
    augment a token in SDP from its child nodes.

5. **Connecting Language and Knowledge with Heterogeneous Representations for Neural Relation Extraction**
    _Peng Xu and Denilson Barbosa_ NAACL 2019 [paper](https://arxiv.org/abs/1903.10126) [code](https://github.com/billy-inn/HRERE)

    | KBE | DSRE |

    > Multi-task learning: DSRE + KBE

6. :star: **GAN Driven Semi-distant Supervision for Relation Extraction**
   _Pengshuai Li, Xinsong Zhang, Weijia Jia, Hai Zhao_
   NAACL2019 [paper](https://www.aclweb.org/anthology/N19-1307)

   | GAN | DSRE | false negative |
   >To alleviate the effect of false negative instances, there are two possible ways. One is improving the accuracy of the automatically labeled dataset, and the other is properly leveraging unlabeled instances which cannot be labeled as positive or negative

   >We assume that if an entity is relevant to another entity,
    its name is possibly mentioned in the description of the other entity.

## ACL 2019

1. :star: **Graph Neural Networks with Generated Parameters for Relation**
    _Hao Zhu and Yankai Lin and Zhiyuan Liu, Jie Fu, Tat-seng Chua, Maosong Sun_
    ACL 2019
    [paper](https://arxiv.org/pdf/1902.00756.pdf)

    | GNN | new task | new perspective 

    > This paper considers multi-hop relation extraction, which constructs a fully-connected graph for all entities in a sentence. Experiments show that modeling entity-relation as a graph signifcantly improves the performance. 
2. :star: **Entity-Relation Extraction as Multi-turn Question Answering**
   _Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li
    Arianna Yuan, Duo Chai, Mingxin Zhou and Jiwei Li_
    ACL2019
    [paper](https://arxiv.org/abs/1905.05529)

    | new dataset | new perspective| joint extraction of relations and entities

    > In this paper, we propose a new paradigm for
the task of entity-relation extraction. We cast
the task as a multi-turn question answering
problem, i.e., the extraction of entities and relations
is transformed to the task of identifying
answer spans from the context. This multi-turn
QA formalization comes with several key advantages:
firstly, the question query encodes
important information for the entity/relation
class we want to identify; secondly, QA provides
a natural way of jointly modeling entity
and relation; and thirdly, it allows us to exploit
the well developed machine reading comprehension
(MRC) models.
    > Additionally, we construct a newly developed
    dataset RESUME in Chinese, which requires
    multi-step reasoning to construct entity dependencies,
    as opposed to the single-step dependency
    extraction in the triplet exaction in previous
    datasets.

3. :star: **Matching the Blanks: Distributional Similarity for Relation Learning**
   _Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, Tom Kwiatkowski_
   ACL2019
   [paper](https://arxiv.org/pdf/1906.03158.pdf)

   | few shot | sota | BERT |

   > In this paper we study the problem of producing
useful relation representations directly from text.
We describe a novel training setup, which we call
matching the blanks, which relies solely on entity
resolution annotations. When coupled with
a new architecture for fine-tuning relation representations in BERT, our models achieves state-of-the-art results on three relation extraction tasks, and outperforms human accuracy on few-shot relation matching. In addition, we show how the new model is particularly effective in low-resource regimes, and we argue that it could significantly reduce the amount of human effort required to create relation extractors.


4. **Exploiting Entity BIO Tag Embeddings and Multi-task Learning for Relation Extraction with Imbalanced Data**
   _Wei Ye1*, Bo Li*, Rui Xie, Zhonghao Sheng, Long Chen and Shikun Zhang1_ ACL2019 [paper](https://arxiv.org/pdf/1906.08931.pdf)

   | NRC | imbalance | trick | ranking loss |

   > This paper detailed analyze the impact of imbalanced data (other relation) to the final performance. Incorporating BIO tagging to the embedding layer is an important trick!

5. **GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction**
    _Tsu-Jui Fu, Peng-Hsuan Li and Wei-Yun Ma_ ACL2019 [paper](https://tsujuifu.github.io/pubs/acl19_graph-rel.pdf)

    | joint extraction of relations and entities | sota |

    > They use GCN and argue they are the new sota.


## AAAI 2019

1. **Hybrid Attention-based Prototypical Networks for Noisy Few-Shot Relation Classification**
   _Tianyu Gao*, Xu Han*, Zhiyuan Liu, Maosong Sun. (* means equal contribution)_
   AAAI2019 [paper](https://gaotianyu1350.github.io/assets/aaai2019_hatt_paper.pdf) [code](https://github.com/thunlp/HATT-Proto)

   | few shot |

   > instance level and feature-level attention schemes based on prototypical networks

2. **A Hierarchical Framework for Relation Extraction with Reinforcement Learning**
   _Takanobu, Ryuichi and Zhang, Tianyang and Liu, Jiexi and Huang, Minlie_
   AAAI2019 [paper](https://arxiv.org/pdf/1811.03925.pdf) [code](https://github.com/truthless11/HRL-RE)

   | joint extraction of relations and entities | RL |  

   > This paper use RL based multi-pass tagging to tackle relation overlapping problem.

3. **Kernelized Hashcode Representations for Biomedical Relation Extraction**
   _Sahil Garg, Aram Galstyan, Greg Ver Steeg Irina Rish, Guillermo Cecchi, Shuyang Gao_
   AAAI2019
   [paper](https://arxiv.org/pdf/1711.04044.pdf) [code](https://github.com/sgarg87/HFR) code not released on 07/05/2019

   | ML |

   >Here we propose to use random subspaces of KLSH codes for efficiently constructing an explicit representation of NLP structures suitable for general classification methods. Further, we propose an approach for optimizing the KLSH model for classification problems by maximizing an approximation of mutual information between the KLSH codes (feature vectors) and the class labels


4. **Cross-relation Cross-bag Attention for Distantly-supervised Relation Extraction**
   _Yujin Yuan, Liyuan Liu, Siliang Tang, Zhongfei Zhang, Yueting Zhuang, Shiliang Pu, Fei Wu, Xiang Ren_
   AAAI2019
   [paper](https://arxiv.org/pdf/1812.10604.pdf)

   | DSRE | cross bag |

   > see title

