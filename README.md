# Relation Extraction in 2019

This repo covers almost all the papers (35) related to **Neural Relation Extraction** in ACL, EMNLP, COLING, NAACL, AAAI, IJCAI in 2019.

Use tags to search papers you like.

**tags: DSRE | PGM | Combining Direct Supervision | ...**

## NAACL 2019

1. **Structured Minimally Supervised Learning for Neural Relation Extraction**
   _Fan Bai and Alan Ritter_
   NAACL 2019
   [paper](https://arxiv.org/pdf/1904.00118.pdf)
   [code](https://github.com/bflashcp3f/PCNN-NMAR)

   PGM | DSRE

   > This paper adds a PGM inference into training stage.

2. **Combining Distant and Direct Supervision for Neural Relation Extraction**
   _Iz Beltagy, Kyle Lo and Waleed Ammar_
   NAACL 2019
   [paper](https://arxiv.org/pdf/1810.12956.pdf)
   [code](https://github.com/allenai/comb_dist_direct_relex)

   Combining Direct Supervision | DSRE

   > This paper combines direct supervision and distant supervision. It innovatively uses direct supervision for training sigmoid attention in a multi-task way. Further, when applying to the CNN backbone with different filter sizes, adding entity embedding as additional inputs is a useful trick, which performs comparable to RESIDE and better than PCNN-ATT. After combining the supervised sigmoid attention, this paper become a new sota.   