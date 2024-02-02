# Multimodal and Unimodal Recommendation Paperlist
This is the sumarized paperlist that has been mentioned in our mutlimodal recommendation survey paper:
>xxx

Moreover, we also include the unimodal recommendation papers and will update the paperlist.

The framework code is avaiable at: https://github.com/enoche/MMRec

## Table of Contents

- [Background](#background)
- [Survey](#survey)
- [Multimodal Rec](#multimodal-rec)
- [Textual Based Rec](#textual-based-rec)
	- [Title Abstract Tag](#title-abstract-tag)
	- [Review Description](#reviews-description) 
- [Visual Based Rec](#visual-based-rec)
- [Evaluation Result](#evaluating-the-sota-models)

## Background
The traditional recommendation system requires a large number of interactions between users and items to make accurate recommendation. The mulitmodal information has been utilized to alleviate the data sparsity problem and cold start issue. The unimodal information has been utilized to enrich the representations and recently, the fused multimodal information has been also leverage to improve the performance accuracy.

We classified the papers according to the modality information they used and list out the paper list according to the publish time.

## Survey

* Recommender Systems Leveraging Multimedia Content [ACM Computing Surveys Sep 2021] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3407190)]

## Multimodal Rec

* Collaborative Knowledge Base Embedding for Recommender Systems [KDD Aug 2016] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/2939672.2939673)]
* Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources [CIKM Nov 2017] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3132847.3132892)] [***github***](https://github.com/evison/JRL)
* User-Video Co-Attention Network for Personalized Micro-video Recommendation [WWW May 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3308558.3313513)]
* Personalized Fashion Recommendation with Visual Explanations based on Multimodal Attention Network [SIGIR Jul 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3331184.3331254)]
* Multimodal Representation Learning for Recommendation in Internet of Things [IEEE internet of things journal Sep 2019] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=8832204)]
* User Diverse Preference Modeling by Multimodal Attentive Metric Learning [MM Oct 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3343031.3350953)] [***github***](https://github.com/liufancs/MAML)
* MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video [MM Oct 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3343031.3351034)] [***github***](https://github.com/weiyinwei/MMGCN)
* Adversarial Training Towards Robust Multimedia Recommender System [TKDE May 2020] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=8618394)] [***github***](https://github.com/duxy-me/AMR)
* MGAT: Multimodal Graph Attention Network for Recommendation [Elsevier Apr 2020] [[__pdf__](https://www.sciencedirect.com.remotexs.ntu.edu.sg/science/article/pii/S0306457320300182?via%3Dihub)] [***github***](https://github.com/zltao/MGAT)
* Multi-modal Knowledge Graphs for Recommender Systems [CIKM Oct 2020] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3340531.3411947)]
* Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback [MM Oct 2020] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3394171.3413556)] [***github***](https://github.com/weiyinwei/GRCN)
* Recommendation by Users’ Multimodal Preferences for Smart City Applications [IEEE transactions on industrial informatics June 2021] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=9152003)] [***github***](https://github.com/winterant/UMPR)
* MULTIMODAL DISENTANGLED REPRESENTATION FOR RECOMMENDATION [ICME Jul 2021] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=9428193)]
* MM-Rec: Multimodal News Recommendation [SIGIR Jul 2021] [[__pdf__](https://arxiv.org/pdf/2104.07407.pdf)]
* Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems [TMM Sep 2021] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=9535249)] [***github***](https://github.com/jing-1/MVGAE)
* Why Do We Click: Visual Impression-aware News Recommendation [MM Oct 2021] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3474085.3475514)] [***github***](https://github.com/JiahaoXun/IMRec)
* Pre-training Graph Transformer with Multimodal Side Information for Recommendation [MM Oct 2021] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3474085.3475709)] [***github***](https://github.com/RuihongQiu/cornac/tree/master/cornac/models/causalrec)
* Mining Latent Structures for Multimedia Recommendation [MM Oct 2021] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3474085.3475259)] [***github***](https://github.com/CRIPAC-DIG/LATTICE)
* DualGNN: Dual Graph Neural Network for Multimedia Recommendation [TMM Dec 2021] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/abstract/document/9662655)] [***github***](https://github.com/wqf321/dualgnn)
* A two-stage embedding model for recommendation with multimodal auxiliary information [Elsevier Jan 2022] [[__pdf__](https://www.sciencedirect.com.remotexs.ntu.edu.sg/science/article/pii/S0020025521009270?via%3Dihub)]
* Disentangled Multimodal Representation Learning for Recommendation [TMM Oct 2022] [[__pdf__](https://arxiv.org/pdf/2203.05406.pdf)]
* Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation [TKDE Nov 2022] [[__pdf__](https://arxiv.org/pdf/2111.00678.pdf)] [***github***](https://github.com/cripac-dig/micro)
* MEGCF: Multimodal Entity Graph Collaborative Filtering for Personalized Recommendation [TOIS May 2022] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3544106)]  [***github***](https://github.com/kangliu1225/MEGCF)
* Self-supervised Learning for Multimedia Recommendation [TMM Jun 2022] [[__pdf__](https://ieeexplore.ieee.org/document/9811387)] [***github***](https://github.com/zltao/SLMRec/)
* Hierarchical User Intent Graph Network for Multimedia Recommendation [TMM Jun 2021] [[__pdf__](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9453189)] [***github***](https://github.com/weiyinwei/HUIGN)
* Multi-Modal Contrastive Pre-training for Recommendation [ICMR Jun 2022] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3512527.3531378)]




## Textual Based Rec

### Title Abstract Tag

* Tag2Word: Using Tags to Generate Words for Content Based Tag Recommendation [CIKM Oct 2016] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/2983323.2983682)]
* Interactive resource recommendation algorithm based on tag information [WWW Feb 2018] [[__pdf__](https://link.springer.com.remotexs.ntu.edu.sg/content/pdf/10.1007/s11280-018-0532-y.pdf)]
* Leveraging Title-Abstract Attentive Semantics for Paper Recommendation [AAAI Apr 2020] [[__pdf__](https://ojs.aaai.org/index.php/AAAI/article/view/5335)]
* Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation [CIKM Oct 2020] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3340531.3416021)] [***github***](https://github.com/lqfarmer/GraphTR)


### Reviews Description

* Convolutional Matrix Factorization for Document Context-Aware Recommendation [RecSys Sep 2016] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/2959100.2959165)] [***github***](https://github.com/cartopy/ConvMF)
* Joint Deep Modeling of Users and Items Using Reviews for Recommendation [WSDM Feb 2017] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3018661.3018665)] [***github***](https://github.com/winterant/DeepCoNN)
* Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction [RecSys Aug 2017] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3109859.3109890)] [***github***](https://github.com/seongjunyun/CNN-with-Dual-Local-and-Global-Attention)
* Coevolutionary Recommendation Model: Mutual Learning between Ratings and Reviews [WWW Apr 2018] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3178876.3186158)]
* Neural Attentional Rating Regression with Review-level Explanations [WWW Apr 2018] [[__pdf__](https://dl.acm.org/doi/pdf/10.1145/3178876.3186070)] [***github***](https://github.com/chenchongthu/NARRE)
* Multi-Pointer Co-Attention Networks for Recommendation [KDD Jul 2018] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3219819.3220086)] [***github***](https://github.com/vanzytay/KDD2018_MPCN)
* PARL: Let Strangers Speak Out What You Like [CIKM Oct 2018] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3269206.3271695)] [***github***](https://github.com/WHUIR/PARL)
* ANR: Aspect-based Neural Recommender [CIKM Oct 2018] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3269206.3271810)] [***github***](https://github.com/almightyGOSU/ANR)
* A Context-Aware User-Item Representation Learning for Item Recommendation [TOIS Jan 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3298988)] [***github***](https://github.com/WHUIR/CARL)
* Recommendation Based on Review Texts and Social Communities: A Hybrid Model [IEEE Access Feb 2019] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=8635542)] [***github***](https://github.com/pp1230/HybridRecommendation)
* Attentive Aspect Modeling for Review-Aware Recommendation [TOIS Mar 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3309546)]
* NRPA: Neural Recommendation with Personalized Atention [SIGIR Jul 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3331184.3331371)] [***github***](https://github.com/microsoft/recommenders)
* DAML: Dual Attention Mutual Learning between Ratings and Reviews for Item Recommendation [KDD Aug 2019] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3292500.3330906)]
* Reviews Meet Graphs: Enhancing User and Item Representations for Recommendation with Hierarchical Attentive Graph Neural Network [EMNLP | IJCNLP Nov 2019] [[__pdf__](https://aclanthology.org/D19-1494.pdf)]  [***github***](https://github.com/wuch15/Reviews-Meet-Graphs)
* Neural Unified Review Recommendation with Cross Attention [SIGIR Jul 2020] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3397271.3401249)] 
* Learning Hierarchical Review Graph Representations for Recommendation [TKDE Apr 2021] [[__pdf__](https://arxiv.org/pdf/2004.11588.pdf)] [***github***](https://github.com/lqfarmer/GraphTR)
* Improving Explainable Recommendations by Deep Review-Based Explanations [IEEE Access Apr 2021] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=9417205&tag=1)]
* Counterfactual Review-based Recommendation [CIKM Nov 2021] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3459637.3482244)] [***github***](https://github.com/CFCF-anonymous/Counterfactual-Review-based-Recommendation)
* Review-Aware Neural Recommendation with Cross-Modality Mutual Attention [CIKM Nov 2021] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3459637.3482172)] 
* Aligning Dual Disentangled User Representations from Ratings and Textual Content [KDD Aug 2022] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3534678.3539474)] [***github***](https://github.com/PreferredAI/ADDVAE)




## Visual Based Rec

* VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback [AAAI Feb 2016] [[__pdf__](https://ojs.aaai.org/index.php/AAAI/article/view/9973)] [***github***](https://github.com/arogers1/VBPR)
* Do" Also-Viewed" Products Help User Rating Prediction? [WWW Apr 2017] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3038912.3052581)]
* DeepStyle: Learning User Preferences for Visual Recommendation [SIGIR Aug 2017] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3077136.3080658)] 
* Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention [SIGIR Aug 2017] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3077136.3080797)] [***github***](https://github.com/ChenJingyuan91/ACF) 
* Visually-aware fashion recommendation and design with generative image models [ICDM Nov 2017] [[__pdf__](https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=8215493)] [***github***](https://github.com/elenagiarratano/visually-aware-recommender-system) 
* Visually-Aware Personalized Recommendation using Interpretable Image Representations [arxiv 2018] [[__pdf__](https://arxiv.org/pdf/1806.09820.pdf)]
* Exploring the Power of Visual Features for the Recommendation of Movies [UMAP Jun 2019] [[__pdf__](https://dl.acm.org/doi/abs/10.1145/3320435.3320470)]
* Image and Video Understanding for Recommendation and Spam Detection Systems [KDD Aug 2020] [[__pdf__](https://dl.acm.org.remotexs.ntu.edu.sg/doi/pdf/10.1145/3394486.3406485)] 
* CausalRec: Causal Inference for Visual Debiasing in Visually-Aware Recommendation [MM Oct 2021] [[__pdf__](https://arxiv.org/pdf/2107.02390.pdf)]

## EVALUATING THE SOTA MODELS

we validate the effectiveness and efficiency of state-of-the-art multimodal recommendation models by conducting extensive experiments on four public datasets. Furthermore, we investigate the principal determinants of model performance, including the impact of different modality information and data split methods.

### Statistics of the evaluated datasets.
| Datasets | # Users | # Items | # Interactions |Sparsity|
|----------|--------|---------|---------|---------|
| Baby     | 19,445     | 7,050     |160,792|99.8827%|
| Sports   | 35,598      | 18,357   |296,337|99.9547%|
| FoodRec     | 61,668      | 21,874    |1,654,456|99.8774%|
| Elec     | 192,403      | 63,001     |1,689,188|99.9861%|


### Experimental Results
Comparison of performance for different models in terms of Recall and NDCG.

| Dataset                 | Model    | Recall@10          | Recall@20          | Recall@50          | NDCG@10            | NDCG@20            | NDCG@50            |
|-------------------------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| **Baby**   | BPR      | 0.0357             | 0.0575             | 0.1054             | 0.0192             | 0.0249             | 0.0345             |
|                         | LightGCN | 0.0479             | 0.0754             | 0.1333             | 0.0257             | 0.0328             | 0.0445             |
|                         | VBPR     | 0.0423             | 0.0663             | 0.1212             | 0.0223             | 0.0284             | 0.0396             |
|                         | MMGCN    | 0.0378             | 0.0615             | 0.1100             | 0.0200             | 0.0261             | 0.0359             |
|                         | DualGNN  | 0.0448             | 0.0716             | 0.1288             | 0.0240             | 0.0309             | 0.0424             |
|                         | GRCN     | 0.0539             | 0.0833             | 0.1464             | 0.0288             | 0.0363             | 0.0490             |
|                         | LATTICE  | 0.0547             | 0.0850             | 0.1477             | 0.0292             | 0.0370             | 0.0497             |
|                         | BM3      | 0.0564             | 0.0883             | 0.1477             | 0.0301             | 0.0383             | 0.0502             |
|                         | SLMRec   | 0.0529             | 0.0775             | 0.1252             | 0.0290             | 0.0353             | 0.0450             |
|                         | ADDVAE   | _0.0598_ | _0.091_  | _0.1508_ | _0.0323_ | _0.0404_ | _0.0525_ |
|                         | FREEDOM  | **0.0627**    | **0.0992**    | **0.1655**    | **0.0330**    | **0.0424**    | **0.0558**    |
| **Sports**  | BPR      | 0.0432             | 0.0653             | 0.1083             | 0.0241             | 0.0298             | 0.0385             |
|                         | LightGCN | 0.0569             | 0.0864             | 0.1414             | 0.0311             | 0.0387             | 0.0498             |
|                         | VBPR     | 0.0558             | 0.0856             | 0.1391             | 0.0307             | 0.0384             | 0.0492             |
|                         | MMGCN    | 0.0370             | 0.0605             | 0.1078             | 0.0193             | 0.0254             | 0.0350             |
|                         | DualGNN  | 0.0568             | 0.0859             | 0.1392             | 0.0310             | 0.0385             | 0.0493             |
|                         | GRCN     | 0.0598             | 0.0915             | 0.1509             | 0.0332             | 0.0414             | 0.0535             |
|                         | LATTICE  | 0.0620             | 0.0953             | 0.1561             | 0.0335             | 0.0421             | 0.0544             |
|                         | BM3      | 0.0656             | 0.0980             | 0.1581             | 0.0355             | 0.0438             | 0.0561             |
|                         | SLMRec   | 0.0663             | 0.0990             | 0.1543             | 0.0365             | 0.0450             | 0.0562             |
|                         | ADDVAE   | _0.0709_ | _0.1035_ | _0.1663_ | _0.0389_    | _0.0473_ | _0.0600_ |
|                         | FREEDOM  | **0.0717**    | **0.1089**    | **0.1768**    | **0.0385** | **0.0481**    | **0.0618**    |
| **FoodRec** | BPR      | 0.0303             | 0.0511             | 0.0948             | 0.0188             | 0.0250             | 0.0356             |
|                         | LightGCN | 0.0331             | 0.0546             | 0.1003             | 0.0210             | 0.0274             | 0.0386             |
|                         | VBPR     | 0.0306             | 0.0516             | 0.0972             | 0.0191             | 0.0254             | 0.0365             |
|                         | MMGCN    | 0.0307             | 0.0510             | 0.0943             | 0.0192             | 0.0253             | 0.0359             |
|                         | DualGNN  | _0.0338_ | 0.0559             | _0.1027_ | _0.0214_ | _0.0280_ | _0.0394_ |
|                         | GRCN     | **0.0356**   | **0.0578**    | **0.1063**    | **0.0226**    | **0.0295**    | **0.0411**    |
|                         | LATTICE  | 0.0336             | _0.0560_| 0.1012             | 0.0211             | 0.0277             | 0.0388             |
|                         | BM3      | 0.0334             | 0.0553             | 0.0994             | 0.0208             | 0.0274             | 0.0381             |
|                         | SLMRec   | 0.0323             | 0.0515             | 0.0907             | 0.0208             | 0.0266             | 0.0362             |
|                         | ADDVAE   | 0.0309             | 0.0508             | 0.093              | 0.0186             | 0.0247             | 0.035              |
|                         | FREEDOM  | 0.0333             | 0.0556             | 0.1009             | 0.0212             | 0.0279             | 0.0389             |
| **Elec**    | BPR      | 0.0235             | 0.0367             | 0.0621             | 0.0127             | 0.0161             | 0.0212             |
|                         | LightGCN | 0.0363             | 0.0540             | 0.0879             | 0.0204             | 0.0250             | 0.0318             |
|                         | VBPR     | 0.0293             | 0.0458             | 0.0778             | 0.0159             | 0.0202             | 0.0267             |
|                         | MMGCN    | 0.0213             | 0.0343             | 0.0610             | 0.0112             | 0.0146             | 0.0200             |
|                         | DualGNN  | 0.0365             | 0.0542             | 0.0875             | 0.0206             | 0.0252             | 0.0319             |
|                         | GRCN     | 0.0389             | 0.0590             | 0.0970             | 0.0216             | 0.0268             | 0.0345             |
|                         | LATTICE  | -                  | -                  | -                  | -                  | -                  | -                  |
|                         | BM3      | 0.0437             | 0.0648             | 0.1021             | 0.0247             | 0.0302             | 0.0378             |
|                         | SLMRec   | _0.0443_ | _0.0651_ | _0.1038_ | _0.0249_ | _0.0303_ | _0.0382_ |
|                         | ADDVAE   | **0.0451**    | **0.0665**    | **0.1066**    | **0.0253**    | **0.0308**    | **0.0390**    |
|                         | FREEDOM  | 0.0396             | 0.0601             | 0.0998             | 0.0220             | 0.0273             | 0.0353             |

### Ablation Study

#### Recommendation performance comparison using different data split methods.:

We evaluate the performance of various recommendation models using different data splitting methods. The offline evaluation is based on the historical item ratings or the implicit item feedback. As this method relies on the user-item interactions and the models are all learning based on the supervised signals, we need to split the interactions into train, validation and test sets. There are three main split strategies that we applied to compare the performance:

• Random split: As the name suggested, this split strategy randomly selects the train and test boundary for each user, which selects to split the interactions according to the ratio. The disadvantage of the random splitting strategy is that they are not capable to reproduce unless the authors publish how the data split and this is not a realistic scenario without considering the time.

• User time split: The temporal split strategy splits the historical interactions based on the interaction timestamp by the ratio (e.g., train:validation:test=8:1:1). It split the last percentage of interactions the user made as the test set. Although it considers the timestamp, it is still not a realistic scenario because it is still splitting the train/test sets among all the interactions one user made but did not consider the global time.

• Global time split: The global time splitting strategy fixed the time point shared by all users according to the splitting ratio. The interactions after the last time point are split as the test set. Additionally, the users of the interactions after the global temporal boundary must be in the training set, which follows the most realistic and strict settings. The limitation of this strategy is that the number of users will be reduced due to the reason that the users not existing in the training set will be deleted

Our experiments on the Sports dataset, using these three splitting strategies, provide insights into their impact on recommendation performance. The table below presents the performance comparison results in terms of Recall@k and NDCG@k where k=10,20, and the second table shows the performance ranking of models based on Recall@20 and NDCG@20.

| Dataset | Model    |          | Recall@10 |             |          | Recall@20 |             |
|---------|----------|----------|-----------|-------------|----------|-----------|-------------|
|         |          | Random   | User Time | Global Time | Random   | User Time | Global Time |
|         | MMGCN    | 0.0384   | 0.0266    | 0.0140      | 0.0611   | 0.0446    | 0.0245      |
|         | BPR      | 0.0444   | 0.0322    | 0.0152      | 0.0663   | 0.0509    | 0.0258      |
|         | VBPR     | 0.0563   | 0.0385    | 0.0176      | 0.0851   | 0.0620    | 0.0298      |
|         | DualGNN  | 0.0576   | 0.0403    | 0.0181      | 0.0859   | 0.0611    | 0.0297      |
| sports  | GRCN     | 0.0604   | 0.0418    | 0.0167      | 0.0915   | 0.0666    | 0.0286      |
|         | LightGCN | 0.0568   | 0.0405    | 0.0205      | 0.0863   | 0.0663    | 0.0336      |
|         | LATTICE  | 0.0641   | 0.0450    | 0.0207      | 0.0964   | 0.0699    | 0.0337      |
|         | BM3      | 0.0646   | 0.0447    | 0.0213      | 0.0955   | 0.0724    | 0.0336      |
|         | SLMRec   | 0.0651   | 0.0470    | 0.0220      | 0.0985   | 0.0733    | 0.0350      |
|         | FREEDOM  | 0.0708   | 0.0490    | 0.0226      | 0.1080   | 0.0782    | 0.0372      |
| Dataset | Model    |          | NDCG@10   |             |          | NDCG@20   |             |
|         |          | Random   | User Time | Global Time | Random   | User Time | Global Time |
|         | MMGCN    | 0.0202   | 0.0134    | 0.0091      | 0.0261   | 0.0180    | 0.0125      |
|         | BPR      | 0.0245   | 0.0169    | 0.0102      | 0.0302   | 0.0218    | 0.0135      |
|         | VBPR     | 0.0304   | 0.0204    | 0.0115      | 0.0378   | 0.0265    | 0.0153      |
|         | DualGNN  | 0.0321   | 0.0214    | 0.0118      | 0.0394   | 0.0268    | 0.0155      |
| sports  | GRCN     | 0.0332   | 0.0219    | 0.0101      | 0.0412   | 0.0282    | 0.0138      |
|         | LightGCN | 0.0315   | 0.0220    | 0.0139      | 0.0391   | 0.0286    | 0.0180      |
|         | LATTICE  | 0.0351   | 0.0238    | 0.0138      | 0.0434   | 0.0302    | 0.0177      |
|         | BM3      | 0.0356   | 0.0237    | 0.0144      | 0.0436   | 0.0308    | 0.0182      |
|         | SLMRec   | 0.0364   | 0.0253    | 0.0148      | 0.0450   | 0.0321    | 0.0189      |
|         | FREEDOM  | 0.0388   | 0.0255    | 0.0151      | 0.0485   | 0.0330    | 0.0197      |

As demonstrated above, different data splitting strategies lead to varied performance outcomes for the same dataset and evaluation metrics. This variability presents a challenge in comparing the effectiveness of different models when they are based on different data split strategies.

|  Model   |        | Sports, NDCG@20   |             |
|----------|--------|-------------------|-------------|
|          | Random | User Time         | Global Time |
| MMGCN    | 10     | 10                | 10          |
| BPR      | 9      | 9                 | 8↑1         |
| VBPR     | 8      | 8                 | 7↑1         |
| LightGCN | 7      | 5↑2               | 4↑3         |
| DualGNN  | 6      | 7↓1               | 6           |
| DRCN     | 5      | 6↓1               | 9↓4         |
| LATTICE  | 4      | 4                 | 5↓1         |
| BM3      | 3      | 3                 | 3           |
| SLMRec   | 2      | 2                 | 2           |
| FREEDOM  | 1      | 1                 | 1           |
| **Model**    |        | **Sports, Recall@20** |             |
|          | Random | User Time         | Global Time |
| MMGCN    | 10     | 10                | 10          |
| BPR      | 9      | 9                 | 9           |
| VBPR     | 8      | 7↑1               | 6↑2         |
| DualGNN  | 7      | 8↓1               | 7           |
| LightGCN | 6      | 6                 | 5↑1         |
| GRCN     | 5      | 5                 | 8↓3         |
| BM3      | 4      | 3↑1               | 4           |
| LATTICE  | 3      | 4↓1               | 3           |
| SLMRec   | 2      | 2                 | 2           |
| FREEDOM  | 1      | 1                 | 1           |

The above table reports the ranks of SOTA models under each splitting strategy. The rows are sorted by the performance of models under random splitting strategy, with the up and down arrows indicating the relative rank position swaps compared with random splitting. As we can see, the ranking swaps are observed between the models under different splitting strategies

#### Recommendation performance comparison using Different Modalities
We are interested in how the modality information benefits the recommendation, and which modality contributes more. We aim to understand the specific benefits of different modalities in recommendation systems and provide guidelines for researchers on selecting appropriate modalities. We evaluate it by feeding the single modality information, and compare the performance between using both modalities and the single modality. 

The following figure is based on Recall@20 to show the summary and tendency of other modalities, visually summarize the impact of different modalities on various models. The orange point represents the performance of multi-modality, the green one represents the performance of textual modality and the blue point is for visual modality. The specific numerical values will be shown in our github.


<img src="https://github.com/hongyurain/Recommendation-with-modality-information/blob/main/IMG/modality-baby.jpg" alt="image-1" height="50%" width="50%" /><img src="https://github.com/hongyurain/Recommendation-with-modality-information/blob/main/IMG/modality-sports.jpg" alt="image-2" height="50%" width="50%" />



#### Please consider to cite our paper if it helps you, thanks:
```
@article{zhou2023comprehensive,
  title={A Comprehensive Survey on Multimodal Recommender Systems: Taxonomy, Evaluation, and Future Directions},
  author={Zhou, Hongyu and Zhou, Xin and Zeng, Zhiwei and Zhang, Lingzi and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2302.04473},
  year={2023}
}
```



