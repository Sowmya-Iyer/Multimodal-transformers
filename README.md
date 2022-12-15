# A-comprehensive-survey-on-multimodal-transformers.

### This project presents a comprehensive survey of Transformer techniques specifically geared towards multimodal data up until 2022. The contributions of this survey include giving: </br>
(1) a theoretical review of Multimodal learning, Transformers, Vision Transformers, and Multimodal Transformers, </br>
(2) a review of multimodal Transformers through the perspective of two important applications- specific multimodal tasks, </br>
(3) to summarize commonalities in challenges faced and designs of existing Transformer models.

The scope of this survey is limited to discussing the multi-modal specific designs of Transformer architecture to modalities such as: RGB images, RGB-D images, Videos, Audio/Speech, Scene graph, Pose data, Point cloud, Multi-modal knowledge graph, 3D object, 3D scene, Healthcare data. The aim is to contribute a taxonomy for Transformer for MML from application based and challenge based perspectives. This survey excludes multimodal publications where Transformer is employed just as a feature extractor. 

### The survey’s key features can be summarized as 

(1) Highlight transformers which are modality-agnostic, that is, they’re compatible with several modalities (and combinations). To support this notion, we give a geometrically topological interpretation of Transformers’ inherent multi-modal features. Self-attention is suggested to be modeled as a fully-connected graph with uni-modal and multimodal input sequences. Self-attention embeds tokens from any modality as graph nodes.

(2) In this research, we extract the mathematical core and formulations of Transformer-based MML methods, from the perspective network architectures


## Based on Self-attention mechanism:
![Self-attention](https://github.com/Sowmya-Iyer/A-comprehensive-survey-on-multimodal-transformers./blob/main/img/savar.png)

Let the inputs from any two modalities be represented as $X_{(1)}$ and $X_{(2)}$. After token embedding, the inputs become $T_{(1)}$ and $T_{(2)}$. The token sequence given by multimodal interactions become $X^{*}$. Let $M(.)$ stand for the module operations of Transformer blocks and or or layers.

### Early Concatenation
```math
 T \leftarrow concat(T_{(1)},T_{(2)}) 
```
```math 
    X^{*} = M(Q_{T},K_{T},V_{T})
```

- [Graph-CODEBERT](https://github.com/microsoft/CodeBERT)
- [AV-HuBERT](https://github.com/facebookresearch/av_hubert)
- [VideoBERT](https://github.com/ammesatyajit/VideoBERT)

### Early Summation
$T_{12} \leftarrow (c_{1}T_{(1)} \oplus c_{2}T_{(2)})$ </br>
        $\implies Q_{12} = (c_{1}T_{(1)} \oplus c_{2}T_{(2)}) W^Q$ </br>
        $K_{12} = (c_{1}T_{(1)} \oplus c_{2}T_{(2)}) W^K$ </br>
        $V_{12} = (c_{1}T_{(1)} \oplus c_{2}T_{(2)}) W^V$ </br>
        $X^{*} = M(Q_{12},K_{12},V_{12})$ </br>

- [GroupFormer](https://github.com/xueyee/GroupFormer)
- [DeepChange](https://github.com/PengBoXiangShang/deepchange)

### Heirarchial Attention
#### Multi-stream $\rightarrow$ Single-stream
$T \leftarrow concat(M_{(1)}(T_{(1)}), M_{(2)}(T_{(2)}))$ </br>
$X^{*} = M_{(3)}(T)$

- [AI Choreographer](https://google.github.io/aichoreographer/)

####  Single-stream $\rightarrow$ Multi-stream
```math
X^{*}_{(1)}, X^{*}_{(2)} \leftarrow M_{(1)}(concat(T_{(1)}, T_{(2)}
```
```math
X^{*}_{(1)}\leftarrow M_{(2)}(T_{(1)})
```
```math
X^{*}_{(2)}\leftarrow M_{(3)}(T_{(2)})
```
- [InterBERT](https://github.com/black4321/InterBERT)

### Cross-Attention
```math
X^{*}_{(1) }\leftarrow MHSA(Q_{(T_{2})},K_{(T_{(2)})},V_{(T_{(2)})})
```
```math
X^{*}_{(2)} \leftarrow MHSA(Q_{(T_{2})},K_{(T_{(1)})},V_{(T_{(1)})})
```
    
- [Vil-BERT](https://github.com/facebookresearch/vilbert-multi-task)
- [Pano-AVQA](https://github.com/HS-YN/PanoAVQA)

### Cross-Attention + Concatenation

```math
X^{*}_{(1)}\leftarrow MHSA(Q_{(T_{2})},K_{(T_{(2)})},V_{(T_{(2)})})
```
```math
X^{*}_{(2) }\leftarrow MHSA(Q_{(T_{2})},K_{(T_{(1)})},V_{(T_{(1)})})
```
```math
X^{*} \leftarrow M(concat(X^{*}_{(1)}, X^{*}_{(2)}))
```
    
- [CrossViT](https://github.com/IBM/CrossViT)
- [MMT: Unaligned Multimodal Language Sequences](https://github.com/yaohungt/Multimodal-Transformer)
- [Product1M](https://github.com/zhanxlin/Product1M)

## Transformers for specific multimodal task

###  Discriminative tasks

|Modals | Task | Ref.|
|---|---|---|
|Test desc. + point cloud     | Visual Grounding | [3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer)|
|acoustic + text  | Bilingual Speech Translaton | [espnet](https://github.com/espnet/espnet) [neurst](https://github.com/bytedance/neurst)|
|audio + visual observation | Audio-Visual Navigation  | [paper](https://arxiv.org/pdf/2210.01353.pdf)|
|Contextual Tag embeddings |  Cross-modal Alignment of Audio and Tags |[ae w2v attention](https://github.com/xavierfav/ae-w2v-attention) |
|appearance + audio + speech | Video-retrieval | [ConTra](https://github.com/contra)|
|text query + image |Video-retrieval |  [AVSeeker](https://github.com/nvtu/AVSeeker-UI)|
|| image-text retrieval| [VL-BEiT](https://github.com/microsoft/unilm/tree/master/vl-beit), [GilBERT](https://github.com/gilbert)|
|| Document AI | [LayoutLMv3](https://github.com/ImmanuelString/LayoutMV3Trainmodel)|
|audio + video | Audio-Visual Video Parsing ||
|| Audio-Visual speech enhancement (AVSE)| [InterModality Attention flow](https://lupantech.github.io/papers/cvpr19_dynamicvqa.pdf), [AVSE-2022](https://arxiv.org/abs/2210.17456)|
|| Audio-Visual speech recognition | [av-hubert](https://github.com/facebookresearch/av_hubert)|
|video + text | Referring Video Object Segmentation (RVOS) | [MTTR](https://github.com/mttr2021/MTTR), [awesome-rvos](https://github.com/JerryX1110/awesome-rvos)|

### Generative Tasks
#### Single modality to a different single modality
|From | To | Ref. |
|---|---|---|
|image|3D human texture| [Texformer](https://github.com/xuxy09/Texformer)|
|single-image | geo-localition | [TransLocator](https://github.com/shramanpramanick/transformer_based_geo-localization)| 
|RGB | Scene Graph| [Relationformer](https://github.com/suprosanna/relationformer),[Balance Adjustment](https://arxiv.org/abs/2108.13129), [Context-aware](https://www.cs.utoronto.ca/~mvolkovs/ICCV2021_Transformer_SGG.pdf),[MMSceneGraph](https://github.com/Kenneth-Wong/MMSceneGraph)|
|Video | Caption | [SwinBERT](https://github.com/microsoft/SwinBERT), [Sketch, Ground, and Refine](https://paperswithcode.com/conference/cvpr-2021-1), [VX2TEXT](https://paperswithcode.com/conference/cvpr-2021-1)|
|Image | caption |[$M^2$](https://github.com/aimagelab/meshed-memory-transformer), [AoANet](https://github.com/husthuaan/AoANet)|
|text | Speech | [Dict-TTS](https://github.com/DictTTS/DictTTS-Demo), [TransformerTTS](https://github.com/as-ideas/TransformerTTS), [Grad-TTS](https://github.com/WelkinYang/GradTTS), [SC-GlowTTS](https://github.com/Edresson/SC-GlowTTS)|
|Text | Image | [DALLE-URBAN](https://github.com/sachith500/DALLEURBAN), [CogView](https://github.com/THUDM/CogView)|
|RGB | 3D human pose| [GTRS](https://github.com/zczcwh/GTRS) [MeshTransformer](https://github.com/microsoft/MeshTransformer)|
|music  | dance | [Transflower](https://github.com/guillefix/transflower-lightning), [DanceNet3D](https://github.com/DeepVTuber/DanceNet3D)|

#### Multimodality to different modality
|From|To| Ref. |
|---|---|---|
|image + text | scene graph|[DeepChange](https://github.com/PengBoXiangShang/deepchange), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS)|
|image + Query|Answer| [visqa](https://github.com/dhkim16/VisQA-release) |
|Aud + Visual + Scene | Dialog | [FrozenBiLM](https://github.com/antoyang/FrozenBiLM), [X-Vlm](https://github.com/zengyan-97/x-vlm), [ClipBERT](https://github.com/jayleicn/ClipBERT), [HME-VideoQA](https://github.com/fanchenyou/HME-VideoQA)|
