# HiCeller

## Model

The latest advancements in single-cell technology have brought unprecedented opportunities to uncover the molecular complexity of complex biological systems, particularly those associated with human-specific diseases. However, these advancements have also introduced new challenges—how to efficiently annotate long-tail single-cell data related to disease conditions. To effectively address this challenge, we proposed Celler, a cutting-edge generative pre-training model specifically designed for single-cell data annotation.

<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/s3.png" width="1100" align="center">
</p>


## Dataset

### Celler-75

In our deep exploration of this field, we have not only gathered but also constructed a large-scale private dataset, Celler-75, which boasts an unparalleled data volume of 40 million annotated cells, covering 80 human tissues and 75 specific diseases. The scale, depth, and breadth of this dataset far surpass any publicly available dataset currently in existence.

<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/4data.png" width="1100" align="center">
</p>


We have currently released the test and training sets for [**seven tissues**](https://pan.quark.cn/s/dd520f55c876), including **lung, skin, kidney, brain, liver, cortex, and breast**, covering the four tissues and related experiments mentioned in the paper. In the future, we will update with more species and tissues. **Our dataset is strictly prohibited for any commercial use outside of academic purposes.** Violations will be pursued legally.
### MS & hPancreas

The  [**Multiple Sclerosis**](https://pan.quark.cn/s/471174cc07e3) (MS) dataset  and the  [**hPancreas**](https://pan.quark.cn/s/24a3c11553b0) dataset are publicly available resources specifically tailored for investigating particular human diseases.


## Reproduce

You can reproduce our project using the following methods:

```
cd HiCeller-main
```

Install the environment by running the command: 

```
conda env create -f environment.yml
```

After that, you need to modify the model path and dataset path in `test.py`. You can obtain our checkpoint model and data **here**. Finally, run `python test.py` to use **HiCell**.

```
python test.py --data_path /*******.h5ad --model_path /*******.pt
```

## Experiment

Our project introducing Gaussian Inflation Loss (GInf Loss) and a secondary training phase with hard sample mining, achieving higher accuracy in cell annotation compared to scBERT, scGPT, CellPLM, and other methods.
The Gaussian Loss GInf function can be found in the file located at `/hicell/loss.py`. This function is specifically designed to calculate the Gaussian-based loss, which is a critical component in optimizing the model's performance.
<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/78900cf3df77ef27edf507dd0a29b50.png" width="1100" align="center">
</p>

The model weights we used in our experiments correspond to different tissues, such as the  [**brain**](https://pan.quark.cn/s/33e4eccaed23) and [**kidney**](https://pan.quark.cn/s/0ba774da43c5). You can download the weights for the respective tissues by clicking on the corresponding link.

<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/confusion_matrix-27382b412910.png" width="550" align="center">
</p>

## Citing this work

If you use the code, please cite:
```
@article{MSdataset,
  title={Neuronal vulnerability and multilineage diversity in multiple sclerosis},
  author={Schirmer, Lucas and Velmeshev, Dmitry and Holmqvist, Staffan and Kaufmann, Max and Werneburg, Sebastian and Jung, Diane and Vistnes, Stephanie and Stockley, John H and Young, Adam and Steindel, Maike and others},
  journal={Nature},
  volume={573},
  number={7772},
  pages={75--82},
  year={2019},
  publisher={Nature Publishing Group UK London}
}

@article{hPancreas-dataset,
  title={Transformer for one stop interpretable cell type annotation},
  author={Chen, Jiawei and Xu, Hao and Tao, Wanyu and Chen, Zhaoxiong and Zhao, Yuxuan and Han, Jing-Dong J},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={223},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@article{scbert,
  title={scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data},
  author={Yang, Fan and Wang, Wenchuan and Wang, Fang and Fang, Yuan and Tang, Duyu and Huang, Junzhou and Lu, Hui and Yao, Jianhua},
  journal={Nature Machine Intelligence},
  volume={4},
  number={10},
  pages={852--866},
  year={2022}
}

@article{ICLR2024cellplm,
  title={CellPLM: pre-training of cell language model beyond single cells},
  author={Wen, Hongzhi and Tang, Wenzhuo and Dai, Xinnan and Ding, Jiayuan and Jin, Wei and Xie, Yuying and Tang, Jiliang},
  journal={bioRxiv},
  pages={2023--10},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{scgpt,
  title={scGPT: Towards building a foundation model for single-cell multi-omics using generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Wang, Bo},
  journal={bioRxiv},
  pages={2023--04},
  year={2023}
}

```
## Contact us

If you have any questions regarding our work, you can contact us at yimingliu@hnu.edu.cn, but please note this is only for research purposes.
