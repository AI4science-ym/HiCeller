# HiCeller

The latest advancements in single-cell technology have brought unprecedented opportunities to uncover the molecular complexity of complex biological systems, particularly those associated with human-specific diseases. However, these advancements have also introduced new challenges—how to efficiently annotate long-tail single-cell data related to disease conditions. To effectively address this challenge, we proposed Celler, a cutting-edge generative pre-training model specifically designed for single-cell data annotation.
<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/s3.png" width="1100" align="center">
</p>
In our deep exploration of this field, we have not only gathered but also constructed a large-scale private dataset, Celler-75, which boasts an unparalleled data volume of 40 million annotated cells, covering 80 human tissues and 75 specific diseases. The scale, depth, and breadth of this dataset far surpass any publicly available dataset currently in existence.
<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/4data.png" width="1100" align="center">
</p>
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

## Experiment
Our project introducing Gaussian Inflation Loss (GInf Loss) and a secondary training phase with hard sample mining, achieving higher accuracy in cell annotation compared to scBERT, scGPT, CellPLM, and other methods.

<p align="center">
<img src="https://github.com/YaoGina/HiCeller/blob/main/78900cf3df77ef27edf507dd0a29b50.png" width="1100" align="center">
</p>

We have currently released the test and training sets for **seven tissues**, including **lung, skin, kidney, brain, liver, cortex, and breast**, covering the four tissues and related experiments mentioned in the paper. In the future, we will update with more species and tissues. **Our dataset is strictly prohibited for any commercial use outside of academic purposes.** Violations will be pursued legally.
