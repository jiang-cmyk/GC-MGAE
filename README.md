# ComMGAE
The code is based on the implementation of （OFA） [GraphMAE](https://github.com/THUDM/GraphMAE) and [CSGCL](https://github.com/HanChen-HUST/CSGCL)



# Dependencies
* Python 3.9
* torch 11.6
* PyTorch 1.13.0
* dgl 1.1.1
* cdlib 0.2.6
* networkx 2.5.1
* numpy 1.23

Other specific dependencies can be found in the environment.yml

# Quick Start
## Clone code 
~~~shell
cd ~/home
git clone https://github.com/jiang-cmyk/ComMGAE
~~~

## Reproduce the environment
~~~conda
conda env create -f environment.yml
~~~


# 操作流程
## 使用LLM句子嵌入
现在直接用这个代码[OFA (https://github.com/LechengKong/OneForAll/blob/main/run_cdm.py)]

