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
现在直接用这个代码[OFA ](https://github.com/LechengKong/OneForAll/blob/main/run_cdm.py)


主要包括这几步：


    """
    1. Initiate task constructor.
    """
    
    """
    2. Load model 
    """
    
    """
    3. Construct datasets and lightning datamodule.
    """
       
    """
    4. Initiate evaluation kit. 
    """

    """
    5. Initiate optimizer, scheduler and lightning model module.
    """
    
    """
    6. Start training and logging.
    """
## 抽取特征
![把这里的特征写到文件里面去](/images/image.jpg)

## 特征读取

从文件里面读出来
！[替换这里的特征值](/images/image11.jpg)

## 其他

> 要配制huggingface国内镜像
> 可以到官网上下载程序运行不了的对应的LLM文件放到对应的文件夹
> 更换模型对于路径*（云端到本地）
