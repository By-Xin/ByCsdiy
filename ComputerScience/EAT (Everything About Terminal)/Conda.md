# Anaconda Promt

## OS: Windows 11

### Basic Instructions on Anaconda Prompt

#### Setting Environment 虚拟环境设置

- `conda create -n ENVNAME` 创建一个名为【ENVNAME】的虚拟环境
- `conda activate ENVNAME` 激活虚拟环境
- `conda deactivate` 退出虚拟环境

#### Conda 镜像源

- `conda info` 查看目前conda源
- `conda config --remove-key channels` 删除目前conda源
- `conda config --add channels ***` 这里将\*替换为实际存在的镜像网站，即可完成对该源的添加
- `conda config --set show_channel_urls yes` 显示镜像源地址
- `conda config --remove channels ***` 同理，可以删除相应源
- 常用镜像网站汇总：
  - [清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/)
  - [中科大源](https://mirrors.ustc.edu.cn/)
- 一键部署（网站可能发生更新）

  ```
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
    conda config --set show_channel_urls yes
    ```
