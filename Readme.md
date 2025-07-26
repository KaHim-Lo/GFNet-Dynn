# Lightweight Remote Sensing Scene Classification on Edge Devices via Knowledge Distillation and Early-exit

GFNet-Dynn 是结合了 GFNet 和 Dynn 架构优点的混合深度学习模型。此仓库包含了用于训练、评估和部署 GFNet-Dynn 模型的代码库。项目旨在通过利用动态退出机制（早退机制）提高深度神经网络的效率和准确性。


## 安装指南

### 1. **克隆仓库**

```bash
git clone https://github.com/yourusername/GFNet-Dynn.git
cd GFNet-Dynn
```

### 2. **安装依赖**

参考 [GFNet](https://github.com/raoyongming/GFNet) 和 [Dynn](https://github.com/networkslab/dynn) 的依赖安装指南。

### 3. **加载数据**

在与本项目同级的 data/ 文件夹下加载数据集，具体数据加载方式可以参考 main_dynn.py 中的 load_dataset 函数部分。
   支持的数据集包括但不限于：
   - AID
   - NaSC
   - NWPU
   - PatternNet
   - UCM

确保数据集的路径配置正确，以便代码能正确加载数据。(可在 data_loader_help.py 中查看数据集的配置信息)

### 4. **配置检查点**

通过配置 `--resume` 参数选择是否加载模型检查点。您可以在 `main_dynn.py` 的 `main` 函数中根据需要配置 `checkpoint_path`，以恢复训练或进行模型评估。


## 使用方法

### 模型训练

要训练 GFNet-Dynn 模型，可以使用提供的 Bash 脚本：

```bash
./main_dynn.bash
```

或者参考 `main_dynn.txt` 中提供的命令示例，了解如何配置不同参数进行训练。

### 模型评估

对于预训练模型的评估，您可以配置 `--eval` 参数和 `--resume` 参数，并指定模型检查点路径 `checkpoint_path`，来运行评估函数。

### 实验追踪

此项目使用 MLflow 进行实验追踪。启动 MLflow 服务器并追踪实验：

```bash
mlflow ui
```

然后，在浏览器中访问 `http://localhost:5000`，导航到 `mlruns/` 目录查看和比较不同的运行结果。



## 配置说明

GFNet-Dynn 的主要配置参数可以在 `main_dynn.py` 中找到。以下是一些关键参数：

- `--data_set`：指定使用的数据集名称（如 AID、NaSC、NWPU等）。
- `--img_size`：设置输入图像的尺寸。
- `--num_classes`：指定分类任务的类别数。
- `--ce_ic_tradeoff`：控制分类损失与中间分类器损失的权衡系数。
- `--max_warmup_epoch`：设置动态 warmup 阶段的最大训练轮数。
- `--resume`：加载预训练模型或继续训练。
- `--eval`：启用评估模式，仅进行验证集测试。

如需更详细的配置说明，请查看 `main_dynn.py` 的 `argparse` 配置部分。



## 模型训练流程说明

GFNet-Dynn 的训练流程主要分为两个阶段：

1. **动态 Warmup 阶段**：
   - 在此阶段，模型会逐步训练每个中间分类器（exit），并根据验证集准确率冻结表现最好的分类器。
   - 一旦某个 exit 分类器在验证集上连续 `patience` 轮没有提升，该分类器将被冻结，并保存其最优权重。
   - 如果所有 exit 分类器都收敛，则提前终止训练。

2. **完整模型训练阶段**：
   - 在 warmup 阶段结束后，进入完整模型训练阶段。
   - 所有 exit 分类器的权重将被固定，仅训练主干网络（backbone）以进一步提升整体性能。
   - 最终模型将包含多个 exit 分类器，可以在推理阶段根据需求动态选择提前退出的位置。



## 检查点（Checkpoint）文件说明

GFNet-Dynn 在训练过程中会保存模型检查点（checkpoint）文件，用于恢复训练状态或进行模型评估。

所有 checkpoint 文件默认保存在 `checkpoint/` 文件夹中，其下包含多个子文件夹，用于分类存储不同阶段和配置的模型文件：

- **`best/`**  
  保存了训练过程中在验证集上表现最佳的模型文件，是训练完成后最终推荐使用的模型权重（**手动添加**）。

- **`_warmup/`**  
  保存了动态 Warmup 阶段的模型检查点，用于训练和冻结中间分类器（exit）。

- **`checkpoint_{dataset}_{ce_ic_tradeoff}_confEE/`**  
  保存了在不同数据集（`{dataset}`）和损失权衡系数（`{ce_ic_tradeoff}`）配置下训练得到的模型文件，用于区分不同实验条件下的训练结果。

如需查看 checkpoint 文件的保存路径和命名规则，可以参考 `our_train_helper.py` 文件中的相关实现。


## 参考资料

此项目借鉴了以下仓库的内容：

- [GFNet](https://github.com/raoyongming/GFNet)
- [Dynn](https://github.com/networkslab/dynn)

请参阅这些仓库以获取有关底层架构和方法的更多详细信息。

