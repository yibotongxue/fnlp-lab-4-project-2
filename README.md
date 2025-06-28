# 代码结构和运行说明

本项目实现了北京大学自然语言处理基础第四次作业第二个项目——多被告法律判决预测，我们通过对预训练语言模型进行微调实现了对罪名和刑期的预测，以下是对代码结构和运行说明的简要说明。

## 项目结构

代码主要在 `src` 目录下， `scripts` 目录下有一些常用的脚本文件， `configs` 目录下是常用的配置文件。

```bash
.
├── configs # 配置文件
│   ├── finetune # 训练相关的配置文件
│   │   ├── charge.yaml # 单标签罪名预测模型训练配置
│   │   ├── imprisonment.yaml # 刑期预测模型训练配置
│   │   ├── multi_charge.yaml # 多标签罪名预测模型训练配置
│   │   └── multi_task.yaml # 多任务同时训练配置
│   └── predictor # 预测相关的配置文件
│       └── imprisonment_mapper.yaml # 刑期与标签映射配置
├── images # 一些图片
├── Makefile # 一些常用的开发命令
├── Report.md # 实验报告
├── requirements.txt # 项目依赖
├── scripts # 一些脚本文件
│   ├── dataset # 数据集相关脚本
│   │   └── course # 教学网提供的数据集
│   │       ├── split.py # 分割数据集为训练集和评估集
│   │       └── split.sh # 分割数据集为训练集和评估集
│   ├── embed # 生成嵌入向量的脚本
│   │   ├── articles # 生成法律条文的脚本
│   │   │   ├── articles.sh
│   │   │   ├── batch_file_generator.py
│   │   │   └── match_text_embed.py
│   │   └── batch_embed_generator.py
│   ├── multi_task # 多任务同时学习的脚本
│   │   ├── eval.sh # 评估脚本，在测试集上运行
│   │   └── train.sh # 训练模型
│   ├── one-by-one # 分别预测罪名和刑期的脚本
│   │   ├── finetune_allzero_eval.sh # 分类器预测罪名，刑期全部预测为0
│   │   ├── finetune_finetune_eval.sh # 分类器预测罪名，分类器预测刑期
│   │   ├── finetune_mostcommon_eval.sh # 分类器预测罪名，刑期预测为该罪名最常见的刑期
│   │   ├── finetune_zeroshot_eval.sh # 分类器预测罪名，零样本学习预测刑期
│   │   ├── multilabel_allzero_eval.sh # 多标签分类器预测罪名，刑期全部预测为0
│   │   ├── refine.sh # 分类器预测罪名，取概率最高的若干个，由大语言模型从中选择
│   │   └── train # 训练相关的脚本
│   │       ├── charge_train.sh # 罪名预测模型训练
│   │       ├── imprisonment_train.sh # 刑期预测模型训练
│   │       └── multi_label.sh # 多标签罪名预测模型训练
│   ├── rag # 检索增强生成相关代码
│   │   └── eval.sh
│   ├── score # 评分代码，主要用于本地评分
│   │   ├── score.py
│   │   └── score.sh
│   ├── submit # 生成提交到Kaggle平台的csv文件
│   │   ├── submit.py
│   │   └── submit.sh
│   └── zero_shot # 零样本学习相关脚本
│       ├── batch_decode.py
│       ├── batch_decode.sh
│       ├── batch_generate.sh
│       ├── batch_generator.py
│       └── eval.sh # 零样本学习评估脚本
├── src # 主要代码
│   ├── embed # 嵌入相关的代码
│   │   ├── base.py # 嵌入基类
│   │   ├── factory.py # 生成嵌入实例的工厂方法
│   │   ├── __init__.py
│   │   └── openai.py # 调用OpenAI类似接口（包含Qwen）的嵌入实现
│   ├── finetune # 微调相关的代码
│   │   ├── data # 数据处理相关的代码
│   │   │   ├── case_dataset.py # 数据集定义
│   │   │   ├── data_formatter.py # 数据处理代码
│   │   │   ├── __init__.py
│   │   │   └── template_registry.py # 注册数据集处理模板
│   │   ├── model # 模型定义
│   │   │   ├── __init__.py
│   │   │   └── legal_prediction_model.py # 多任务同时学习模型
│   │   ├── trainer # 训练器相关代码
│   │   │   ├── base.py # 训练器基类
│   │   │   ├── multi_task.py # 多任务同时学习训练器
│   │   │   └── normal.py # 分类任务学习训练器，包含多标签分类任务
│   │   └── utils # 一些支持类
│   │       ├── constants.py # 一些常量定义
│   │       ├── __init__.py
│   │       └── pretrained_utils.py # 加载预训练模型相关的代码
│   ├── __init__.py
│   ├── llm # 封装大语言模型调用的接口
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── __init__.py
│   │   ├── local_vllm.py
│   │   └── openai.py
│   ├── metrics # 度量方法，主要是课程提供的两个评估脚本
│   │   ├── __init__.py
│   │   ├── metric_charge.py
│   │   └── metric_imprisonment.py
│   ├── predictor # 预测器，预测罪名和刑期
│   │   ├── base.py # 预测器接口
│   │   ├── charge_predictor # 罪名预测器
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   ├── __init__.py
│   │   │   ├── lawformer.py # 调用分类器预测罪名
│   │   │   ├── multi_label.py # 多标签预测罪名
│   │   │   ├── multiple_predictor # 同时预测多个可能罪名，供大语言模型选择
│   │   │   │   ├── base.py
│   │   │   │   ├── factory.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── lawformer.py
│   │   │   ├── refine.py # 分类器预测多个罪名，由大语言模型选择一个最佳罪名
│   │   │   ├── refiner # 大语言模型从多个可能罪名选择一个最可能罪名
│   │   │   │   ├── base.py
│   │   │   │   ├── factory.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── zero_shot.py
│   │   │   ├── voter.py
│   │   │   └── zero_shot.py # 零样本学习预测罪名
│   │   ├── imprisonment_predictor # 刑期预测器
│   │   │   ├── all_zero.py # 全部预测为0
│   │   │   ├── base.py # 刑期预测基类
│   │   │   ├── factory.py # 生成刑期预测器实例的工厂方法
│   │   │   ├── __init__.py
│   │   │   ├── lawformer.py # 使用分类器预测刑期
│   │   │   ├── most_common.py # 预测为该罪名最常见的刑期
│   │   │   └── zero_shot.py # 零样本学习预测刑期
│   │   ├── __init__.py
│   │   ├── lawformer.py # 使用多任务学习训练的模型同时预测罪名和刑期
│   │   ├── llm.py # 调用大语言模型预测罪名和刑期的基类，主要有零样本学习和检索增强生成两个实现
│   │   ├── one_by_one.py # 分别预测罪名和刑期
│   │   ├── rag.py # 检索增强生成同时预测罪名和刑期
│   │   └── zero_shot.py # 零样本学习同时预测罪名和刑期
│   ├── retriever # 检索器定义
│   │   ├── base.py # 检索器抽象基类
│   │   ├── factory.py # 生成检索器实例的工厂方法
│   │   ├── __init__.py
│   │   ├── numpy_retriver.py # 使用Numpy数组进行检索的检索器
│   │   └── vector.py # 向量检索器基类
│   └── utils # 一些支持类和方法
│       ├── config.py # 加载配置文件的支持
│       ├── data_utils.py # 数据加载处理的支持
│       ├── dict_utils.py # 从模型响应中提取JSON的支持
│       ├── imprisonment_mapper.py # 刑期和标签映射支持
│       ├── __init__.py
│       ├── json_util.py # JSON和JSONL文件读取和写入的支持
│       └── tools.py # 一些常用工具方法
└── test # 一些测试代码
```

## 运行

### 安装依赖

在项目根目录运行命令，安装[requirements.txt](./requirements.txt)中的依赖：

```bash
pip install -r requirements.txt
```

### 运行脚本

使用我们的脚本可以方便的进行模型的训练和推理，但注意可能需要修改脚本中的一些参数。

分割数据集：

```bash
./scripts/dataset/course/split.sh
```

训练单标签罪名预测模型：

```bash
./scripts/one-by-one/train/charge_train.sh
```

训练多标签罪名预测模型：

```bash
./scripts/one-by-one/train/multi_label.sh
```

训练刑期预测模型：

```bash
./scripts/one-by-one/train/imprisonment_train.sh
```

批改作业时如果不希望从头训练模型，可以在[这里](https://disk.pku.edu.cn/link/AAA86D310EC17849B4A8C6EF1F53873CF0)下载我们上传的检查点，提取密码为9QO7。

> Kaggle平台上是我们最高的测试分数，因为最高分数的检查点未能找到，上传的检查点是重新训练的，使用下载的检查点或者重新训练不一定达到Kaggle平台上的分数，至少我们本地测试的时候发现刑期预测的分数少了0.005，尽管分数有差异，但差异很小。

运行推理脚本：

```bash
./scripts/one-by-one/finetune_finetune_eval.sh
```

生成提交文件：

```bash
./scripts/submit/submit.sh
```

## 其他尝试的运行

以下是报告中“其他的尝试“部分的运行，如果批改作业的时候也希望运行这部分，可以参考。

### 多任务学习模型

模型训练：

```bash
./scripts/multi_task/train.sh
```

模型推理：

```bash
./scripts/multi_task/eval.sh
```

### 零样本学习

运行脚本

```bash
./scripts/zero_shot/eval.sh
```

### 检索增强生成

生成嵌入向量

```bash
./scripts/embed/articles/articles.h
```

运行预测：

```bash
./scripts/rag/eval.sh
```

### 大模型校验选择

运行脚本

```bash
./scripts/one-by-one/refine.sh
```
