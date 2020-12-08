# tianchi-multi-task-nlp
NLP中文预训练模型泛化能力挑战赛

---

机器信息：NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2

pytorch 版本 1.6.0

---

已修复inference全为一个种类的bug，inference的时候设了一个max_len的padding。。。。

---

pytorch版本 baseline f1-score 0.616

需要安装 huggingface 和 pytorch

安装 huggingface：

pip install transformers

---

首先去 https://huggingface.co/bert-base-chinese/tree/main 下载config.json vocab.txt pytorch_model.bin

然后把这三个文件放进tianchi-multi-task-nlp/bert_pretrain_model文件夹下

然后把比赛 https://tianchi.aliyun.com/competition/entrance/531841/information 的数据下载

*_train1128.csv 文件改名为 total.csv 并放进 tianchi-multi-task-nlp/tianchi_datasets/数据集名字 文件夹下

*_a.csv 文件改名为 test.csv 并放进 tianchi-multi-task-nlp/tianchi_datasets/数据集名字 文件夹下

例如 tianchi-multi-task-nlp/tianchi_datasets/OCNLI/total.csv

tianchi-multi-task-nlp/tianchi_datasets/OCNLI/test.csv

---

分开训练集和验证集，默认验证集是各3000条数据，参数可以自己修改：

python ./generate_data.py

---

训练模型：

python ./train.py

会保存验证集上平均f1分数最高的模型到 ./saved_best.pt

如果要挂后台进程训练请用如下命令：

nohup python -u ./train.py >train.log 2>&1 &

训练日志会写进 ./train.log

用如下命令查看训练情况：

tail -f ./train.log

---

在输出预测结果前，先在 tianchi-multi-task-nlp/submission 下新建一个文件夹 5928

由于验证集的f1分数是0.5928，所以我就用这个文件夹名

用训练好的模型 ./saved_best.pt 生成结果：

python ./inference.py

结果会写进 tianchi-multi-task-nlp/submission/5928 文件夹下

然后cd 到 tianchi-multi-task-nlp/submission/5928 文件夹下，执行如下命令：

zip -r ./submit.zip ./*.json

生成submit.zip 提交结果即可，本baseline只有大概58的f1分数

---

这个模型是最简单的bert-base，可以考虑以下几点进行优化

1 修改 calculate_loss.py 改变loss的计算方式，从平衡子任务难度以及各子任务类别样本不均匀入手

2 修改 net.py 改变模型的结构

3 使用 cleanlab 等工具对训练数据进行清洗

4 使用 k-fold 交叉验证

5 对训练好的模型再在完整数据集（包括验证集和训练集）上用小的学习率训练一个epoch

6 调整bathSize和a_step，变更梯度累计的程度，当前是batchSize=16，a_step=16，累计256个数据的梯度再一次性更新权重

7 数据增强

---

出现报错信息：

ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found

解决方案：

conda install libgcc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/anaconda3/lib/

---

增加 dtp loss

验证集大小 5000 -> 3000

用 chinese-roberta-wwm-ext 作为预训练模型

增加 weighted_loss

尝试使用attention
