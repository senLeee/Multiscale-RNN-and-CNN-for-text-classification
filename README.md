# Multiscale-RNN-and-CNN-for-text-classification
Multiscale-RNN-and-CNN-for-text-classification
模型的实现基于Python 2.7和tensorflow 1.4版本.

Google Word2Vec的下载地址：https://github.com/mmihaltz/word2vec-GoogleNews-vectors

由于本实验中三个数据集处理后的结果较大，无法提交。在此，我们提供三个数据集的原数据和对应预处理代码.

需要先下载 Google Word2Vec 并将其放在 Source_code\ 目录下.

Source_code\SST：该目录下存放着 process_SST1.py 与 process_SST2.py 两个脚本文件，对应运行即可生成处理完成的数据集.

Source_code\Subj：该目录下存放着 process_Subj.py ，运行即可生成处理完成的数据集.

处理完数据后，回到 Source_code 目录下，然后运行对应的.py文件即可.

运行方式为：python xxx.py

举例：倘若要验证Subj数据的实验结果，就运行 python Subj.py 即可.
