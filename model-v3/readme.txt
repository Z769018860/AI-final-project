代码文件说明：
frontend.py:	模型前端（RNN），仅用于import
backend.py:	模型后端（规则），仅用于import
train.py:	训练代码
predict.py:	预测代码
evaluate.py:	评估代码

运行流程：
1.将train.py, predict.py, evaluate.py中train_file_name和test_file_name设置为正确的训练集和测试集
（目前训练集和测试集分别为原训练集的前2500项和后500项）
2.运行train.py开始训练，训练结果以./default.ckpt保存
3.运行predict.py，以测试集进行预测，输出结果到result.json
4.运行evaluate.py，以测试集的ground truth和预测结果进行评估

注意事项：
目前的训练参数为epochs=20,batchsize=500,iterations=5，在服务器上使用GPU训练大约需10分钟，在个人笔记本电脑上可能内存不够

requirements（依赖项）
没有新增框架代码以外的依赖项

补充说明：
这个Project的总体进程，文件更新日志可以看github：https://github.com/Z769018860/AI-final-project
包括了没有上交的各种其他文件（张丽玮）