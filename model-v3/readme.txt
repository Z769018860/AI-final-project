�����ļ�˵����
frontend.py:	ģ��ǰ�ˣ�RNN����������import
backend.py:	ģ�ͺ�ˣ����򣩣�������import
train.py:	ѵ������
predict.py:	Ԥ�����
evaluate.py:	��������

�������̣�
1.��train.py, predict.py, evaluate.py��train_file_name��test_file_name����Ϊ��ȷ��ѵ�����Ͳ��Լ�
��Ŀǰѵ�����Ͳ��Լ��ֱ�Ϊԭѵ������ǰ2500��ͺ�500�
2.����train.py��ʼѵ����ѵ�������./default.ckpt����
3.����predict.py���Բ��Լ�����Ԥ�⣬��������result.json
4.����evaluate.py���Բ��Լ���ground truth��Ԥ������������

ע�����
Ŀǰ��ѵ������Ϊepochs=20,batchsize=500,iterations=5���ڷ�������ʹ��GPUѵ����Լ��10���ӣ��ڸ��˱ʼǱ������Ͽ����ڴ治��

requirements�������
û��������ܴ��������������

����˵����
���Project��������̣��ļ�������־���Կ�github��https://github.com/Z769018860/AI-final-project
������û���Ͻ��ĸ��������ļ��������⣩