conda create -n DL python=3.8  #创建虚拟环境
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  #pytorch不用于实验，只用于下载数据


#在args.py中设置实验超参数及其他信息
#学习率，隐藏层，可视化图片索引，正则化强度，激活函数，保存最优模型文件名
args.py

#训练
python train.py

#测试
python model.py
