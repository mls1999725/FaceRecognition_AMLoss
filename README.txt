1. train.py 采用sphereface20网络和提出的loss，用casia-webface数据集训练，并在LFW上测试。
2. net.py 网络结构定义
3. layer.py 损失函数定义
4. dataset.py 数据载入
5. YTF_TEST 采用sphereface20网络和提出的loss，用casia-webface数据集训练，并在YTF上测试。YTF_TEST_Arcface采用Arcface中的loss，其他以此类推。其中IO.py是训练和测试的main文件。
6. Megaface 其中megaface_feature.py和probe_feature.py分别是用现有模型提取megaface和facescrub数据集中特征的文件。
