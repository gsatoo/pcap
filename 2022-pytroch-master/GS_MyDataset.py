######################################
# 【必読】PyTorchのDatasetの使い方｜カスタム（自作）Datasetも作成できる！
# https://dreamer-uma.com/pytorch-dataset/

import torch
import torchvision
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, features, labels, transform=None):
        # #
        # print("!!!!!!!!!!!!!!!!")
        # print(df)
        # print(df.columns)
        # print(features)
        # print(labels)
        # print("!!!!!!!!!!!!!!!!")
        # #
        self.features_values = df[features].values
        self.labels = df[labels].values
        self.transform = transform

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.features_values)

    # 要素を参照すると呼ばれる関数    
    def __getitem__(self, idx):
        features_x = torch.FloatTensor(self.features_values[idx])
        labels = torch.LongTensor(self.labels[idx])
        # labels = torch.FloatTensor(self.labels[idx])

        # 前処理を施す
        if self.transform:
            features_x = self.transform(features_x)
        
        return features_x, labels


if __name__ == '__main__':
    
    import seaborn as sns
    iris = sns.load_dataset('iris')

    iris.loc[:, 'species'] = iris.loc[:, 'species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
    iris.head()

    # 特徴量の名前
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # ラベルの名前
    label = ['species']

    Dataset = MyDataset(iris, features, label)    

    print(len(Dataset))

    features_sample, label_sample = Dataset[0]
    print(features_sample, label_sample)
