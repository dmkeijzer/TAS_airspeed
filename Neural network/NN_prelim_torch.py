import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch import tensor

file = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\data_list.csv"

df = pd.read_csv(file)

#Create training data
train_data1 = df[:30]
train_data1 = train_data1[['engine', 'alpha', 'sum2', 'sum3', 'mean2', 'mean3', 'stdev2', 'stdev3']].values
train_data1 = tensor(train_data1)

#Create target data
target = tensor(df["v"].values)




class prelim_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(6,1)
        self.weights = self.L1.weight
        

    def forward(self, x):
        x = F.relu(self.L1(x))
        # x = F.relu(self.fc2(x))
        return x


params = list(prelim_model.parameters())
print(f"\nlearning parameters = {params}\n")
print(params[0].size())

output = prelim_model(train_data1)
print(output)
target = target.view(1, -1)  # make it the same shape as output
print(target)
criterion = nn.MSELoss()

#loss = criterion(output, target)
# print(loss)

# model = prelim_model(train_data1)
# print(model)
# print(model.weights)

