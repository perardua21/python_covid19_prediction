import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import datetime
import torch
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm
from matplotlib import rc
from torch import nn, optim
plt.show()
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
###########C:\Users\User\AppData\Local\Programs\Python\Python38\Scripts

url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson'
params ={'serviceKey' : 'I6HM0np/2qg18VbWIl15dksy5denKo/ZpgFgoiX5K8ACjdEwpcViPYgS1QhTBHKzN+MloI9it/OGvegZf3DDDw==', 'type' : 'json', 'pageNo' : '1', 'numOfRows' : '1000', 'startCreateDt' : '20200101', 'endCreateDt' : '20211130' }

result = requests.get(url, params=params)
def parse():
    try:
        STATE_DT = item.find("stateDt").get_text()
        DECIDE_CNT = item.find("decideCnt").get_text()
        CREATE_DT = item.find("createDt").get_text()
        return {
            "기준일":STATE_DT,
            "날짜":CREATE_DT,
            "확진자 수":DECIDE_CNT
        }
    except AttributeError as e:
        return {
            "기준일":None,
            "날짜":None,
            "확진자 수":None
        }

row=[]
soup=BeautifulSoup(result.text,'lxml-xml')
items=soup.find_all("item")

for item in items:
    row.append(parse())

df=pd.DataFrame(row)

df.to_csv("covid19.csv",mode='w')
df.sort_values(by='기준일', ascending=True)
data=pd.read_csv("covid19.csv", index_col=0)
df2=pd.DataFrame(data)
df2=df2.sort_values(by=['기준일'], axis=0)
df2=df2.set_index("날짜")
df2.drop(['기준일'], axis=1, inplace=True)
df2.index=pd.to_datetime(df2.index)
df2=df2.diff().fillna(df2.iloc[0]).astype('int')

rcParams['figure.figsize'] = 12, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def create_sequences(data, seq_length):
    xs=[]
    ys=[]
    for i in range(len(data)-seq_length):
        x=data.iloc[i:(i+seq_length)]
        y=data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seq_length=5
X, y=create_sequences(df2, seq_length)

train_size=int(675*0.8)
X_trained_data=X[:train_size]
y_trained_data=y[:train_size]
X_value_data=X[train_size:train_size+67]
y_value_data=y[train_size:train_size+67]
X_test_data=X[train_size+67:]
y_test_data=y[train_size+67:]
MIN = X_trained_data.min()
MAX = X_trained_data.max()

def MinMaxScale(array, min, max):
    return (array-min)/(max-min)
X_trained_data = MinMaxScale(X_trained_data, MIN, MAX)
y_trained_data = MinMaxScale(y_trained_data, MIN, MAX)
X_value_data = MinMaxScale(X_value_data, MIN, MAX)
y_value_data = MinMaxScale(y_value_data, MIN, MAX)
X_test_data = MinMaxScale(X_test_data, MIN, MAX)
y_test_data = MinMaxScale(y_test_data, MIN, MAX)

def make_tensor(array):
    return torch.from_numpy(array).float()
X_trained_data=make_tensor(X_trained_data)
y_trained_data=make_tensor(y_trained_data)
X_value_data=make_tensor(X_value_data)
y_value_data=make_tensor(y_value_data)
X_test_data=make_tensor(X_test_data)
y_test_data=make_tensor(y_test_data)

class Covid19(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(Covid19, self).__init__()
        self.n_hidden=n_hidden
        self.seq_len=seq_len
        self.n_layers=n_layers
        self.lstm=nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers
        )
        self.linear=nn.Linear(in_features=n_hidden, out_features=1)
    def reset_hidden_state(self):
        self.hidden=(
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    def forward(self, sequences):
        lstm_out, self.hidden=self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step=lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred=self.linear(last_time_step)
        return y_pred

def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose = 10, patience = 10):
    loss_fn=torch.nn.L1Loss() #
    optimiser=torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist=[]
    val_hist=[]
    for t in range(num_epochs):
        epoch_loss=0

        for idx, seq in enumerate(train_data): 
            model.reset_hidden_state()

            seq=torch.unsqueeze(seq, 0)
            y_pred=model(seq)
            loss=loss_fn(y_pred[0].float(), train_labels[idx])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss+=loss.item()
        train_hist.append(epoch_loss/len(train_data))

        if val_data is not None:
            with torch.no_grad():
                val_loss=0
                for val_idx, val_seq in enumerate(val_data):
                    model.reset_hidden_state()
                    val_seq=torch.unsqueeze(val_seq, 0)
                    y_value_data_pred=model(val_seq)
                    val_step_loss=loss_fn(y_value_data_pred[0].float(), val_labels[val_idx])
                    val_loss+=val_step_loss
            val_hist.append(val_loss/len(val_data))
            if t % verbose==0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')
            if (t % patience==0) & (t!=0):
                if val_hist[t-patience]<val_hist[t] :
                    print('\n Early Stopping')
                    break

        elif t%verbose==0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')

    return model, train_hist, val_hist

model=Covid19(
    n_features=1,
    n_hidden=4,
    seq_len=seq_length,
    n_layers=1
)
model, train_hist, val_hist=train_model(
    model,
    X_trained_data,
    y_trained_data,
    X_value_data,
    y_value_data,
    num_epochs=100,
    verbose=10,
    patience=50
)
pred_dataset=X_test_data

with torch.no_grad():
    preds=[]
    for _ in range(len(pred_dataset)):
        model.reset_hidden_state()
        y_test_data_pred=model(torch.unsqueeze(pred_dataset[_], 0))
        pred=torch.flatten(y_test_data_pred).item()
        preds.append(pred)

plt.plot(df2.index[-len(y_test_data):], np.array(y_test_data)*MAX, label='The actual number of confirmed cases')
plt.plot(df2.index[-len(preds):], np.array(preds)*MAX, label='prediction graph')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('static/photo/result.png')
