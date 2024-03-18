from import import *
from functions import data_process

df = yf.download('^SPGSCI', start="2010-09-23", end="2019-01-10", interval='1mo')
df.reset_index(inplace=True)

data_list = []
data_test_list = []

data_close = df.loc[:,'Adj Close']
data = data_close.values
test_data = data
data = data[:-40]



