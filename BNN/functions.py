from imports import *


def split_sequences(sequences, n_steps):
  """
    Parameters:
      sequences:  a time series sequence
      n_steps: number of steps used for prediction
    
    Returns: n_steps of the series as input value and a single label y
  """

    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix >= len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def multi_time(data, n_input, n_out):

    """
    Parameters:
      data:  a time series sequence
      n_input: number of steps used for prediction
      n_out: number of steps to predict
    
    Returns: n_input of the series as input value and n_out steps as label y
  """


    X,y = list(), list()
    in_start= 0

    for i in range(len(data)):

        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end])
            y.append(data[in_end:out_end])
            in_start +=1
    return np.array(X), np.array(y)


class myDataset(Dataset):
    """
    Parameters:
      Dataset: any dataset with item and labels
    
    Returns: items and labels
  """
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item,label

def plot_results(inputs, test_y, prediction, plus_error, minus_error, title = None, x_train = None):

  """
    Parameters:
      inputs: length of input vector used for the prediction 
      test_y: test dataset
      prediction: predictions of the network
      plus_error: vector of mean + 2 * standard deviation
      minus_error: vector of mean - 2 * standard deviation
      title: title of plot
      x_train: training data

    Returns: plot of prediction, real values and errors. Optional adds test_train split if x_train is provided
  """

  
  plt.rcParams["figure.figsize"] = (10, 7)
  x = np.arange(1, len(test_y)+1)

  plt.plot(x, test_y, label='Truth', color = 'blue')
  plt.plot(x, prediction, label='Prediction',color = 'orange')

  plt.fill_between(x, plus_error, minus_error, color='lightgray', alpha=0.5)

  if x_train is not None:
    plt.axvline(x=len(x_train), linestyle='-', color='purple', label='Train-Test Split')

  plt.xlabel('Months')
  plt.ylabel('Index')
  plt.title(title, color = 'blue')
  plt.legend()

  plt.show()


def rmse(test_data, prediction):

    """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      RMSE
  """
  
  mse = ((test_data - prediction)**2).sum()/len(prediction)
  return round(np.sqrt(mse), 3)

def mape(test_data, prediction):

  """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      MAPE
  """
  mape = abs((test_data-prediction)/test_data).sum()/len(prediction)
  return round(mape, 3)

def mae(test_data, prediction):
  
    """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      MAE
    """

    mae = abs(test_data - prediction).sum()/len(prediction)
    return round(mae,3)

def chi_square_cdf_probabilities(nssr, df = 1):
	"""
	Compute the probabilities of the values of nssr under a chi-square CDF with df dof.

	Parameters:
    	nssr (float or array-like): Non-centrality parameter(s).

	Returns:
    	float or ndarray: Probability value(s).
	"""
	return stats.chi2.cdf(nssr, df)



def count_elements_bigger_than_p(vector, p):
    """
    Count the number of elements in the vector that are bigger than p.

    Parameters:
        vector (list): The input vector.
        p (float): The value to compare against.

    Returns:
        int: Number of elements in the vector that are bigger than p.
    """
    return sum(1 for x in vector if x <= p)






