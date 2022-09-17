import pandas as pd
import torch
import numpy as np
import tqdm
from torch import nn

train_data = pd.read_csv('./dataset/assistments2009/train.csv')
test_data = pd.read_csv('./dataset/assistments2009/test.csv')
df_item = pd.read_csv('./dataset/assistments2009/item.csv')

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']),  np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))
user_n, item_n, knowledge_n


def parse_all_seq(students,data):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.user_id == student_id])
        all_sequences.extend([student_sequence])
    return all_sequences
def parse_student_seq(student):
    seq = student.item_id
    q = [item2knowledge[q] for q in seq]
    q = sum(q, [])
    return q

train_sequences = parse_all_seq(train_data.user_id.unique(),train_data)
test_sequences = parse_all_seq(test_data.user_id.unique(),test_data)


def sequence_length(sequence):
    l = []
    for i in range(len(sequence)):
        le = len(sequence[i])
        l.extend([le])
    return l


train_veclen = sequence_length(train_sequences)
test_veclen = sequence_length(test_sequences)

MAX_STEP = 50
NUM_QUESTIONS = 110
def encode_onehot(sequences, max_step, num_questions):
    result = []
    for q in tqdm.tqdm(sequences, 'convert to one-hot format: '):
        length = len(q)
        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        onehot = np.zeros(shape=[length + mod, num_questions])
        for i, q_id in enumerate(q):
            index = int(q_id - 1)
            onehot[i][index] = 1
        result = np.append(result, onehot)
    return result.reshape(-1, max_step, num_questions)
# reduce the amount of data for example running faster
train_data = encode_onehot(train_sequences, MAX_STEP, NUM_QUESTIONS)
test_data = encode_onehot(test_sequences, MAX_STEP, NUM_QUESTIONS)


train_tensor = torch.tensor(train_data)
train_tensor1 = train_tensor.to(torch.float32)
test_tensor = torch.tensor(test_data)
test_tensor = test_tensor.to(torch.float32)
train_realtensor = train_tensor1[:,1:train_tensor1.size(axis=1),:]
train_tensor = train_tensor1[:,0:train_tensor1.size(axis=1)-1,:]


NUM_QUESTIONS = 110
BATCH_SIZE = 64
HIDDEN_SIZE = 1
NUM_LAYERS = 1
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=110, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层
        self.softmax = nn.Softmax()
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        x = self.softmax(x)
        return x
lstm_model = LstmRNN(NUM_QUESTIONS, 1 , output_size=110, num_layers=1)  # 20 hidden units
print('LSTM model:', lstm_model)
print('model.parameters:', lstm_model.parameters)


device = torch.device("cpu")
if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
prev_loss = 1000
max_epochs = 2000
train_tensor = train_tensor.to(device)
for epoch in range(max_epochs):
    output = lstm_model(train_tensor).to(device)
    loss = criterion(output, train_realtensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < prev_loss:
        torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
        prev_loss = loss
    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch + 1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

def get_lastvector(predictive, vector_len):
    all = []
    alllen = []
    p = 0
    for i in range(len(vector_len)):
        a = vector_len[i]
        p = a + p
        alllen.extend([p])
    for j in range(len(alllen)):
        b = alllen[j]
        pre = predictive[b-1]
        all.extend([pre])
    return all

# predictive_for_training = lstm_model(train_tensor1)
# predictive_for_training = predictive_for_training.view(-1, NUM_QUESTIONS).tolist()
# train_result = get_lastvector(predictive_for_training, train_veclen)
#  lstm_model = lstm_model.eval()
predictive_for_testing = lstm_model(test_tensor)
predictive_for_testing = predictive_for_testing.view(-1, NUM_QUESTIONS).tolist()
test_result = get_lastvector(predictive_for_testing, test_veclen)
result = pd.DataFrame(test_result)
result.to_csv('./result/Assist2009_result.csv', encoding='gbk')