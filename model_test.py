
# coding: utf-8

# In[35]:


import csv
import re
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk
nltk.download('punkt')
import time

import pandas as pd
import pickle


# In[36]:


def load_and_preprocess(filename):
    #./data_toeic/5000+2000.csv
    f = open(filename, 'r',encoding='utf-8')
    csvReader = csv.reader(f)

    raw_data = []
    sentences = []
    label = []
    for row in csvReader:
        raw_data.append(row)
    f.close()

    del raw_data[0]
    
    print("총 문제 개수: ", len(raw_data))
    print("총 문장 개수: ", len(raw_data)*4)
    for row_index, row in enumerate(raw_data):
        row_sentence = []
        row_label = []
        hype1 = 0
        hype2 = 0
        for item_index, item in enumerate(row):
            # .뒤에 나오는 것 다 없애기 .이 여러 개 있을지 모르니 마지막 .을 이용하기. 두 문장인 경우 .과 ?과 !이 존재
            # 특수문자 앞에 공백으로 하기
            # 공백 없애기
            # 공백 두개, 세개 -> 한개로 바꾸기
            # _____,----- 연달아 있을시 index 찾아 양쪽 공백 만들기
            # 소문자로 바꾸기

            if item_index == 0:
                #
                index_list = []
                index_of_dot = item.rfind('.')
                index_of_question = item.rfind('?')
                index_of_surprise = item.rfind('!')
                index_list.append(index_of_dot)
                index_list.append(index_of_question)
                index_list.append(index_of_surprise)
                standard = max(index_list)+1
                if standard >0:
                    item = item[:standard]   

                #
                item = item.strip()
                #
                item = item.replace("  ", " ")
                item = item.replace("  ", " ")

                #
                index_of_hype1 = item.find('__')
                index_of_hype2 = item.find('--')
                index_of_hype3 = item.rfind('__')
                index_of_hype4 = item.rfind('--')
                if index_of_hype1 > index_of_hype2:
                    if index_of_hype3 > index_of_hype4:
                        hype1 = index_of_hype1
                        hype2 = index_of_hype3+1
                elif index_of_hype2 > index_of_hype1:
                    if index_of_hype4 > index_of_hype3:
                        hype1 = index_of_hype2
                        hype2 = index_of_hype4+1

                #----- => hype1 = 0 , hype2 = 4
                if hype1 == 0:
                    if (hype2+1) < len(item):
                        if item[hype2+1] != ' ':
                            item = item[:hype2+1] + ' ' + item[hype2+1:]
                else:
                    if item[hype1-1] != ' ':
                        item = item[:hype1] + ' ' + item[hype1:]
                        hype1 = hype1+1
                        hype2 = hype2+1
                    if (hype2+1) < len(item):
                        if item[hype2+1] != ' ':
                            item = item[:hype2+1] + ' ' + item[hype2+1:]
                #
                item = item.lower()
                # 문장이 아닐 때
            else:
                 item = item.strip()
            raw_data[row_index][item_index] = item
       
        #sentence 4개 만들기
        row[0] = row[0].replace(row[0][hype1:hype2+1],"")
        for i in range(1,5):
            sentence = row[0][:hype1] + row[i] + row[0][hype1:]
            row_sentence.append(sentence)

        sentences.append(row_sentence)

        #label 만들기
        count = 0
        for i in range(1,5):
            if row[5] == row[i]:
                row_label.append(1)
                count+=1
            else:
                row_label.append(0)

        #모두 0일 경우 찾아서 1 넣어 주기
        if count!=1:
            if 'A' in row[5]:
                row_label[0]=1
            elif 'B' in row[5]:
                row_label[1]=1
            elif 'C' in row[5]:
                row_label[2]=1
            elif 'D' in row[5]:
                row_label[3]=1

        #레이블에 추가해주기
        label.append(row_label)

    sentences= np.array(sentences)
    sentences= sentences.flatten()
    label = np.array(label)
    label = label.flatten()
    sentences = sentences.tolist()
    label = label.tolist()
    
    sentences_train = sentences[:27000]
    label_train = label[:27000]
    sentences_valid = sentences[27000:28000]
    label_valid = label[27000:28000]
    return sentences_train, label_train, sentences_valid, label_valid


# In[37]:


def pre_process_test(sentence,a1,a2,a3,a4):

    raw_data = []
    sentences = []
    
    temp =[]
    temp.append(sentence)
    temp.append(a1)
    temp.append(a2)
    temp.append(a3)
    temp.append(a4)
    raw_data.append(temp)
    
    print("총 문제 개수: ", len(raw_data))
    print("총 문장 개수: ", len(raw_data)*4)
    for row_index, row in enumerate(raw_data):
        row_sentence = []
        row_label = []
        hype1 = 0
        hype2 = 0
        for item_index, item in enumerate(row):
            # .뒤에 나오는 것 다 없애기 .이 여러 개 있을지 모르니 마지막 .을 이용하기. 두 문장인 경우 .과 ?과 !이 존재
            # 특수문자 앞에 공백으로 하기
            # 공백 없애기
            # 공백 두개, 세개 -> 한개로 바꾸기
            # _____,----- 연달아 있을시 index 찾아 양쪽 공백 만들기
            # 소문자로 바꾸기

            if item_index == 0:
                #
                index_list = []
                index_of_dot = item.rfind('.')
                index_of_question = item.rfind('?')
                index_of_surprise = item.rfind('!')
                index_list.append(index_of_dot)
                index_list.append(index_of_question)
                index_list.append(index_of_surprise)
                standard = max(index_list)+1
                if standard >0:
                    item = item[:standard]   

                #
                item = item.strip()
                #
                item = item.replace("  ", " ")
                item = item.replace("  ", " ")

                #
                index_of_hype1 = item.find('__')
                index_of_hype2 = item.find('--')
                index_of_hype3 = item.rfind('__')
                index_of_hype4 = item.rfind('--')
                if index_of_hype1 > index_of_hype2:
                    if index_of_hype3 > index_of_hype4:
                        hype1 = index_of_hype1
                        hype2 = index_of_hype3+1
                elif index_of_hype2 > index_of_hype1:
                    if index_of_hype4 > index_of_hype3:
                        hype1 = index_of_hype2
                        hype2 = index_of_hype4+1

                #----- => hype1 = 0 , hype2 = 4
                if hype1 == 0:
                    if (hype2+1) < len(item):
                        if item[hype2+1] != ' ':
                            item = item[:hype2+1] + ' ' + item[hype2+1:]
                else:
                    if item[hype1-1] != ' ':
                        item = item[:hype1] + ' ' + item[hype1:]
                        hype1 = hype1+1
                        hype2 = hype2+1
                    if (hype2+1) < len(item):
                        if item[hype2+1] != ' ':
                            item = item[:hype2+1] + ' ' + item[hype2+1:]
                #
                item = item.lower()
                # 문장이 아닐 때
            else:
                 item = item.strip()
            raw_data[row_index][item_index] = item
        #sentence 4개 만들기
        row[0] = row[0].replace(row[0][hype1:hype2+1],"")
        for i in range(1,5):
            sentence = row[0][:hype1] + row[i] + row[0][hype1:]
            row_sentence.append(sentence)

        sentences.append(row_sentence)

    sentences= np.array(sentences)
    sentences= sentences.flatten()
    
    return sentences


# In[38]:


def load_example(filename):
    df = pd.read_csv(filename, error_bad_lines=False)
    sentences = df['SentimentText'][:120000]
    label = df['Sentiment'][:120000]
    R_S = []
    R_L = []
    
    for s in sentences:
        R_S.append(s)
    for l in label:
        R_L.append(l)
    R_S_train = R_S[:100000]
    R_L_train = R_L[:100000]
    R_S_test = R_S[100000:120000]
    R_L_test = R_L[100000:120000]
    print(R_S[:5])
    print(R_L[:5])
    return R_S_train,R_L_train, R_S_test, R_L_test


# In[39]:


#가공만 된 문장 indexing, embedding 전 입니다.
class ToeicDataset(Dataset):
    def __init__(self, sentences, label):
        self.sentences = sentences
        self.label = label
        self.len = len(self.label)
        
        self.label_list = list(sorted(set(self.label)))
        
    def __getitem__(self, index):
        return self.sentences[index], self.label[index]
    def __len__(self):
        return self.len
    
    def get_labels(self):
        return self.label_list
    
    def get_label(self,id):
        return self.label_list[id]
    def get_label_id(self,label):
        return self.label_list.index(label)


# In[40]:


def make_vocab(sentences):
    vocab = set()
    for index, sentence in enumerate(sentences):
        temp = nltk.word_tokenize(sentence)
        vocab.update(temp)
    word_to_ix = {word:i+1 for i, word in enumerate(vocab)}
    word_to_ix['_PAD'] = 0
    word_to_ix['_UNK'] = 1
    vocabsize = len(word_to_ix)
    return vocab, vocabsize, word_to_ix


# In[41]:


#make torch
def make_variables(sentences, label,vocabulary):
    
    final_sentences = []
    
    #tokenizing
    for index, sentence in enumerate(sentences):
        final_sentences.append(nltk.word_tokenize(sentence))
    
    #indexing
    for index, sentence in enumerate(final_sentences):
        final_sentences[index]=[vocabulary[word] for word in sentence]

    #각자의 seq_length 구하기 (미니배치별로 진행)
    seq_lengths = []
    for sentence in final_sentences:
        seq_lengths.append(len(sentence))
    seq_lengths = torch.LongTensor(seq_lengths)

#     print("패딩전")
#     print(final_sentences[:5])
#     print(seq_lengths[:5])
#     print(label[:5])
    return padding_tensor_sorting(final_sentences,seq_lengths,label)


# In[42]:


def padding_tensor_sorting(sentences, seq_lengths, label):
    seq_tensor = torch.zeros((len(sentences), seq_lengths.max())).long()
    print('seq_tensor(max):' , seq_lengths.max())
    for idx, (seq, seq_len) in enumerate(zip(sentences, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    
    if len(seq_tensor) !=1:
        seq_lengths, perm_idx = seq_lengths.sort(0,descending=True)
        seq_tensor = seq_tensor[perm_idx]
    
    
    target = torch.tensor(label, dtype=torch.long)
    
    if len(seq_tensor) !=1 and len(target)!=0:
        target = target[perm_idx]
        
#     print("패딩후")
#     print(seq_tensor[:5])
#     print(seq_lengths[:5])
#     print(target[:5])
    return create_variable(seq_tensor),         create_variable(seq_lengths),         create_variable(target)


# In[43]:


def create_variable(tensor):
    #tensor를 gpu 이용 가능한지
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# In[44]:


def train():
    total_loss = 0
    for i, (sentences, label) in enumerate(train_loader,1):
        input, seq_lengths, target = make_variables(sentences, label,word_to_ix)
        
        output = classifier(input, seq_lengths)
        
        loss = criterion(output, target)
        
        total_loss +=loss.data[0]
        
        classifier.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%10 == 0:
            print("output, target", output, target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                epoch,  i *
                len(sentences), len(train_loader.dataset),
                100. * i * len(sentences) / len(train_loader.dataset),
                total_loss / i * len(sentences)))
        
    return total_loss


# In[45]:


# def validation_make_vocab(sentences_test):
#     vocab = set()
#     for index, sentence in enumerate(sentences_test):
#         temp = nltk.word_tokenize(sentence)
#         vocab.update(temp)
#     word_to_ix2 = word_to_ix
    
#     length = len(word_to_ix2)
    
#     for i,word in enumerate(vocab,length):
#         if word not in word_to_ix2:
#             word_to_ix2[i]=word
    
#     vocabsize = len(word_to_ix2)
#     return vocab,vocabsize,word_to_ix2


# In[46]:


def validation():
    correct =0
    test_data_size = len(test_loader.dataset)
    
    print(test_loader)
    for i,(sentences, label) in enumerate(test_loader,1):
        
        input, seq_lengths, target = make_variables(sentences, label,word_to_ix_v)
        
        output = valid_model(input, seq_lengths)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('prediction, ', pred)
        print('target, ',target)
    print('correct: ', correct)
    print('전체 싸이즈: ', test_data_size)
    
#     print('evaluating trained model...')
    
#     input,seq_lengths, target = make_variables([sentences],[])
#     output = classifier(input, seq_lengths)
#     print(output)
#     pred = output.data.max(1, keepdim = True)[1]
#     print(pred)
#     label_id = pred.cpu().numpy()[0][0]
#     print(label_id)
#     print(sentences, " is ", train_dataset.get_label(label_id))


# In[47]:


def one_problem_test(sentence, a1,a2,a3,a4):
    
    sentences = pre_process_test(sentence,a1,a2,a3,a4)
    print(sentences)
    #단어집 불러오기
    file = open("vocabulary", "rb")
    vocabulary = pickle.load(file)
    file.close()
    
    vocab = set()
    temp = nltk.word_tokenize(sentence)
    vocab.update(temp)
    
    length = len(vocabulary)
    
    for i, word in enumerate(vocab):
        vocabulary[word]= length+i
    vocabsize_t = len(vocabulary)
    print(vocabsize_t)
    test_model = torch.load('saved_gru')
    test_model.n_vocab = vocabsize_t
    test_model.embed = nn.Embedding(valid_model.n_vocab, valid_model.embed_dim)
    
    input, seq_lengths, target = make_variables(sentences,[],vocabulary)
    
    output = test_model(input,seq_lengths)
    pred = output.data.max(1,keepdim=True)[1]
    print("softmax 직후...", output)
    print("1개만 고른후...", pred)


# In[48]:


class myModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, bidirectional=True ,dropout_p=0.2):
        super(myModel, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_directions = int(bidirectional) + 1
        
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
#         self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim,
#                             num_layers=self.n_layers,
#                             dropout=dropout_p,
#                             batch_first=True)
        self.lstm = nn.GRU(self.embed_dim, self.hidden_dim,
                            self.n_layers,
                            #dropout=dropout_p,
                            batch_first=True)
#         self.relu = nn.ReLU()
        self.out = nn.Linear(self.hidden_dim, self.n_classes)
#         self.fc1 = nn.Linear(self.hidden_dim, 50)
#         self.fc2 = nn.Linear(50, self.n_classes)
    def forward(self, x, seq_lengths):
        
        x = x.t()
        sen_len = x.size(0)
        batch_size = x.size(1)
        print(self.n_layers, self.hidden_dim, self.n_vocab, self.embed_dim, self.n_classes, self.n_directions)
        
        embedded = self.embed(x)
        
#         print(embedded)
        lstm_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())
        self.hidden = self._init_hidden(batch_size)
        self.lstm.flatten_parameters()
        
        lstm_out, self.hidden = self.lstm(lstm_input)
        lstm_out, lengths = pad_packed_sequence(lstm_out)
        
#         h_t = self.dropout(self.hidden[-1])
#         logit = self.out(h_t[-1])
        logit = self.out(self.hidden[-1])
#         print(logit)
        return logit
    
    def _init_hidden(self, batch_size):
        hidden = torch.zeros((self.n_layers, self.n_directions,
                             batch_size, self.hidden_dim))
        return create_variable(hidden)


# In[49]:


if __name__ =="__main__":
    #
    BATCH_SIZE = 64
    EPOCHS = 20
    learning_rate =0.01
    torch.manual_seed(42)
    
    #load data
#     sentences, label, sentences_test, label_test = load_and_preprocess('./data_toeic/5000+2000.csv')
    sentences, label, sentences_test, label_test = load_example('./Sentiment Analysis Dataset.csv')
    train_dataset = ToeicDataset(sentences, label)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_dataset = ToeicDataset(sentences_test,label_test)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    
    # hyper parameter, user data
    n_layer = 2
    n_classes = 2
    hidden_dim = 100
    embed_dim = 64
    bidirectional = True
    dropout_p = 0.4
    vocab,vocabsize,word_to_ix = make_vocab(sentences)
    sentence_temp = sentences
    sentence_temp +=sentences_test
    vocab_v,vocabsize_v,word_to_ix_v = make_vocab(sentence_temp)
    file=open("vocabulary","wb")
    pickle.dump(word_to_ix, file)
    file.close()
    #make model
    classifier = myModel(n_layer, hidden_dim, vocabsize, embed_dim, n_classes, bidirectional, dropout_p)
    
    #classifier and cuda
    #병렬 처리
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        classifier = nn.DataParallel(classifier)
    if torch.cuda.is_available():
        classifier.cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 3):
        train()

    


# In[50]:


torch.save(classifier,'saved_gru')

valid_model = torch.load('saved_gru')
valid_model.n_vocab = vocabsize_v
valid_model.embed = nn.Embedding(valid_model.n_vocab, valid_model.embed_dim)
print('validation!!!!')
print(len(vocab))
print(vocabsize)
print(len(word_to_ix))
print(len(vocab_v))
print(vocabsize_v)
print(len(word_to_ix_v))
print(valid_model)
print(valid_model.n_vocab)
validation()


# In[51]:


one_problem_test('____________ harvesting techniques have been instrumental in helping many of the local small farmers to become more self-sufficient. ',
                ' improve', 'improved', 'improvement', 'improves')


# In[55]:


print(word_to_ix_v['_UNK'])


# In[ ]:


list_ = [1,2]
list2 = [3,4]
list_ +=list2
list_

