from nltk.corpus import stopwords
import re
import torch
from torch import nn
import time
import random
import numpy as np

class CBOW(nn.Module):

    def __init__(self, vocab_size, hidden_size, context_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(vocab_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, inputs):
        one_hot = torch.zeros((len(inputs),vocab_size)).to(device)
        one_hot[np.repeat([i for i in range(len(inputs))],context_size*2),inputs.view(-1)] = 1/(context_size*2)      
        out = self.layer1(one_hot)
        out = self.relu1(out)
        out = self.layer2(out)
        return out
    

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return idxs

def shuffle_batches(data, batch_size = 128):
        
    random.shuffle(data)
        
    n_batch = int(len(data)/batch_size)
        
    batches = []
    row_count = 0
    for i in range(n_batch):
            
        train_data = []
        train_labels = []
        
        # this can be much faster... but training is the bottleneck
        for j in range(batch_size):
                
            train_data.append(make_context_vector(data[row_count][0], word_to_ix))
            train_labels.append(make_context_vector([data[row_count][1]], word_to_ix))
                
            row_count += 1
        batches.append((torch.tensor(train_data), torch.tensor(train_labels)))
      
    return batches
    

if __name__ == '__main__':  
    
    context_size = 2
    
    with open('data/nytimes_news_articles.txt', encoding='utf8') as f:
        lines = f.read().splitlines()
    
    lines = lines[0:20000]
    stopwords_dict = stopwords.words('english')
    articles = []
    num_articles = 0
    for line in lines:
        if line[0:4] == 'URL:':
            articles.append('')
            num_articles += 1
        
        elif line == '':
            continue
        
        else:
            for word in line.split(' '):
                clean_word = re.sub('[^A-Za-z0-9]+', '', word).lower() 
                if clean_word in stopwords_dict or clean_word == '' or clean_word == ' ':
                    continue
                else:
                    articles[num_articles-1] += clean_word+' '
        
    idx = 0
    for article in articles:
        articles[idx] = article[:-1]
        idx += 1
    
    vocab = set(' '.join(articles).split(' '))
    vocab_size = len(vocab)
    
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {ix:word for ix, word in enumerate(vocab)}
    
    data = []
    for article in articles:
        raw_text = article.split(' ')
        for i in range(context_size, len(raw_text) - context_size):
            context = (
                [raw_text[i - j - 1] for j in range(context_size)]
                + [raw_text[i + j + 1] for j in range(context_size)]
            )
            target = raw_text[i]
            data.append((context, target))

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = 128    
    model = CBOW(vocab_size, hidden_size, context_size)
    model = model.to(device)
    
    criterion  = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    
    epochs = 200
    train_loss = []
    train_acc = []
    for epoch in range(epochs):
        
        start = time.time()
    
        # train loop
        loss = 0
        losses = 0
        correct = 0
    
        model.train()
        batch_size = 2**10
        batches = shuffle_batches(data, batch_size = batch_size)
        
        for x, y in batches:

            x = x.to(device)
            y = y.to(device)
            
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y.view(-1))
            
            pred = torch.argmax(out, axis=1)
            correct += torch.sum(pred == y.view(-1)).item()
            
            loss.backward()
            opt.step()
            
            losses += loss.item()
            
        train_loss.append(losses/len(batches))
        train_acc.append(correct/len(batches)/batch_size)
        
        end = round(time.time()-start)
        print(epoch, train_loss[-1], train_acc[-1], end)
    
    
    # embeddings1 = model.layer1.weight
    # embeddings2 = model.layer2.weight

    # embeddings = (embeddings2 + torch.transpose(embeddings1,1,0))/2
    # embeddings = torch.transpose(embeddings1,1,0)
    # embeddings = embeddings2
    # embeddings = embeddings.cpu().detach().numpy()
    
    # def find_nearest_analogy(a0,b0,a1,embeddings):
        
    #     item2 = b0 - a0 + a1
        
    #     cossim_add = []
    #     for i in range(len(embeddings)):
    #         item1 = embeddings[i]
            
    #         num = np.matmul(item2, item1.transpose())
    #         d1 = np.sqrt(np.matmul(item2, item2.transpose()))
    #         d2 = np.sqrt(np.matmul(item1, item1.transpose()))
    #         cossim_add.append(num/d1/d2)
    #     largest_indices = np.argsort(cossim_add)[-10:]
    #     top_5_words1 = [ix_to_word[i] for i in largest_indices]
    #     top_5_cossim1 = np.array(cossim_add)[largest_indices]
    #     for i in range(10):
    #         print(top_5_words1[i], round(top_5_cossim1[i],3))         
        
    #     return b1s
   
    # a0 = embeddings[word_to_ix['smart']]
    # b0 = embeddings[word_to_ix['smarter']]
    # a1 = embeddings[word_to_ix['small']]

    # word1_vec = b0
    
    # cossim1 = []
    # for i in range(len(embeddings)):
    #     compare_word = embeddings[i]
    #     ab = np.matmul(compare_word, word1_vec.transpose())
    #     a = np.sqrt(np.matmul(compare_word, compare_word.transpose()))
    #     b = np.sqrt(np.matmul(word1_vec, word1_vec.transpose()))
    #     cossim1.append(ab/(a*b))
    # largest_indices = np.argsort(cossim1)[-10:]
    # top_5_words1 = [ix_to_word[i] for i in largest_indices]
    # top_5_cossim1 = np.array(cossim1)[largest_indices]
    # for i in range(10):
    #     print(top_5_words1[i], round(top_5_cossim1[i],3))
        
        