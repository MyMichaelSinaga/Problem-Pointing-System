# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:13:23 2018

@author: Seira
"""
import cv2
import sys
import nltk
import string
import re
import pickle
import csv
import codecs
import unicodedata
import os
import torch.nn.functional as F
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication,QDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5 import QtGui, QtWidgets, QtCore
from pandas import DataFrame, read_csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn import feature_extraction
from sklearn.cluster import KMeans
from pylab import figure, axes, pie, title, show
from torch.autograd import Variable
from torch import optim


USE_CUDA = False
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 500

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS
      
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return self.index_word
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
    

class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
    


class Tampilan(QDialog):    
    def __init__(self):
        super(Tampilan,self).__init__()
        loadUi('E:/Tugas Akhir 2/produk/QT/GUI/ProblemPoint.ui',self)
        self.setWindowTitle('Problem Pointing For Epoch 10.000')

        #Table Presentation
        self.model = QtGui.QStandardItemModel(self)
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.setGeometry(30, 295, 430, 350)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        
    def on_btnBrowse_clicked(self):
        filePath = QtWidgets.QFileDialog.getOpenFileName(self,'Open File')
        self.linePath.setText(str(filePath[0]))
        #data = self.linePath.text()
    
    @QtCore.pyqtSlot()
    def on_btnAllPro_clicked(self):
        data = self.linePath.text()
        
        #Show Result Classfier Process
        self.showTable()
        
        #Show Result Cluster Process
        self.loadImage('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/plotlib.png')
        self.loadImage2('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/bar.png')
        
        #Show Result Summarize Process
        self.testSu()
        self.lblSum0.setText(str('Summary of Cluster 1'))
        input0 = open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/out_cluster0.txt').read()
        self.lblSumma0.setText(str(input0))
        self.lblSum1.setText(str('Summary of Cluster 2'))
        input1 = open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/out_cluster1.txt').read()
        self.lblSumma1.setText(str(input1))
        self.lblSum2.setText(str('Summary of Cluster 3'))
        input2 = open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/out_cluster2.txt').read()
        self.lblSumma2.setText(str(input2))
        self.lblSum3.setText(str('Summary of Cluster 4'))
        input3 = open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/out_cluster3.txt').read()
        self.lblSumma3.setText(str(input3))
        self.lblSum4.setText(str('Summary of Cluster 5'))
        input4 = open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/out_cluster4.txt').read()
        self.lblSumma4.setText(str(input4))
        
        self.successTpr.setText(str('All Process Has Been Success'))
                       
    def loadImage(self, fname):
        self.image=cv2.imread(fname)
        self.displayImage()
    
    def loadImage2(self, fname):
        self.image=cv2.imread(fname)
        self.displayImage2()
    
    def displayImage(self):
        qformat = QImage.Format_Indexed8
        
        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat = QImage.Format_RGB8888
            else:
                qformat = QImage.Format_RGB888
        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)
        img = img.rgbSwapped()
        self.imgPlot.setPixmap(QPixmap.fromImage(img))
        self.imgPlot.setAlignment(QtCore.Qt.AlignCenter)
        self.imgPlot.setScaledContents(True)
        
    def displayImage2(self):
        qformat = QImage.Format_Indexed8
        
        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat = QImage.Format_RGB8888
            else:
                qformat = QImage.Format_RGB888
        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)
        img = img.rgbSwapped()
        self.imgPlot_2.setPixmap(QPixmap.fromImage(img))
        self.imgPlot_2.setAlignment(QtCore.Qt.AlignCenter)
        self.imgPlot_2.setScaledContents(True)
        
        
    def textPreprocessing(self):
        print(self.linePath.text())
        data = pd.read_csv(self.linePath.text(),encoding = 'unicode_escape')
       
        #data = pd.read_csv('E:/Tugas Akhir 2/produk/2. preprocessing/preprocessing/dataFixE1.csv',encoding = 'unicode_escape')
        file1 = open("E:/Tugas Akhir 2/produk/QT/All_Data/Korpus Normalization/normalisasi.txt",'r')
        contractions = {}
        clean_texts = []

        pattern=r"(@[a-zA-Z0-9_]+)"
        for i in range(len(data)):
            data['text'].iloc[i] = re.sub(pattern,' ', data['text'].iloc[i], flags=re.MULTILINE)

        pattern=r"http\S+|https:S+"
        for i in range(len(data)):
            data['text'].iloc[i] = re.sub(pattern,'', data['text'].iloc[i], flags=re.MULTILINE)

        pattern=r"https"
        for i in range(len(data)):
            data['text'].iloc[i] = re.sub(pattern,'', data['text'].iloc[i], flags=re.MULTILINE)

        pattern=r'RT'
        for i in range(len(data)):
            data['text'].iloc[i] = re.sub(pattern,'', data['text'].iloc[i], flags=re.MULTILINE)

        pattern=r'[^a-zA-Z]'
        for i in range(len(data)):
            data['text'].iloc[i] = re.sub(pattern,' ', data['text'].iloc[i], flags=re.MULTILINE)

        data = data.apply(lambda x: x.astype(str).str.lower())

        for s in file1.readlines():
            sep = s.find(':')
            key = s[:sep]
            value = s[sep+1:len(s)]
            value = value.strip('\n')
            contractions[key] = value

        def clean_textNormalization(text, remove_stopwords = True):
            if True:
                text = re.findall(r"[\w']+", text)
                new_text = []
                for word in text:
                    if word in contractions:
                        new_text.append(contractions[word])
                    else:
                        new_text.append(word)
                text = " ".join(new_text)
            return text

        for text in data.text:
            clean_texts.append(clean_textNormalization(text))

        Data_Tweetwer = {'no':data['no'],'text':clean_texts}
        frame = pd.DataFrame(Data_Tweetwer , columns = ['no', 'text'], index=None)
        frame.to_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/2. norm.csv')
        datas = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/2. norm.csv')

        datas = datas.apply(lambda x: x.astype(str).str.lower())

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        for i in range(len(datas)):
            sent=datas['text'].iloc[i]
            output = stemmer.stem(sent)
            datas['text'].iloc[i]=output

        stopwords.words("indonesian")
        stopw = set(stopwords.words("indonesian"))
        for i in range(len(datas)):
            sentence=datas['text'].iloc[i]
            join=" ".join([word for word in sentence.split() if word not in stopw])
            datas['text'].iloc[i]=join

        datas=datas[~data['text'].duplicated()]
        datas=datas.reset_index(drop=True)
        datas.dropna()
        datas.to_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/cleanKriminal1.csv', index=False)
        
        
    def textPreprocess(self):
        kembali = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/2. norm.csv')
        pres = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/cleanKriminal1.csv')
        kamus = {}
        for index, row in kembali.iterrows():
            kamus[row['no']] = row['text']
        inputt = {}
        for index, row in pres.iterrows():
            inputt[row['no']] = row['text'] 
        for i in inputt:
            inputt[i] = kamus[i]
        
        tweet = {'text':inputt}

        frame = pd.DataFrame(tweet , columns = ['text'])
        #print (frame)
        frame.head()
        frame.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/kembalikan.csv")
        
        print('sukses')
        
        
    def textClassification(self):
        df = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/Model/cleanKriminalLabel.csv', encoding='utf-8')
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(df.text)
        count_vect.vocabulary_
        
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        
        #load model
        filename1='E:/Tugas Akhir 2/produk/QT/All_Data/Model/Model_Klasifikasi.pickle'
        loaded_model = pickle.load(open(filename1, 'rb'))
        df1 = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/cleanKriminal1.csv', encoding='utf-8')
        X_testcv = count_vect.transform(df1.text)
        predict_test = loaded_model.predict(X_testcv)
        
        columns = defaultdict(list)
        with open('E:/Tugas Akhir 2/produk/QT/All_Data/Model/dataTestAwalClean.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for (k,v) in row.items(): 
                    columns[k].append(v)
        text = df1['text']
        no=df1['no']
        tweet = {'no':no,'text':text, 'sentiment':predict_test}
        frame = pd.DataFrame(tweet , columns = ['no','text','sentiment'])
        frame.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasil Sentiment.csv", index=False)
        hasil = pd.read_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasil Sentiment.csv", encoding="utf-8")
        hasil.set_index("sentiment", inplace = True)
        negative = hasil.drop('pos')
        negative.to_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasilSentimentNegative.csv')
        
        kembalikan = pd.read_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataTextProcessing/kembalikan.csv")
        frameNo = frame['no']
        frameSentiment = frame['sentiment']
        kembalikanText = kembalikan['text']

        tweet = {'no':frameNo, 'text':kembalikanText, 'sentiment':frameSentiment}
        frames = pd.DataFrame(tweet , columns = ['no','text','sentiment'])
        frames.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/tampilkanSentiment.csv", index=False)
        
        frames1=pd.read_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/tampilkanSentiment.csv")
        frames1.set_index("sentiment", inplace=True)
        negative1 = frames1.drop('pos')
        negative1.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasilSentimentNegativeAsli.csv")
        
    def showTable(self):
        with open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasil Sentiment.csv') as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)
        
    def textClustering(self):
        data = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasilSentimentNegative.csv')
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                           min_df=0.13,
                                           use_idf=True, ngram_range=(1,3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
        print(tfidf_vectorizer.get_feature_names())

        #elbow
        cluster_range = range( 2, 20 )
        cluster_errors = []
        for num_clusters in cluster_range:
            clusters = KMeans (num_clusters)
            clusters.fit(tfidf_matrix)
            cluster_errors.append(clusters.inertia_)
        clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

        plt.figure(figsize=(12,6))
        plt.title('Elbow Graph')
        plt.xlabel('k-Cluster')
        plt.ylabel('Sum of squared error (SSE)')
        plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" ) #upayakan tampilkan dalam gamar plot
        plt.savefig('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/plotlib.png')
        
        num_clusters = tfidf_matrix.shape[1]
        km = KMeans(n_clusters = num_clusters)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()

        columns = defaultdict(list)
        with open('E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasilSentimentNegative.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for (k,v) in row.items():
                    columns[k].append(v)
        twet = columns['text']

        tweet = {'text':twet, 'cluster':clusters}
        frame = pd.DataFrame(tweet , columns = ['text', 'cluster'])

        frames= frame['cluster'].astype(int)
        frames1=frame.groupby('cluster').agg({'cluster': 'count'})
        frames1.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/jumlah.csv")

        jumlah=pd.read_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/jumlah.csv")
        jumlah.head()
        x = jumlah['cluster']
        y = jumlah['cluster.1']
        plt.clf()
        plt.bar(x,y)
        plt.title('Cluster Graph')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Data')
        plt.savefig('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/bar.png')
        
        frame.to_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/hasilCluster.csv')
        
        #back text
        frames = pd.read_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataSentimen/hasilSentimentNegativeAsli.csv")
        framesText = frames['text']
        frameCluster = frame['cluster']
        
        tweet = {'text':framesText, 'cluster':frameCluster}
        summary = pd.DataFrame(tweet , columns = ['text', 'cluster'])
        summary.to_csv("E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/hasilClusterAsli.csv", index=False)
        
        df11 = pd.read_csv('E:/Tugas Akhir 2/produk/QT/All_Data/DataCluster/hasilClusterAsli.csv')
        
        kamus = {}
        for index, row in df11.iterrows():
            kamus[row['cluster']] = []
        for index, row in df11.iterrows():
            kamus[row['cluster']].append(row['text'])
            
        for i in kamus:
            file = open('cluster'+str(i)+'.txt','w')
            for j in kamus[i]:
                file.write(str(j)+' ')
        
    
    def testSu(self):    
        def unicode_to_ascii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )

        # Lowercase, trim, and remove non-letter characters
        def normalize_string(s):
            s = unicode_to_ascii(s.lower().strip())
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s

        def read_langs(lang1, lang2, reverse=False):
            # Read the file and split into lines
            lines1 = open('E:/Tugas Akhir 2/produk/QT/New folder/%s.txt' % lang1).read().strip().split('\n')
            lines2 = open('E:/Tugas Akhir 2/produk/QT/New folder/%s.txt' % lang2).read().strip().split('\n')

            # Split every line into pairs and normalize
            pairs = [[normalize_string(lines1[x]), normalize_string(lines2[x])] for x in range(len(lines1))]

            # Reverse pairs, make Lang instances
            if reverse:
                pairs = [list(reversed(p)) for p in pairs]
                voc = Voc(lang1+lang2)
            else:
                voc = Voc(lang1+lang2)

            return voc, pairs

        MAX_LENGTH = 500

        def filter_pair(p):
            return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 

        def filter_pairs(pairs):
            return [pair for pair in pairs if filter_pair(pair)]


        def prepare_data(lang1_name, lang2_name, reverse=False):
            voc, pairs = read_langs(lang1_name, lang2_name, reverse)
            pairs = filter_pairs(pairs)

            for pair in pairs:
                voc.index_words(pair[0])
                voc.index_words(pair[1])

            return voc, pairs

        voc, pairs = prepare_data('2', '1', True)


        def indexes_from_sentence(voc, sentence):
            return [voc.word2index[word] for word in sentence.split(' ')]

        def variable_from_sentence(voc, sentence):
            indexes = indexes_from_sentence(voc, sentence)
            indexes.append(EOS_token)
            var = Variable(torch.LongTensor(indexes).view(-1, 1))

            if USE_CUDA: var = var.cuda()
            return var

        def variables_from_pair(pair):
            input_variable = variable_from_sentence(voc, pair[0])
            target_variable = variable_from_sentence(voc, pair[1])
            return (input_variable, target_variable)    

        decoder = torch.load('E:/Tugas Akhir 2/produk/QT/All_Data/Model/Rev_Dec2.pt')
        encoder = torch.load('E:/Tugas Akhir 2/produk/QT/All_Data/Model/Rev_Enc2.pt')


        def summarization(sentence, max_length=200):
            input_variable = variable_from_sentence(voc, sentence)
            input_length = input_variable.size()[0]

            # Run through encoder
            encoder_hidden = encoder.init_hidden()
            encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

            # Create starting vectors for decoder
            decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
            decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
            #if USE_CUDA:
                #decoder_input = decoder_input.cuda()
                #decoder_context = decoder_context.cuda()

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            # Run through decoder
            for di in range(max_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                if ni == EOS_token:
                    #decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(voc.index2word[ni])

                # Next input is chosen word
                decoder_input = Variable(torch.LongTensor([[ni]]))
                #if USE_CUDA: decoder_input = decoder_input.cuda()

            return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

        
        #cluster0
        filename = 'E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/cluster0.txt'
        with open(filename) as f:
            cluster0 = f.read()
            
        #cluster1
        filename1 = 'E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/cluster1.txt'
        with open(filename1) as f:
            cluster1 = f.read()
        
        #cluster2
        filename2 = 'E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/cluster2.txt'
        with open(filename2) as f:
            cluster2 = f.read()

        #cluster3
        filename3 = 'E:/Tugas Akhir 2/produk/QT/All_Data/DataSumma/cluster3.txt'
        with open(filename3) as f:
            cluster3 = f.read()
        
        
        def summarize(input_sentence):
            output_words, attentions = summarization(input_sentence)
            print('input =', input_sentence)
            print('output =', ' '.join(output_words))
            return (' '.join(output_words))    
    
def run():
    app=QApplication(sys.argv)
    widget = Tampilan()
    widget.show()
    sys.exit(app.exec_())
    
run()