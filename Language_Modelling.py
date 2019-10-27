
# coding: utf-8

# ## Build unigram dictionary

# In[45]:

import math
import nltk
from nltk import SimpleGoodTuringProbDist, FreqDist
import time
from matplotlib import pyplot as plt
import copy
from nltk.corpus import brown as b
from nltk import bigrams, ngrams, trigrams 
import re
import numpy as np
from math import log
start_time = time.clock()
global laplace_constant
laplace_constant=[1,0.1,0.01,0.001,0.0001]

fp_out=open("output","w")
x = [sent for sent in b.sents()[0:40000]]
temp = [[" ".join([re.sub('[^A-Za-z-\']+', '', item.lower()) for item in sent])] for sent in x]
temp = [[" ".join([item.strip() for item in sent])] for sent in temp]
temp = [[" ".join([re.sub('\'+', '', item) for item in sent])] for sent in temp]
sent =[[' '.join([re.sub(' +', ' ',item) for item in sent])] for sent in temp]
sent =[[' '.join([re.sub('-', ' ',item) for item in sent1])] for sent1 in sent]
sent = [[" ".join([item.strip() for item in sent1])] for sent1 in sent]
#sent = [[" ".join([filter(None,item) for item in sent1])] for sent1 in sent]
sentences = []
for i in range(len(sent)):
    sentences.append(''.join(sent[i]))


unigrams=[]

for elem in sentences:
    unigrams.extend(elem.split())
   
from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)
sort = unigram_counts.most_common()
values = [j for (i,j) in sort]
x=range(len(values))
plt.figure(figsize=(7,10))
plt.subplot(3,1,1)
plt.tight_layout()
plt.title('Unigram Zipf\'s Law', fontsize=18)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.semilogx(x,values)
for word in unigram_counts:
    unigram_counts[word]=float(unigram_counts[word])/unigram_total
lap_counts = copy.deepcopy(unigram_counts)
fp_out.write('Top 10 Unigrams\n')
fp_out.write('---------------\n') 
for i in range(10):
    fp_out.write(str(sort[i])+'\n')
    
global vocabulary_size 
vocabulary_size = len(set(unigrams))
vocabulary = set(unigrams)
#fp_out.write(unigram_counts)


# ## Build bigram dictionary

# In[46]:


def bigram_model(sentences):
    model={}
    count_biag={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    count_biag = copy.deepcopy(model)
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]/=tot_count
     
    return model, count_biag

bigram_counts, count_bigrams= bigram_model(sentences)
#lap_bigram = copy.deepcopy(bigram_counts)
count=0
top_ten=[]
for key,value in count_bigrams.items():
    for key1,value1 in value.items():
        top_ten.append([key,key1,value1])
        values.append(value1)

values.sort(reverse=True)
top_ten=sorted(top_ten,key=lambda x:x[2],reverse=True)
plt.subplot(3,1,2)
plt.tight_layout()
plt.title('Bigram Zipf\'s Law', fontsize=18)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.semilogx(range(len(values)),values)
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Top 10 Bigrams\n')
fp_out.write('--------------\n') 
count = 1
for i in range(10):
    fp_out.write(str(top_ten[i])+'\n')
    



# ## Build trigram dictionary

# In[47]:


def trigram_model(sentences):
    model={}
    counter={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    counter = copy.deepcopy(model)
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]/=tot_count
     
    return model,counter

trigram_counts,counter= trigram_model(sentences)
#fp_out.write(trigram_counts)
values_tri=[]
top_ten=[]
for key1,value in counter.items():
    for key2,value1 in value.items():
        values_tri.append(value1)
        top_ten.append([key1,key2,value1])
            
values_tri.sort(reverse=True)
top_ten=sorted(top_ten,key=lambda x:x[2],reverse=True)
plt.subplot(3,1,3)
plt.tight_layout()
plt.title('Trigram Zipf\'s Law', fontsize=18)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.semilogx(range(len(values_tri)),values_tri)
plt.savefig('zifhs.pdf')
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write("Top 10 Trigrams\n")
fp_out.write('---------------\n') 
count=0
for i in range(len(top_ten)):
    fp_out.write(str(top_ten[i])+'\n')
    count+=1
    if count==10:
        break


# ## Test Scores of each model

# In[48]:
fp = open("test_examples.txt","r")
test_sentences=[]
test_unigram_arr=[]
test_bigram_arr=[]
test_trigram_arr=[]

for line in fp:
    temp = line.replace('\n','')
    test_sentences.append(temp)


def perplexity(p_val,N):
    perp = (1/(math.pow(math.pow(10,p_val),N)))
    return perp

def log_likelihood(p_val):
    p_val=[log(y,10) for y in p_val]
    p_val=sum(p_val)
    return p_val
    
'''-------------------------------Simple Unigram------------------------------------'''
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Unigram test probabilities\n')
fp_out.write('--------------------------\n')
def unigram_output(test_sentences,unigram_counts):
    for elem in test_sentences:
        p_val=[unigram_counts[i] for i in elem.split()]
        N = 1/float(len(elem.split()))
        log_prob = log_likelihood(p_val)
        fp_out.write('The sequence '+ str(elem) +' has perplexity of '+ str(perplexity(log_prob,N))+'\n')
        fp_out.write('The sequence '+str(elem)+' has unigram log likelihood of '+ str(log_prob)+'\n')

unigram_output(test_sentences,unigram_counts)

'''---------------------------Laplace Smoothing Unigrams------------------------'''
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Laplace Unigram test probabilities\n')
fp_out.write('----------------------------------\n')
def laplace_smoothing_unigram_model(unigram_counts,unigram_total,laplace_constant):
    for word in unigram_counts:
        unigram_counts[word]=float(laplace_constant+unigram_counts[word])/((laplace_constant*vocabulary_size)+unigram_total)
    return unigram_counts

def laplace_ouput(test_sentences, laplace_unigram, laplace_constant):
    for elem in test_sentences:
        p_val=[laplace_unigram[i] for i in elem.split()]
        N = 1/float(len(elem.split()))
        log_prob = log_likelihood(p_val)
        fp_out.write('The sequence '+ elem+' has perplexity of '+str(perplexity(log_prob,N))+'\n')
        fp_out.write('The sequence '+elem+' has unigram log likelihood of '+ str(log_prob)+'\n')

for i in laplace_constant: 
    fp_out.write('For laplace constant = '+str(i)+'\n')
    fp_out.write('----------------------------\n')
    fp_out.write('--------------------------------------------------------------------\n')
    laplace_unigram = laplace_smoothing_unigram_model(lap_counts,unigram_total,i)
    test_unigram_arr=[]
    laplace_ouput(test_sentences,laplace_unigram,i)

'''-------------------------------Simple Bigram------------------------------------'''
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Bigram test probabilities\n')
fp_out.write('-------------------------\n')

def bigram_output(test_sentences,bigram_counts):
    for elem in test_sentences:
        p_val=0
        for w1,w2 in bigrams(elem.split()):
            try:
                p_val+=math.log10(bigram_counts[w1][w2])
            except Exception as e:
                p_val=0
                break
        N = 1/float(len(elem.split()))
        if p_val!=0:
            perp = (1/(math.pow(math.pow(10,p_val),N)))
            fp_out.write('The sequence '+ elem +' has bigram log likelihood of '+str(p_val)+'\n')
            fp_out.write('The sequence '+ elem +' has perplexity of '+ str(perp)+'\n')
        else:
            fp_out.write('The sequence '+ elem +' has bigram log likelihood of inf\n')
            fp_out.write('The sequence '+ elem +' has perplexity of inf\n')
        #test_bigram_arr.append(p_val)
bigram_output(test_sentences,bigram_counts)

'''---------------------------Laplace Smoothing Bigrams------------------------'''

def laplace_smoothing_bigram_model(sentences, laplace_constant):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]=float(laplace_constant+model[w1][w2])/((laplace_constant*vocabulary_size)+tot_count)
     
    return model

def laplace_ouput_bigram(test_sentences, laplace_bigram, count_bigram, laplace_constant):
    for elem in test_sentences:
        p_val=0
        for w1,w2 in bigrams(elem.split()):
            #print(laplace_bigram[w1][w2])
            try:
                #print(laplace_bigram[w1][w2])
                p_val+=math.log10(laplace_bigram[w1][w2])
                N = 1/float(len(elem.split()))
            except Exception as e:
                temp = (laplace_constant)/(len(count_bigram[w1])+laplace_constant+float(vocabulary_size))
                p_val+=math.log10(temp)
                N = 1/float(len(elem.split()))
      
        fp_out.write('The sequence '+ elem +' has bigram log likelihood of '+ str(p_val)+'\n')
        fp_out.write('The sequence '+ elem +' has perplexity of '+ str(perplexity(p_val,N))+'\n')


fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Laplace Bigram test probabilities\n')
fp_out.write('---------------------------------\n')
for i in laplace_constant:
    fp_out.write('For laplace constant = '+str(i)+'\n')
    fp_out.write('----------------------------\n')
    fp_out.write('--------------------------------------------------------------------\n')
    laplace_bigram_counts= laplace_smoothing_bigram_model(sentences, i)
    laplace_ouput_bigram(test_sentences,laplace_bigram_counts,count_bigrams, i)


'''-------------------------------Simple Trigram------------------------------------'''
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Trigram test probabilities\n')
fp_out.write('--------------------------\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split()):
        try:
            p_val+=math.log10(trigram_counts[(w1,w2)][w3])
        except Exception as e:
            p_val=0
    N = 1/float(len(elem.split()))
    if p_val!=0:
        perp = (1/(math.pow(math.pow(10,p_val),N)))
        fp_out.write('The sequence '+ elem +' has trigram log likelihood of '+ str(p_val)+'\n')
        fp_out.write('The sequence '+ elem +' has perplexity of '+ str(perp)+'\n')
    else:
        fp_out.write('The sequence '+ elem +' has trigram log likelihood of inf\n')
        fp_out.write('The sequence '+ elem +' has perplexity of inf\n')
    
    
    test_trigram_arr.append(p_val)
    

'''-----------------------Laplace Smooth Trigram--------------------------------'''
            
def laplace_smoothing_trigram_model(sentences, laplace_constant):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]=float(laplace_constant+model[(w1,w2)][w3])/((laplace_constant*vocabulary_size)+tot_count)
     
    return model

def laplace_ouput_trigram(test_sentences, laplace_trigram, count_trigrams,laplace_constant):
    for elem in test_sentences:
        p_val=0
        for w1,w2,w3 in trigrams(elem.split()):
            try:
                p_val+=math.log10(laplace_trigram[(w1,w2)][w3])
            except Exception as e:
                if count_trigrams.get((w1,w2),0) !=0:
                    temp = (laplace_constant)/(len(count_trigrams.get((w1,w2),0))+laplace_constant+float(vocabulary_size))
                else:
                    temp = (laplace_constant)/(laplace_constant+float(vocabulary_size))
                p_val+=math.log10(temp)        
        N = 1/float(len(elem.split()))      
        fp_out.write('The sequence '+ elem +' has trigram log likelihood of '+ str(p_val)+'\n')
        fp_out.write('The sequence '+ elem +' has perplexity of '+ str(perplexity(p_val,N))+'\n')
        
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Laplace Trigram test probabilities\n')
fp_out.write('----------------------------------\n')

for i in laplace_constant:
    fp_out.write('For laplace constant = '+str(i)+'\n')
    fp_out.write('---------------------------\n')
    fp_out.write('--------------------------------------------------------------------\n')
    laplace_trigram_counts= laplace_smoothing_trigram_model(sentences,i)
    laplace_ouput_trigram(test_sentences,laplace_trigram_counts, counter,i)

  
'''------------------------------Good Turing Bigram--------------------------------'''

turing_bigram={}
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Good Turing Bigram test probabilities\n')
fp_out.write('-------------------------------------\n')  
for key,value in count_bigrams.items():
    for key1,value1 in value.items():
        turing_bigram[(key,key1)]=value1
        
def good_turing_bigram_model(data): 
    bigram_distribution = FreqDist(data)
    good_turing_bigram = SimpleGoodTuringProbDist(bigram_distribution)
    return good_turing_bigram
    
good_turing_bigram = good_turing_bigram_model(turing_bigram)

def bigram_turing_output(test_sentences, good_turing_bigram):
    for elem in test_sentences:
        p_val=0
        for w1,w2 in bigrams(elem.split()):
            p_val+=math.log10(good_turing_bigram.prob((w1,w2)))
        N = 1/float(len(elem.split()))
            #log_prob = log_likelihood(p_val)
        fp_out.write('The sequence '+ elem+' has perplexity of '+str(perplexity(p_val,N))+'\n')
        fp_out.write('The sequence '+elem+' has bigram log likelihood of '+ str(p_val)+'\n')

bigram_turing_output(test_sentences,good_turing_bigram)


    
'''------------------------------Good Turing Trigram--------------------------------'''

turing_trigram=Counter()
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Good Turing Trigram test probabilities\n')
fp_out.write('--------------------------------------\n') 
  
for key1,key2 in counter.keys():
    for key3 in counter[(key1,key2)].keys():
        turing_trigram[(key1,key2,key3)]=counter[(key1,key2)][key3]
        
def good_turing_trigram_model(data): 
    trigram_distribution = FreqDist(data)
    good_turing_trigram = SimpleGoodTuringProbDist(trigram_distribution)
    return good_turing_trigram
    
good_turing_trigram = good_turing_trigram_model(turing_trigram)

def trigram_turing_output(test_sentences, good_turing_trigram):
    for elem in test_sentences:
        p_val=0
        for w1,w2,w3 in trigrams(elem.split()):
            p_val+=math.log10(good_turing_trigram.prob((w1,w2,w3)))
        N = 1/float(len(elem.split()))
            #log_prob = log_likelihood(p_val)
        fp_out.write('The sequence '+ elem+' has perplexity of '+str(perplexity(p_val,N))+'\n')
        fp_out.write('The sequence '+elem+' has bigram log likelihood of '+ str(p_val)+'\n')

trigram_turing_output(test_sentences,good_turing_trigram)

'''----------------------------Interpolation----------------------------------'''
lambda_value = [0.2, 0.5, 0.8]

def interpolation_model(sentences,unigram_counts, lambda_value):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]= lambda_value*(model[w1][w2]/tot_count)+ (1-lambda_value)*unigram_counts[w2]
     
    return model

def interpolation_output(test_sentences, interpolation_probability,unigram_counts, lambda_value):
        for elem in test_sentences:
            p_val=0
            for w1,w2 in bigrams(elem.split()):
                try:
                    p_val+=math.log10(interpolation_probability[w1][w2])
                except Exception as e:
                    p_val+=(1-lambda_value)*unigram_counts[w2]
                    break
            N = 1/float(len(elem.split()))
            if p_val!=0:
                perp = (1/(math.pow(math.pow(10,p_val),N)))
                fp_out.write('The sequence '+ elem +' has interpolation log likelihood of '+str(p_val)+'\n')
                fp_out.write('The sequence '+ elem +' has perplexity of '+ str(perp)+'\n')
            else:
                fp_out.write('The sequence '+ elem +' has interpolation log likelihood of inf\n')
                fp_out.write('The sequence '+ elem +' has perplexity of inf\n')
            #test_bigram_arr.append(p_val)
fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Interpolation Bigram test probabilities\n')
for i in lambda_value:
    fp_out.write("For lambda = "+str(i)+"\n")
    fp_out.write('-------------------\n')
    fp_out.write('--------------------------------------------------------------------\n')
    interpolation_probability = interpolation_model(sentences, unigram_counts, i)
    interpolation_output(test_sentences,interpolation_probability,unigram_counts, i)

fp_out.write('--------------------------------------------------------------------\n')
fp_out.write('Running Time: '+str(time.clock() - start_time)+ " seconds")

fp_out.close()


