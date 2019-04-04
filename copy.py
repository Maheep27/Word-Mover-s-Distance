#import all necesarry library

import gensim                                    #for Word Mover's distance algorithm and also for word2vec  
from nltk.tokenize import word_tokenize          #for generating token from sentence 
from nltk.corpus import stopwords                #for removing stopwords from each sentence

#load word2vec model, here GoogleNews is used which is pre-trained model used for generating vectors coressponding to each words in sentence 
model = gensim.models.KeyedVectors.load_word2vec_format('/home/maheep/Music/GoogleNews-vectors-negative300.bin', binary=True)

#through removestopwords function remove stopwords using nltk library
def removestopwords(doc):                                                         #definition of a function
    stop_words = set(stopwords.words('english'))                                  #load english stopwords from nltk
    word_tokens = word_tokenize(doc)                                              #generate token for coressponding sentence
    filtered_sentence = [w for w in word_tokens if not w in stop_words]           #remove stopwords from sentence if present
    return filtered_sentence                                                      

#two sample sentences 
s1 = 'I am going to India'
s2 = 'I am going to bharat'

#remove stopwords by calling above removestopwords function 
s_1=removestopwords(s1)                                     #After removing stopwords it give S_1 = ['I','going','India']
s_2=removestopwords(s2)                                     #After removing stopwords it give S_2 = ['I','going','bharat']

#calculate distance between two sentences using WMD algorithm

#This algorithm allows transfer every word from sentence-1 to sentence-2 
#because algorithm does not know which word of senetcne-1 should be transfer 
#to which word of senetence-2. At the end it will choose the minimum transportation 
#cost to transport every word from sentence-1 to sentence-2

distance_1=model.wmdistance(s_1,s_2)   
print('distance_1',distance_1)                             # distance_1= 1.1158282554856542 (distance less means similarity high)

#two sample sentences 
s3 = 'I will be eating coffee'                             
s4 = 'I will be drinking coffee'                            

#remove stopwords by calling above removestopwords function 
s_3=removestopwords(s3)                                    #After removing stopwords it give S_1 = ['I','eating','coffee']
s_4=removestopwords(s4)                                    #After removing stopwords it give S_1 = ['I','drinking','coffee']

#calculate distance between two sentences using WMD algorithm
distance_2=model.wmdistance(s_3,s_4)                       
print('distance_2',distance_2)                             # distance_2= 1.0779358435019317 (distance less means similarity high)
