# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:46:36 2020

@author: DELL
"""

#IMPORTS
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings
import os
from gtts import gTTS
import playsound

#Ignoring warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')  #Sentence tekenizer
nltk.download('wordnet') #Lexical database for the English language

#Downloading the article on covid-19
article=Article('https://www.medicalnewstoday.com/articles/256521')
article.download()
#Parsing the data
article.parse()
article.nlp()

corpus=article.text
print(corpus)

#Sentence tokenization: splitting a string, text into a list of tokens
text=corpus
sent_tokens=nltk.sent_tokenize(text)
print(sent_tokens)

#Removing punctuation
remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)
print(string.punctuation)
print(remove_punct_dict)

#Word tokenization: extract the tokens from string of characters by using tokenize
def LemNormalize(text):
  return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

print(LemNormalize(text))

#Greeting inputs and outputs
GREETING_INPUTS=['hey','hi','hello','heya','howdy','whatsup','wassup','hola']
GREETING_OUTPUTS=['hey!','hello!','heya!','howdy!','hey there!','holaa!']

#Function to return a random greeting from GREETING_OUTPUTS if a greeting form 
#GREETING_INOUTS is in the text
def greeting(sentence):
  for word in sentence.split():
    if word.lower() in GREETING_INPUTS:
      return random.choice(GREETING_OUTPUTS)

#Processing user query to generate the appropriate bot response
def response(user_response):
  #Change the user_response to lower case  
  user_response.lower()

  #Defining robo_response as empty string
  robo_response=''
  #Append user_response to sentence tokens
  sent_tokens.append(user_response)

  #Transform text to feature vectors
  tfidfvec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
  tfidf=tfidfvec.fit_transform(sent_tokens)

  #Get similarity scores (user's response with all other tokens)
  vals=cosine_similarity(tfidf[-1],tfidf)

  idx=vals.argsort()[0][-2]
  #we give -2 as argument since -1 will give the sentence with max similarity, 
  #and that would be the sentence itself(since it is appended at the end)
  #hence we use -2 which give the second most similar sentence

  #reduce the dimensionality of vals
  flat=vals.flatten()

  #sort the list in ascending order
  flat.sort()

  #Get the most similar score to user response
  score=flat[-2]

  #if score==0 => no text similar to user response
  if(score==0):
    robo_response=robo_response+"I am sorry, I don't quite understand"

  else:
    robo_response=robo_response+sent_tokens[idx]

  #Remove user_response from sentence tokenizer
  sent_tokens.remove(user_response)

  return robo_response

#Function to play the chatbot response audio
def audioResponse(text):
    print(text)
    #convert text to speech
    myobj=gTTS(text=text,lang='en',slow=False)
    #save audio
    myobj.save('audio_response.mp3')
    #play audio
    playsound.playsound('audio_response.mp3',True)
    os.remove('audio_response.mp3')

#Print the chatbot response
flag=True
print('Bot: ')
audioResponse( '''Hey there! 
Hope you're safe and doing well! 
I am CoronaBot, here to clear all your doubts about corona virus. 
So ask away! If you wish to exit, type 'bye'.
       '''
        )
while(flag==True):
  user_response=input()
  user_response=user_response.lower()

  if(user_response!='bye'):
    if(user_response=='thanks' or user_response=='thank you'):
      print("Bot: ")
      audioResponse("You're welcome!")

    else:
      if(greeting(user_response)!=None):
        print("Bot: ")
        audioResponse(greeting(user_response ))
      else:
        print("Bot: ")
        audioResponse(response(user_response))
        
  else:
    flag=False
    print("Bot: ")
    audioResponse("Talk to ya later!")

