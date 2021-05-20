#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import pandas as pd
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import tweepy # to use Twitter’s API
from textblob import TextBlob# for doing sentimental analysis
import re # regex for cleaning the tweets
import nltk
from nltk.corpus import wordnet
import plotly.express as px
import string
import collections
import regex as re
from nltk import pos_tag
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import PorterStemmer 
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
nltk.download('punkt')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# In[2]:


################### import the dummy data  #############
#df = pd.read_csv("C:\\Users\\bhavya.adlakha\\Bangalore-tweets.csv")
df = pd.read_csv("https://raw.githubusercontent.com/bhav-cloud/Twitter-Text-Analysis/main/Dummy.csv")


# In[3]:


################## CREATING A COLUMN FOR DATE ############
df['date1'] = pd.to_datetime(df['Datetime'],format='%m/%d/%Y %H:%M').dt.date


# In[4]:


##########clean the text without stemming##########
def clean1(text):
    text = re.sub(r'\&\w*;', '', text)
    #Convert @username to AT_USER
    text = re.sub('@[^\s]+','',text)
    # Remove tickers
    text = re.sub(r'\$\w*', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    # Remove hashtags
    #text = re.sub(r'#\w*', '', text)
    ##########Remove RT###########
    text = re.sub('RT','',text)
    ##################################
    text = text.lower()
    #########tokenize text#######
    text = word_tokenize(text)
    text = [word.strip(string.punctuation) for word in text]
    ######remove stopwords#######
    stop = stopwords.words('english')
    #text = [x for x in text if x not in stop]
    #######remove empty tokens#######
    text = [t for t in text if len(t)>1]
    ###########Stemming########3
    #text = ' '.join(PorterStemmer().stem(word)for word in text)
    ####After doing port atemming you should join the words back again########
    #text = [[stem(word) for word in sentence.split(" ")] for sentence in documents]
    text = ' '.join(c for c in text if c <= '\uFFFF') 

    
    return text


# In[7]:


###########definE a function for stemming#####
def stemming(tweet):
    tweet = word_tokenize(tweet)
    tweet = ' '.join(PorterStemmer().stem(word)for word in tweet)
    return tweet


# In[5]:


df2 = df


# In[6]:


################ Add column for cleaned text without stemming ###########
df2['text_clean1'] = df2['Text'].apply(clean1)


# In[8]:
##################Function to tokenize the text###########
def tokenize(text,list):
    text = word_tokenize(text)
    list.extend(text)
    
################### CREATE A LIST OF ALL THE POSITIVE WORDS ##############
positive = ["care","education","love","luck","focus","power","soft","hopes","suspend"]

#########################Function to store positive words##############
def positive_words(text,list_P):
    "Function for positive words"
    for i in range(len(text)):
        for j in range(len(positive)):
            if(text[i] == positive[j]):
                list_P.append(text[i])
    

####### Use text blob ########

################# ADD POLARITY AND SUBJECTIVITY TO GET THE SENTIMENT ##############
polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity
df2["polarity"] = df2["text_clean1"].apply(polarity)
df2["subjectivity"] = df2["text_clean1"].apply(subjectivity)


# In[9]:


############## Create a function to get the sentiment using polarity ###### 
def ratio(x):
    if x > 0:
        return "Positive"
    elif x < 0:
        return "Negative"
    else:
        return "Neutral"
    
    
df2['Sentiment'] = df2['polarity'].apply(ratio)    


# In[10]:

################## Datasets that will be used for visualization #################
df3 = df2.groupby(["Geolocation","date1"])["polarity"].mean().reset_index()
df4 = df3.pivot(index='date1', columns='Geolocation', values='polarity')


# In[11]:


def mentions(text):
    text = re.findall(r"@\w+", text)
    return text


# In[12]:


df["mentions"] = [re.findall(r"@\w+", x) for x in df["Text"]]
df["hashtags"] = [re.findall(r"#\w+", x) for x in df["Text"]]


# In[13]:


############# Filter the dataframe for mentions #########
df['len'] = df.apply(lambda row: len(row.hashtags), axis=1)
df5 = df[df["len"] != 0].reset_index()
Banglore_hashtags = []
Mumbai_hashtags = []
All_hashtags = []


# In[14]:


for i in range(len(df5["Geolocation"])):
    if(df5["Geolocation"][i] == "Bangalore"):
        Banglore_hashtags.extend(df5.hashtags[i])
    elif(df5["Geolocation"][i] == "Mumbai"):
        Mumbai_hashtags.extend(df5.hashtags[i])
    else:
        All_hashtags.extend(df5.hastags)
        


# In[ ]:


##################### Show data on dashboard #######################
st.title("City based Happiness Analysis")
st.markdown("The dashboard will do a location based analysis of the tweets")
st.sidebar.checkbox("Show Analysis by city", True, key=1)
#ALL  = ("Bangalore","Mumbai")
City_select = st.sidebar.multiselect('Select a City',df['Geolocation'].unique())

###############Add a Slider to select the number of days###########
st.sidebar.markdown("Drag the sliders to choose the number of days")

days = st.sidebar.slider("Display days:",0,7,3)


#get the city selected in the selectbox
city_data = df['Geolocation'].isin(City_select)
city_data1 = df[city_data]
st.dataframe(city_data1)



# In[56]:


df2x = df2[city_data] #######will be subsetted for dates#####
df3x = df3[city_data]


# In[ ]:

######################### subset the dataset according to days ################
if (days == 7):
    df2x = df2x
    df3 = df3
elif(days == 0):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=5)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=5)]
elif(days == 1):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=5)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=5)]
elif(days == 2):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=4)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=4)]
elif(days == 3):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=3)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=3)]
elif(days == 4):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=2)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=2)]
elif(days == 5):
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=1)]
    df3 = df3[df3["date1"] > df3["date1"].min()+timedelta(days=1)]    
else:
    df2x = df2x[df2x["date1"] > df2x["date1"].min()+timedelta(days=0)]
    df3 = df3[df3["date1"] > df3x["date1"].min()+timedelta(days=0)]
    
        
    


# In[ ]:


Sent_by_location = px.histogram(df2x, x = "Geolocation",color = "Sentiment", histfunc = "count", title = "Sentiment by Location",color_discrete_sequence=px.colors.qualitative.Dark24)
Happiness_score = px.line(df3, 
        x="date1", 
        y="polarity", 
        color="Geolocation", 
        title="Happiness Index")
Sent_by_time = px.histogram(df2x, x = "date1",color = "Sentiment", histfunc = "count", title = "Sentiment by time",animation_frame="Geolocation")


# In[ ]:


st.plotly_chart(Sent_by_location)
st.plotly_chart(Happiness_score)
st.plotly_chart(Sent_by_time)


# In[60]:


from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 
import string
stopwords1 = set(STOPWORDS)
stopwords1.update(["br", "href","good","great","rt","pt"]) 
##
i = 1
for w in City_select: 
    globals()[w+"_df"] =df2x[df2x["Geolocation"]==w]
    pos = " ".join(review for review in globals()[w+"_df"].text_clean1)
    wordcloud2 = WordCloud(stopwords=stopwords1).generate(pos)
    plt.subplot(1,2,i)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    i = i + 1
    plt.show()
    st.markdown("What's on " + w+"'s mind!")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)


# In[61]:
###########################To get postive words from each city####################

for w in City_select: 
    globals()[w+"_df"] =df2[df2["Geolocation"]==w]
    ############ Empty list for words of each city ############
    globals()[w+"_words"] =[]
    ########### Empty list for positive words ########
    globals()[w+"_Positive_words"] =[]
    globals()[w+"_Pcounts"] =[]
    globals()[w+"_df"]["text_clean1"].apply(tokenize,list = globals()[w+"_words"])
    positive_words(globals()[w+"_words"],list_P =globals()[w+"_Positive_words"])
    globals()[w+"_Pcounts"] = pd.DataFrame.from_dict(collections.Counter(globals()[w+"_Positive_words"]),orient = 'index').reset_index()
    globals()[w+"_Pcounts"].columns = ["Words","Frequency"]
    fig = px.bar(globals()[w+"_Pcounts"], x='Words', y='Frequency')
    #fig = pandas.plotting.table(
    #fig.show()
    st.markdown("Positive words in " +w)
    st.plotly_chart(fig)
    




###################### CREATE A FUNCTION TO GET THE MOST COMMON WORDS IN A LIST############
stop_words = set(stopwords.words('english'))
all_words = []
def common(tweet):
    tweet = word_tokenize(tweet)
    tweet = [t for t in tweet if len(t)>3] 
    all_words.extend(x for x in tweet if x not in stop_words)
    


# In[62]:


#######apply on the dataframe#######
df2x["text_clean1"].apply(common)
########### EACH WORD WITH IT'S COUNT ##########
counts = collections.Counter(all_words)
words = pd.DataFrame(counts.most_common(12))
words.columns = ["Words","Frequency"]


# In[66]:


x = words["Words"]
y = words["Frequency"]
import seaborn as sns
plt.figure(figsize=(14, 6))
plt.bar(x,y,color = "lavender")
plt.xlabel("Words", fontweight='bold')
ax = plt.axes()
  
# Setting the background color of the plot 
# using set_facecolor() method

ax.set_facecolor("black")
plt.show()
st.markdown("Most common words by count")
st.pyplot()
                 
                 
                 

