# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:58:23 2020

@author: vw178e
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

########### Extracting reviews from snapdeal website for the product perfect nova trimmer ##############

nova_trimmer=[]
#url1 = "https://www.snapdeal.com/product/apple-iphone-5c-16-gb/988871559/reviews?page="
url1 = "https://www.snapdeal.com/product/n-r-faishon-hub-nova/629070106011/reviews?page="
url2 = "&sortBy=RECENCY&vsrc=rcnt#defRevPDP"
### Extracting reviews from Snapdeal website ################
for i in range(1,16):
  ip=[]  
  base_url = url1+str(i)+url2
  response = requests.get(base_url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  temp = soup.findAll("div",attrs={"class","user-review"})# Extracting the content under specific tags  
  for j in range(len(temp)):
    ip.append(temp[j].find("p").text)
  nova_trimmer=nova_trimmer+ip  # adding the reviews of one page to empty list which in future contains all the reviews



### Removing repeated reviews 
nova_trimmer = list(set(nova_trimmer))

# Writing reviews into text file 
with open("ip_snapdeal.txt","w",encoding="utf-8") as snp:
    snp.write(str(nova_trimmer))
    
    
    # Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(nova_trimmer)



# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)



# words that contained in iphone 7 reviews
ip_reviews_words = ip_rev_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:/Training/Analytics/Text_Mining/stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

ip_reviews_words = [w for w in ip_reviews_words if not w in stopwords]


# =============================================================================
# Creating DTM
# =============================================================================
# Preparing email texts into word count matrix format 

def split_into_words(i):
    return [word for word in i.split(" ")]

product_reviews_bow = CountVectorizer(analyzer=split_into_words).fit(nova_trimmer)

# For all reviews
all_reviews_matrix = product_reviews_bow.transform(nova_trimmer)
all_reviews_matrix.shape # (131,315)


# =============================================================================
# Sentiment Analysis
# =============================================================================

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("C:/Training/Analytics/Text_Mining/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("C:/Training/Analytics/Text_Mining/negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
