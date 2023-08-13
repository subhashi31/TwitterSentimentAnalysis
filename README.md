# Twitter Sentiment Analysis: Political Events and Leaders

# Abstract
Sentiment Analysis is used to access the underlying sentiments of a text. People cite their honest opinions online. This makes sentiment analysis of social media data quite insightful. In this paper, we look into the machine learning-based approach to sentiment analysis. The tweets were scraped using Snscrape and were pre-processed using Python string manipulation techniques. We made our own training dataset and labelled it using Flair pre-trained model. To get a more accurate label, only tweets with a score > 0.9 were considered to be a part of the dataset. Keras and TensorFlow were used to build the machine learning model used for the sentiment analysis of the tweets. The model had an accuracy score of ‘0.98’ and a validation accuracy score of ‘0.75’. 

# Keywords
Sentiment Analysis; Machine Learning; Keras; TensorFlow; Twitter; Tweets

# Introduction
Natural Language Processing (NLP) is an interdisciplinary subfield of linguistics, computer science and machine learning. It deals with making machines understand human language. The goal of NLP is to create techniques and algorithms that enable computers to interpret, generate and manipulate human languages. Various NLP tasks include,

1. Text & speech processing  
2. Text classification 
3. Language generation 
4. Language interaction
  
Sentiment Analysis, also referred to as opinion mining, is a Natural Language Processing technique. It is the most common text classification method that takes the incoming message and analysis the underlying sentiment. It classifies the sentence as positive, negative or neutral. Sentiment analysis finds its application in various fields such as customer service, marketing, politics, healthcare, etc. Social Media Sentiment Analysis is a popular field of research and is often used to analyse the public’s perception of government policies and various political and non-political events. 

There are three primary techniques used in sentiment analysis: rule-based, lexicon-based, and machine learning-based approaches. Rule-based approaches use a set of predefined rules to identify sentiments in text. Lexicon-based approaches use a pre-built dictionary of words with associated sentiment scores to determine the overall sentiment of a piece of text. Machine learning-based approaches use algorithms to learn the sentiment of a text by training on a large dataset of labelled text.

This project provides insight into the machine learning-based approach to sentiment analysis.

# Related Work
Zulfadzli Drus and Haliyana Khalid (2019) analysed various studies conducted on sentiment analysis between 2014 and 2019. The result of the analysis showed that most of the articles applied the opinion-lexicon method to analyse text sentiment in social media, extracted data on microblogging sites, mainly Twitter and sentiment analysis applications can be seen in world events, healthcare, politics and business. Their study further concluded that if the data structure is messy and a small amount of data and limited time is available for analysis, it is recommended to go for the lexicon-based method while for a large amount of data machine learning-based method is suitable as it requires more time and data to train. To improve the quality and accuracy of the result, it is suggested to combine both the lexicon and machine learning methods [1].

Alexandra Balahur (2013), in his paper on Sentiment Analysis in Social Media Text, emphasised tweet pre-processing. Pre-processing was done to normalize the language and generalize the vocabulary employed to express sentiment. The use of such generalized features significantly improves the results of sentiment classification [2].

Akshat Bakliwal et al. (2013), performed sentiment analysis on political tweets. Sarcastic tweets were omitted from the dataset. They used supervised learning and a feature set consisting of subjectivity-lexicon-based scores and achieved an accuracy of 61.6%. Their supervised learning approach performed better than unsupervised approaches which use subjectivity lexicons to compute an overall sentiment score [3].

In his article, Prajwal Shreyas (2019), discusses the inaccuracies in prediction by the NLTK sentiment classifier and other resources available in Python. His deep learning-based model is a multiclass classification model with 10 classes, one for each decile. He build a Long Short-Term Memory (LSTM) model in Keras and achieved an accuracy of 48.6%. The accuracy of this model will be much higher on a binary class dataset [4].

Sergio Virahonda (2020), in his tutorial about sentiment analysis with deep learning and Keras, implemented three RNN types: a single LSTM (long short-term memory) model, a bidirectional LSTM and a Conv1D model. The bidirectional LSTM seemed to have a better performance than the other two. The article is also helpful for deciding on which loss function and optimizer to use for the model based on the type of labelled dataset [5].

Amy (2022), in her article about sentiment analysis without modelling, compared the accuracy of TextBlob, VADER and Flair. All three models were tested on the same dataset. TextBlob had an accuracy of 68.8%, VADER had an accuracy of 76.8% and Flair had an accuracy of 94.8%. Flair clearly outperformed the other two models [6].

Keras is an API that provides a Python interface for neural networks. It acts like an interface for the TensorFlow library and can be used to build most types of deep learning models. Keras is used by companies such as Netflix, Yelp, Uber, etc. The core data structures of Keras are layers and models. It was developed with a focus on enabling fast experimentation and providing a delightful developer experience. It was developed by François Chollet and was initially released in 2015 [7].

# Methodology 
Sentiment Analysis broadly has five stages:

1. Data extraction
2. Data pre-processing
3. Data labelling
4. Tokenization
5. Training & Testing

# Model Analysis
The analysis of the model is done based on its accuracy and loss function. The model has a ‘0.98’ accuracy score and a ‘0.75’ validation accuracy score. The model tends to learn fast and by the 7th epoch has already reached 95% accuracy, the accuracy further reaches over 98%. The validation accuracy, on the other hand, tends to be between 71% to 77%. The model tends to overfit after the 17th epoch.  

![image](https://github.com/subhashi31/TwitterSentimentAnalysis/assets/106196897/434fd6bd-683d-47db-938c-f8703b51997a)

Model Accuracy

We have used Binary cross-entropy loss, also called log loss, to train the model. It compares the predicted probability to the actual value and the score is calculated by penalizing based on the distance of probabilities from the actual value. The model has a loss score of ‘0.4’ and a validation loss score of ‘1.5’. 

![image](https://github.com/subhashi31/TwitterSentimentAnalysis/assets/106196897/59797307-3f89-4172-a6df-1c10cf82da32)

Log Loss
