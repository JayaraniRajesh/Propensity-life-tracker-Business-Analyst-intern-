# Propensity-life-tracker-Business-Analyst-intern
Worked as Business Analyst intern at a non-profit start up company. Conducted data cleaning and data source research, tested and developed ML models using Python, created interactive dashboard using Power BI and provided weekly reports to update the team.

# Introduction
Propensity-Life Tracker is a project idea by Teresa Woods Snelgrove and she is the project lead of this innovative project. Teresa is a successful serial entrepreneur who has had success in building numerous businesses.
Teresa is accompanied by John Pickard, who is an exceptional entrepreneur, whose expertise and inputs would bring additional value and insights to the project. Teresa’s vision of creating this project would in turn changes the lives of many individuals who are suffering from mental illness and stress. This project is a tool that would have a wide reach of audiences as it is can be used for monitoring the well-being of the kids from school, monitoring old people in homes, in hospitals for recovering patients, and so on.

# Project Description
Propensity is a tool that helps a user capture a succession of thoughts or memories through dialogues/conversations with an intelligent agent (Like Alexa or Siri). Our agent is called “Ariadne”. Through this tool, people can have conversations with Ariadne, which in turn analyses those conversations and converts them into meaningful reports which highlight the sentiment, subject, mood analysis of a person. The tool could have alternative uses:  A mood monitor, mental health monitor, sentiment analysis, or bias analysis, etc. It could be offered as a wearable device, a smartphone app, etc.
Technologies used:
Python (Jupyter notebook and Google colab)
Machine Learning
PowerBI

# Objective
Further development of the project, from the end of Phase One. Phase One of the development of the project largely focused on the initial research and the design of the Machine Learning (ML) models. The team has developed two ML models: one for the classification of the Taxonomy and the other model is for the Sentiment Classification.
70%. The Taxonomy classification model is based on the Naïve Bayes algorithm and this model has a prediction accuracy of 25%. The Sentiment classification model is based on the VADER (Valence Aware Dictionary for Sentiment Reasoning) model and it has a prediction accuracy rate of 70%.
Phase Two of the project involves improving the performance of the working ML models for the Taxonomy and Sentiment classification, developing new models for Subject and Tense classification, creating datasets for training and testing the models, connecting the models with the database to store data, creating visualizations for the output data.

# Methodology
Phase Two of the project was split into three parts – Research, Data collection and labeling, and model and dashboard development. The Research part largely focused on machine
learning which would be a good fit for the problem/requirement which is at hand. It involved topics and data from various white papers, blogs, etc. 
The data collection and labeling was the process of collecting the conversational data from Open sources. Once the data was collected, the data needs to be custom labeled to train the machine learning model. 
This process is essential because the machine learning model predicts the needed results based on the data that is used for training.

# Research Findings
The following research questions are the key challenges in Phase Two and researching and analyzing those paves the way for the project.
• How to improve the accuracy of the Taxonomy model. Should we change the current ML model or find techniques to improve the accuracy.
The Taxonomy model was designed and built using the Naïve Bayes classifier algorithm in Phase One of the project. This model gave an accuracy of 25%. So there was a need to improve the accuracy of the model to make a realistic prediction of data. To increase the prediction accuracy of the model, there was a need to change the model. The Naïve Bayes classifier was replaced with the Logistic Regression model and this model gave a prediction accuracy of approximately 70%.
• What models would be the best fit for Subject and Tense classification?
The Subject and Tense classification model were the new ML models which were built from scratch in Phase Two. After extensive research, for the Subject model, the spacy library was used and for the Tense classification model, the Logistic Regression has been used. The former model is a pre-trained model which needed no training or labeled datasets to make predictions. Whereas the latter is a model which is a custom-trained one based on the training from the labeled datasets.
• How can we create/find data that matches or closely corresponds to real-life conversations?
The project is mostly based on the conversational data of everyday life as the user would be talking/texting our agent Ariadne. So finding the necessary data that accurately reflected the real-world data posed a challenge. After browsing and searching hundreds of sites and white papers, the MELD dataset which was used in a white paper provided to be a good dataset to train the models. This dataset is 
conversational dataset between Joey and Chandler from the American Sitcom “F.R.I.E.N.D.S”.
• What database can be used to store the processed data? And how to design it?
With numerous databases available in the market, it was tough to find a good database to store the application data. After some extensive research, the MySql database was chosen to be an apt candidate for storing the data. MySql is an open-sourced relational database management system. Also, for the design part, the database was designed in such a way that it stores the raw data, that was required by the models for training purposes, and also the test data along with the prediction values.
• There are many visualization tools in the market. Which tool would be apt for our analysis visualization?
Once the data has been ingested and predictions have been made, the results must be shown using visuals to make them more understandable and appealing to the end-user. Also, the application requires a dashboard that summarizes the data and shows the necessary and needed insights to the user. For designing the dashboard, Microsoft Power BI was used. Power BI is one of the leading visualization tools and has many features and cloud infrastructure. This software has all the necessary components to design and show the necessary insights in a dashboard.

# Outcomes and Benefits
With the completion of Phase Two, the core part of the project is almost over. The outcomes and benefits of completion of Phase Two are the following:
• Two new models were added in Phase Two of the project for Subject and Tense classification.
• Database has been implemented so that the team can view and analyze the past data.
• As the data is stored, the visuals developed can be across a wide time frame.
• The core part of the project is completed, so that the next phase of the project which might include building UI and developing app/wearables tech can proceed.
• A SharePoint site is created containing all the necessary research documents, analysis documents, workflow documents, source code, sample datasets, etc.
• Client has an idea of what has been completed so far and undertakes the next necessary steps to carry forward the project.

# Dataset Collection and Labelling
The project is based on analyzing the conversations between Ariadne and a user. So the data required for training the models must closely resemble the real-world conversations. Finding such data was a huge problem as most of the NLP datasets available online were not conversational.
After a lot of research and searching online, the MELD dataset was extracted from a white paper, which had conversations from the American Sitcom F.R.I.E.N.D.S.
Once the dataset was extracted it had to be labeled manually to train the models. The manual labeling of the data was time-consuming. After manually labeling the data, the data was stored in a MySql database using a python program that reads the CSV/excel files and then inserts those records into the input_table of the database.
Sentiment Model: As this is a pre-trained model, it doesn’t require labeling.
Taxonomy/Mood Model: The labels for this model were anger, disgust, fear, joy, sadness, surprise, and neutral. Based on these labels the sentences were labeled for training the model.
Tense Model: The Tense model is custom-trained. So, the data was labeled for this model. The labels were past, present, and future.
Subject Model: This model is based on a pre-trained model; hence labeling was not necessary.

# Database Design and Architecture
In Phase One of the project, all the input and output data were handled as files. This approach could work better in the short term but will have performance issues in the long run.
This was the reason why storing the data in the database is a better option. There are several advantages to this, such as,
• The implementation of this database would help to create a centralized repository for the raw data to be used for the models.
• Avoids the need to maintain multiple files for different models.
• Data from the database can be exported into multiple file formats if needed.
• Would be easy to search a particular data using SQL queries.
• The predicted output data can be maintained, and we can use them to compare performances if we are changing the ML model for prediction.
• Quicker processing of a large amount of data.
To store and retrieve data, the MySql database was used. MySql is an open-sourced database that has good support for python and Jupyter notebooks. The MySql Server and MySql client has been used to create databases, tables, and querying data.
The database has two tables. They are:
• Input_data: This table holds the raw labeled data to be used by the models.
• Output_data: This table holds the predicted output data from the different models.

# Classification Models
The conversation data from users were analyzed for four different subjects, namely
• Sentiment
• Taxonomy/Mood
• Tense
• Subject

# Sentiment Classification Model:
Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. It is the process of detecting positive or negative sentiment in text. It’s often used by businesses to detect sentiment in social data, gauge brand reputation, and understand customers.
Since humans express their thoughts and feelings more openly than ever before, sentiment analysis is fast becoming an essential tool to monitor and understand the sentiment in all types of data. Our application heavily relies on capturing the sentiment of a conversation as it is one of the key elements in judging a customer’s behavior.
The Vader model is the backbone of the Sentiment classification model. VADER (Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It is available in the NLTK package and can be applied directly to unlabeled text data.
VADER sentimental analysis relies on a dictionary that maps lexical features to emotion intensities known as sentiment scores. The sentiment score of a text can be obtained by summing up the intensity of each word in the text.
Process
• The data preprocessing stage involves removing stopwords and unwanted characters using regex from the sentences.
• After the sentences are cleaned, the words in the sentences are split and the lemmatization technique is applied to the words.
• Once it has been done, all the words are put up back together to form sentences.
• Since the VADER model is pre-trained, it does not require the labeling of data for making predictions.
• After the data preprocessing and cleaning, the data was fed to the machine learning model to make predictions.
• The output of the machine learning process is the predicted sentiment of the sentences.
• This predicted output can be saved as an excel file or stored in a database.

# Taxonomy Classification model:
The taxonomy classification model is more of a mood classification model. Here the emotion/mood from the sentences of a conversation will be captured. The sentences were labeled with different moods such as anger, disgust, fear, joy, sadness, surprise, and neutral.
For the Taxonomy model, the model built in Phase One was showing an accuracy of 20%. After a lot of research, different techniques were implemented to improve the accuracy of the model. But even after fine-tuning the model, it was not showing a considerable increase in accuracy. Hence, the taxonomy model was rebuilt from the scratch using a different ML algorithm.
Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. It is another powerful supervised ML algorithm used for binary classification problems (when the target is categorical). It essentially uses a logistic
function to model a binary output variable. Hence, Logistic Regression seemed to be an ideal candidate for the Taxonomy model.
Process:
• Logistic Regression is a supervised learning algorithm, so the dataset needs to be labeled.
• The sentences are manually labeled with the seven different labels marking the emotion/mood of each sentence.
• After labeling, the data is then cleaned for stopwords and unwanted characters are removed using the regex.
• Then the sentences are split into words and lemmatization is applied to those words.
• The lemmatized words are joined back to a sentence and stored in a list.
• The LabelEncoder is then used on the target variable and vectorization is carried out on the dataset.
• Then the dataset is split into a training dataset and a test dataset.
• The model is trained with the training dataset and after that, the model is fed with the testing dataset to make predictions.
• The model gives the prediction output for the test dataset.
• The prediction results and actual results are used to find the accuracy of the model.

# Tense Classification model:
Analyzing the conversations for Sentiment or Mood is not enough to understand the mental well-being of an individual. The Sentiment and the Mood of the individual may vary with past experiences, present circumstances, or future decisions. So, it is really
important to extract the tense of the conversations and use them along with Sentiment and Mood to enhance our findings on the mental wellbeing of an individual.
The Tense model is a custom-built supervised learning model. The logistic regression algorithm is used for building the model. This is the same algorithm used in the Taxonomy/Mood model.
Process:
• The process for building the Tense classification model is the same as the Taxonomy model.
• Logistic Regression is a supervised learning algorithm, so the dataset needs to be labeled.
• The sentences are manually labeled with the three different labels marking the tense of each sentence.
• After labeling, the data is then cleaned for stopwords and unwanted characters are removed using the regex.
• Then the sentences are split into words and lemmatization is applied to those words.
• The lemmatized words are joined back to a sentence and stored in a list.
• The LabelEncoder is then used on the target variable and vectorization is carried out on the dataset.
• Then the dataset is split into a training dataset and a test dataset.
• The model is trained with the training dataset and after that, the model is fed with the testing dataset to make predictions.
• The model gives the prediction output for the test dataset.
• The prediction results and actual results are used to find the accuracy of the model.

# Subject Classification Model:
Whenever there is a conversation between two people it will be based on something, that’s what is called the subject. Similarly, when users talk to Ariadne they talk about their interests, their experiences, their families and friends, etc. Users mostly have meaningful conversations with Ariadne. Finding out the subject of the conversations may help us analyze what is the mood or sentiment attached to that particular subject and how it can affect the mental health of the individual in the future.
For the Subject classification model, the pre-trained model from the Spacey library has been used. This ML algorithm generates different tags based on the sentences it reads. From these tags, the subject-related tag is filtered out to identify the subject of the sentence.
Process:
• For the Subject classification model, the pre-trained model from the Spacey library has been used.
• This ML algorithm generates different tags based on the sentences it reads.
• From these tags, the subject-related tag is filtered out to identify the subject of the sentence.
• Since the Spacey model is pre-trained, it does not require the labeling of data for making predictions.
• The data was fed to the machine learning model to make predictions.
• The output of the machine learning process is the subject of the sentences.
• This output can be saved as an excel file or stored in a database.

# Conclusion
With all the four models working and the prototype being built, the base required for the Ariadne project has been set up. Further developments might include integrating 
the dashboard with the code base, moving the codebase to the cloud, saving the data required in the database, and building a mobile app.
