"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import nltk
import string
import re
from PIL import Image

#import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud, ImageColorGenerator
import warnings
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings(action = 'ignore') 

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data dependencies
import pandas as pd
import numpy as np
import base64
# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
retweet = 'RT'
mask = np.array(Image.open('resources/10wmt-superJumbo-v4.jpg'))
import streamlit.components.v1 as components



def sentiment_visual(df1_train):
	sid = SentimentIntensityAnalyzer()
	df1_train['scores'] = df1_train['message'].apply(lambda review: sid.polarity_scores(review))
	df1_train['compound']  = df1_train['scores'].apply(lambda score_dict: score_dict['compound'])
	df1_train['comp_score'] = df1_train['compound'].apply(sent_decider)
	fig, axis = plt.subplots(ncols=2, figsize=(15, 5))
	ax = sns.countplot(x='comp_score', data=df1_train, palette='winter', hue='sentiment', ax=axis[0])
	axis[0].set_title('Number of Tweets Per Vader Sentiment',fontsize=14)
	axis[0].set_xlabel('')
	axis[0].set_ylabel('Tweets')
	put_numbers_on_bars(ax)
	ax = sns.countplot(x='sentiment', data=df1_train, palette='winter', hue='comp_score', ax=axis[1])
	axis[1].set_title('Number of Tweets Per Sentiment Class',fontsize=14)
	axis[1].set_xlabel('')
	axis[1].set_ylabel('')
	put_numbers_on_bars(ax)
	plt.show()
	fig, axis = plt.subplots(ncols=2, figsize=(15, 5))
	counts = (df1_train.groupby(['comp_score'])['sentiment'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())
	ax = sns.barplot(x="comp_score", y="percentage_tweets", palette='winter', hue="sentiment", data=counts, ax=axis[0])
	put_numbers_on_bars(ax)
	ax.set_xlabel('Vader Sentiment')
	ax.set_ylabel('Tweets (%)')
	count = (df1_train.groupby(['sentiment'])['comp_score'].value_counts(normalize=True).rename('percentage_tweets').mul(100).reset_index())
	ax = sns.barplot(x="sentiment", y="percentage_tweets", palette='winter', hue="comp_score", data=count, ax=axis[1])
	put_numbers_on_bars(ax)
	plt.xlabel('Sentiment Class')
	plt.ylabel('')
	fig, axis = plt.subplots(ncols=4, figsize=(20, 5))
	group = ["(Anti)", "(Neutral)", "(Pro)", "(News)"]
	for i in range(4):
		df1_train[df1_train['sentiment']==i-1]['comp_score'].value_counts().plot.pie(autopct='%1.1f%%',colormap='winter_r',ax=axis[i])
		axis[i].set_title('Proportion of Tweets Per Vader Sentiment '+ group[i],fontsize=12)
		axis[i].set_ylabel('')
	plt.show()
	st.pyplot()

def put_numbers_on_bars(axis_object):
    """
    Function to plot labels above countplot bars.
    """
    for p in axis_object.patches:
        axis_object.text(p.get_x() + p.get_width()/2., p.get_height(),'%d' % round(p.get_height()), fontsize=11,ha='center', va='bottom')

def sent_decider(compound):
    """
    Function to determine if sentiment is positive, nuetral or negative.
    """
    neutral_point = 0.00
    if compound > neutral_point:
        return 'positive'#1
    elif compound < -neutral_point:
        return 'negative' #-1
    else: 
        return 'neutral'#0


st.cache(suppress_st_warning=True,allow_output_mutation=True)
def class_analysis(df):
    df['sent_labels']  = df['sentiment'].map({-1: 'Anti',0:'Neutral', 1:'Pro', 2:'News'})
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), dpi=100)
    
    sns.countplot(df['sent_labels'], ax=axes[0])
    code_labels=['Pro', 'News', 'Neutral', 'Anti']
    axes[1].pie(df['sent_labels'].value_counts(),labels= code_labels,autopct='%1.0f%%',startangle=90,explode = (0.1, 0.1, 0.1, 0.1))
    fig.suptitle('Sentiment Class Analysis', fontsize=20)
    st.pyplot(fig)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def class_dist(df):
    df['sent_labels']  = df['sentiment'].map({-1: 'Anti',0:'Neutral', 1:'Pro', 2:'News'})
    df['text_length'] = df['message'].apply(lambda x: len(x))
    fig, axis = plt.subplots(ncols=2,nrows=1, dpi=100)
    
    sns.boxplot(x=df['sent_labels'],y=df['text_length'],data=df,ax=axis[0],color = 'orange')

    sns.violinplot(x=df['sent_labels'], y=df['text_length'],ax=axis[1])
    plt.xlabel('Sentiment Class')
    plt.ylabel('Tweet Length')
    plt.tight_layout()
    st.pyplot(fig)


st.cache(suppress_st_warning=True,allow_output_mutation=True)
def mentions(x):
    x = re.sub(r"(?:\@|https?\://)\S+", "", x)
    return x

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def remove_punc(x):
    x = re.sub(r"([^A-Za-z0-9]+)"," ",x)
    return x

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def StopWords():
    stop_words = set(stopwords.words('english'))
    return stop_words

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def word_count(train):
    cnt = Counter()
    for message in train['message'].values:
        for word in message:
            cnt[word] +=1
    return cnt.most_common(20)


st.cache(suppress_st_warning=True,allow_output_mutation=True)
def data_cleaning(df):
    wnl = WordNetLemmatizer()
    df['message'] = df['message'].apply(mentions)
    #df['message'] = df['message'].apply(lambda x: contractions.fix(x))
    df['message'] = df['message'].str.replace(r"http\S+|www.\S+", "", case=False)
    df['message'] = df['message'].map(lambda x: remove_punc(str(x)))
    df['message'] = df['message'].apply(word_tokenize)
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in retweet])
    df['message'] = df['message'].apply(lambda x : [word.lower() for word in x])
    df['message'] = df['message'].apply(lambda x: [word for word in x if word not in StopWords()])
    df['pos_tags'] = df['message'].apply(nltk.tag.pos_tag)
    df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['message'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    return df

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def pro_mostpopular(df):
    pro_popular = df[df['sentiment'] == 1]
    pro_pop = word_count(pro_popular)
    return pro_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def anti_mostpopular(df):
    anti_popular = df[df['sentiment']== -1]
    anti_pop = word_count(anti_popular)
    return anti_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def neutral_mostpopular(df):
    neutral = df[df['sentiment']==0]
    neutral_pop = word_count(neutral)
    return neutral_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def news_mostpopular(df):
    news = df[df['sentiment']==2]
    news_pop = word_count(news)
    return news_pop

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def popularwords_visualizer(data):
    news = news_mostpopular(data)
    pro = pro_mostpopular(data)
    anti=anti_mostpopular(data)
    neutral = neutral_mostpopular(data)
    
    #Creating the Subplots for Most Popular words
    fig, axs = plt.subplots(2, 2)
    
    plt.setp(axs[-1, :], xlabel='Most popular word (Descending)')
    plt.setp(axs[:, 0], ylabel='# of times the word appeard')
    axs[0,0].bar(range(len(news)),[val[1] for val in news],align='center')
    axs[0,0].set_xticks(range(len(news)), [val[0] for val in news])
    axs[0,0].set_title("News Class")
    
    axs[0,1].bar(range(len(neutral)),[val[1] for val in neutral],align='center')
    axs[0,1].set_xticks(range(len(neutral)), [val[0] for val in neutral])
    axs[0,1].set_title("Neutral Class")
    
    axs[1,0].bar(range(len(pro)),[val[1] for val in pro],align='center')
    axs[1,0].set_xticks(range(len(pro)), [val[0] for val in pro])
    axs[1,0].set_title("Pro Class")
    
    axs[1,1].bar(range(len(anti)),[val[1] for val in anti],align='center')
    axs[1,1].set_xticks(range(len(anti)), [val[0] for val in anti])
    axs[1,1].set_title("Anti Class")
    fig.tight_layout()
    st.pyplot(fig)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def wordcloud_visualizer(df):
    news = df['message'][df['sentiment']==2].str.join(' ')
    neutral = df['message'][df['sentiment']==2].str.join(' ')
    pro = df['message'][df['sentiment']==2].str.join(' ')
    anti = df['message'][df['sentiment']==2].str.join(' ')
    #Visualize each sentiment class
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    news_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter').generate(str(news))
    axis[0, 0].imshow(news_wordcloud)
    axis[0, 0].set_title('News Class',fontsize=14)
    axis[0, 0].axis("off") 
    neutral_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(neutral))
    axis[1, 0].imshow(neutral_wordcloud)
    axis[1, 0].set_title('Neutral Class',fontsize=14)
    axis[1, 0].axis("off") 
    
    pro_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(pro))
    axis[0, 1].imshow(pro_wordcloud)
    axis[0, 1].set_title('Pro Class',fontsize=14)
    axis[0, 1].axis("off") 
    anti_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(anti))
    axis[1, 1].imshow(anti_wordcloud)
    axis[1, 1].set_title('Anti Class',fontsize=14)
    axis[1, 1].axis("off")
    st.pyplot(fig)


def tweet_cloud(df):
	mask = np.array(Image.open('resources/10wmt-superJumbo-v4.jpg'))
	words = df['message']
	allwords = []
	for wordlist in words:
		allwords += wordlist
	mostcommon = FreqDist(allwords).most_common(10000)
	wordcloud = WordCloud(width=1000, height=1000, mask = mask, background_color='white').generate(str(mostcommon))
	fig = plt.figure(figsize=(30,10), facecolor='white')
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis('off')
	plt.tight_layout(pad=0)
	st.pyplot(fig)

def prediction_output(predict):
    if predict[0]==-1:
        output="The text has been classified as Anti "
        st.error("Text Sentiment Categorized as: {}".format(output))
    elif predict[0]==0:
        output="The text has been classified as Neutral"
        st.info("Text Sentiment Categorized as: {}".format(output))
    elif predict[0]==1:
        output ="The text has been classified as Pro"
        st.success("Text Sentiment Categorized as: {}".format(output))
    else:
        output = "The text has been classified as News"
        st.warning("Text Sentiment Categorized as: {}".format(output))

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def markup(selection):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h1 style="color:{};text-align:center;">"""+selection+"""</h1></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

st.cache(suppress_st_warning=True,allow_output_mutation=True)
def title_tag(title):
    html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h2 style="color:#00ACEE;text-align:center;">"""+title+"""</h2></div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

#Getting the WordNet Parts of Speech
st.cache(suppress_st_warning=True,allow_output_mutation=True)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
 

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	logo = Image.open('logo1.png')
	st.sidebar.image(logo, use_column_width=True)
	set_bg_hack('bakn.png')

	options = ["Company Profile","About Predict","Text Classification","Exploratory Data Analysis","Model Metrics Evaluation"]
	selection = st.sidebar.selectbox("Menu", options)
	# Building out the "Information" page
	if selection == "Company Profile":
		
		# title_tag("Siders Analytics EST 2022")
		# You can read a markdown file from supporting resources folder
		st.image('resources/p1n.png', use_column_width=True)
		# def st_display_pdf(pdf_file):
		with open("pa2.pdf","rb") as f:
			base64_pdf = base64.b64encode(f.read()).decode('utf-8')
		pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1200" type="application/pdf"></iframe>'
		st.markdown(pdf_display, unsafe_allow_html=True) 
		# st_display_pdf("pa2.pdf")

			
	if selection == "About Predict":
		title_tag("Climate Change Sentiment Analysis brougth to you by siders analytics")
		# You can read a markdown file from supporting resources folder
		st.image('resources/twitter-removebg-preview.png', caption='Tweeet Attack',use_column_width=True)

		# st.markdown("<h2 style='color:#00ACEE'>Introduction</h3><br/>",unsafe_allow_html=True)
		title_tag("Introduction")
		st.image('intro.png',use_column_width=True)
		title_tag("Problem Statement")
		
		st.image('problem statment.png',use_column_width=True)
		st.image('output-onlinegiftools.gif',use_column_width=True)
		# st.markdown("<h3 style='color:#00ACEE'>  zf5 #SIDERS ANALYTICS</h3><br/>",unsafe_allow_html=True)
		title_tag("zf5 #SIDERS ANALYTICS")
		
	# Building out the predication page
	if selection == "Text Classification":
		markup(selection)
		# Creating a text box for user input
		models = ["Decision Tree Classifier","RandomForest Classifier","LGBMClassifier","Stochastic Gradient Classifer","Support Vector Classifier","Linear Support Vector Classifier","Logisitic Regression Classifier"]
		modeloptions = st.selectbox("Choose Predictive Classification Model",models)
		if modeloptions =="Decision Tree Classifier":
			st.info("* Non-parametric supervised learning approaches for classification and regression are Decision Trees (DTs). With a set of if-then-else decision rules, decision trees learn from data to approximate a sine curve. The decision criteria get more complicated as the tree grows deeper, and the model becomes more accurate.\n * In the shape of a tree structure, the decision tree constructs classification or regression models. It partitions data into subgroups following each choice, while also developing an associated decision tree progressively. The end result is a tree containing leaf nodes and decision nodes. There are two or more branches on a decision node. A categorization or decision is represented by the leaf node. Root node is the topmost decision node in a tree that corresponds to the best predictor. Both category and numerical data may be handled using decision trees.")
			st.image('https://i.ibb.co/wYXFgxd/24.png',use_column_width=True)
			tweet_text = st.text_area("Enter Text", "Type Here")
			if st.button("Predict text class with Decision Tree Classifier"):
				pred = joblib.load(open(os.path.join("resources/ridge_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
		if modeloptions =="Stochastic Gradient Classifer":
			st.info("Stochastic Gradient Descent (SGD) Classifier is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.")
			tweet_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Stochastic Gradient Classifer"):
				pred = joblib.load(open(os.path.join("resources/SGD_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
		if modeloptions == "Linear Support Vector Classifier":
			st.info("SVM or Support Vector Machine is a linear model for classification and regression problems.It constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression.")
			tweet_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Linear Support Vector Classifier"):
				pred = joblib.load(open(os.path.join("resources/Lsvc_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
		elif modeloptions =="Support Vector Classifier":
			st.info("a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.")
			svc_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Support Vector Classifier"):
				pred = joblib.load(open(os.path.join("resources/SVCGrid.pkl"),"rb"))
				predict = pred.predict([svc_text])
				prediction_output(predict)
		elif modeloptions =="Logisitic Regression Classifier":
			st.info("the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc.")
			logi_text = st.text_area("Enter Text","Type Here")
			if st.button("Predict text class with Logisitic Regression Classifier"):
				pred = joblib.load(open(os.path.join("resources/logreg_tfidf.pkl"),"rb"))
				predict = pred.predict([logi_text])
				prediction_output(predict)
	
	if selection == "Exploratory Data Analysis":
		markup(selection)
		print('....Cleaning the Raw data')
		train = data_cleaning(raw)
		visuals =["Sentiment Class Analysis","Message length for each sentiment class","Popular Words Analysis","Word Cloud Analysis","Sentiment Analysis Insights"]
		visualselection = st.selectbox("Choose EDA visuals",visuals)

		if visualselection == "Sentiment Class Analysis":
			print('..... Creating the sentiment class analysis visual')
			title_tag("Sentiment Class Analysis")
			class_analysis(train)
		elif visualselection == "Message length for each sentiment class":
			print('...... Creating the sentiment class message length visual')
			title_tag('Message length for each sentiment class')
			class_dist(train)
		elif visualselection =="Popular Words Analysis":
			print('...... Creating the popular words visual')
			title_tag("Popular Words Analysis")
			popularwords_visualizer(train)
		elif visualselection == "Word Cloud Analysis":
			print('..... Creating the WordClouds for sentiment classes')
			title_tag("Word Cloud Analysis")
			wordcloud_visualizer(train)
			title_tag("Word Cloud for the entire Data set")
			tweet_cloud(train)
	if selection == "Model Metrics Evaluation":
		title_tag(selection)
		st.markdown("<h3 style='color:#00ACEE'>Performance Metrics for model evaluation</h3>",unsafe_allow_html=True)
		components.html(
			"""
			<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous" />
			<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
			<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="anonymous"></script>
			<div class="d-flex justify-content-center mb-0">
				<p class="font-weight-bold">We will evaluate our models using the the F1 Score which is the number of true instances for each label.</p>
			</div>
			"""
		)
		modelselection = ["Linear Support Vector Classifier","Support Vector Classifier","Ridge Classifier","Logisitic Regression Classifier","Stochastic Gradient Classifier"]
		modeloptions = st.selectbox("Choose Model Metrics By Model Type",modelselection)
		if modeloptions =="Linear Support Vector Classifier":
			title_tag("Evaluation Of the Linear Support Vector Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('LinearSVC-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('LinearSVC-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<ul>
					<li>We see that the LinearSVC model did a far better job at classifiying Pro and News sentiment classes compared to Decision Tree and RandomForest models with both classes achieving an f1 score of 0.85 and 0.81 respectively
					</li>
					<li>
						The LinearSVC model also did a far better job at classifying Anti sentiment class comapred to both the Decision tree and the Randrom Forest
					</li>
					<li>
						There was a slight improvement in the classification of neutral tweets with the LinearSVC, which is by far overshadowed by the improvements we see in other sentiments classes
					</li>
					<li>
						The LinearSVC has done a better job overall in classifying the sentiments, we see that Anti and Neutral sentiments have almost the same score, same applies with Pro and News sentiments which is consistent with the distribution of the data between the sentiment classes
					</li>
				</ul>
			</div>""",unsafe_allow_html=True)
		elif modeloptions =="Support Vector Classifier":
			title_tag("Evaluation Of the Support Vector Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Support Vector Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/SCV-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/SVC-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<ul>
					<li>
						Much like the LinearSVC we see that the the SVC does a really good job at classifying Pro sentiment class with a score of 0.81, followed by the News sentiment class with an f1 score of over 0.77.
					</li>
					<li>
						Unlike most of the models we've build this far, the Support Vector Classifier struggle more with classifying the Antisentiment class
					</li>
				</ul>
			</div>""",unsafe_allow_html=True)
		elif modeloptions =="Ridge Classifier":
			title_tag("Evaluation Of the Ridge Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Ridge Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('LinearSVC-cm.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<p>A Classification report is used to measure the quality of predictions from a classification algorithm.The confusion matrix heatmap shows the model's ability to classify positive samples, each class achieving a recall score of:</p>
				<ul>
					<li>
						Anti Climate Change : 0.46
					</li>
					<li>
						Neutral : 0.5
					</li>
					<li>
					Pro : 0.88
					</li>
					<li>
					News : 0.81
					</li>
				</ul>
				<p>The major concern here is that the Ridge classification classified 40% of of neutral tweets as Pro climate change tweets</p>
			</div>""",unsafe_allow_html=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('LinearSVC-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<p>The above bar graph shows the f1 score for each sentiment class using the Classificatio</p>
				<ul>
					<li>
						Much like the LinearSVC we see that the the Ridge classifier does a really good job at classifying Pro sentiment class with a score of 0.85, followed by the News sentiment class with an f1 score of over 0.79.
					</li>
					<li>
						Just like the support Vector Classifier, we see that Ridge Classifier does very good job at classifying the anti and neutral sentiment class
					</li>
			</div>""",unsafe_allow_html=True)
		elif modeloptions =="Logisitic Regression Classifier":
			title_tag("Evaluation Of the Logisitic Regression Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Logisitic Regression Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/LR-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Logisitic Regression Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/LR-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<ul>
					<li>
						The Logistic Regression Classifier performed almost as good as the LinearSVC at classifying each sentiment class with <b>Pro</b> and <b>News</b> sentiment class achieving f1 scores of 84 and 81 respetively
					</li>
				</ul>
			</div>""",unsafe_allow_html=True)
		elif modeloptions =="Stochastic Gradient Classifier":
			title_tag("Evaluation Of the Stochastic Gradient Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Stochastic Gradient Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/SGD-cm.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<p>
					A Classification report is used to measure the quality of predictions from a classification algorithm.<br/>
					The confusion matrix heatmap shows the model's ability to classify positive samples, each class achieving a recall score of:
				</p>
				<ul>
					<li>
						Anti Climate Change : 0.54
					</li>
					<li>
						Neutral : 0.53
					</li>
					<li>
						Pro : 0.85
					</li>
					<li>
						News : 0.84
					</li>
				</ul>
				<p>
					SGD classifier scored the highest in classification of positive classes for anti and neutral sentiment classes despite incorretly classsifying anti and neutral sentiment classes as Pro sentiment class 35% and 42% of the time respectively
				</p>
			</div>""",unsafe_allow_html=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Stochastic Gradient Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/SGD-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<p>
					The above bar graph shows the f1 score for each sentiment class using Stochastic Gradient Descent classifier
				</p>
				<ul>
					<li>
						The SGD classifier is just as good at classifying Pro sentiment classs as the LinearSVC both achieving an f1 score of 0.84 however falls short in classifying the rest of the sentiment classes
					</li>
				</ul>
			</div>""",unsafe_allow_html=True)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
