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
# from wordcloud import WordCloud, ImageColorGenerator
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

# new
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
#-------------resources\imgs\pkl\count_vect.pkl
# Vectorizer
news_vectorizer = open("resources/imgs/pkl/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
retweet = 'RT'
mask = np.array(Image.open('resources/imgs/images for streamlit/10wmt-superJumbo-v4.jpg'))
import streamlit.components.v1 as components




def put_numbers_on_bars(axis_object):
    """
    Function to plot labels above countplot bars.
    """
    for p in axis_object.patches:
        axis_object.text(p.get_x() + p.get_width()/2., p.get_height(),'%d' % round(p.get_height()), fontsize=11,ha='center', va='bottom')

def sent_decider(compound):
    """
    Function to determine if sentiment is positive, neutral or negative.
    """
    neutral_point = 0.00
    if compound > neutral_point:
        return 'positive'#1
    elif compound < -neutral_point:
        return 'negative' #-1
    else: 
        return 'neutral'#0



	




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
from nltk.tokenize import TreebankWordTokenizer
def data_cleaning(df):
		def preprocess(text):
			"""This function takes in pandas dataframe, removes URL hyperlinks, stopwords, punctuation noises,contractions and lemmatize the text."""

			tokenizer = TreebankWordTokenizer() 
			lemmatizer = WordNetLemmatizer()
			stopwords_list = stopwords.words('english')
			point_noise = string.punctuation + '0123456789'
			
			cleanText = re.sub(r'@[a-zA-Z0-9\_\w]+', '', text)#Remove @mentions
			cleanText = re.sub(r'#[a-zA-Z0-9]+', '', cleanText) #Remove '#' symbols
			cleanText = re.sub(r'RT', '', cleanText)#Remove RT from text
			#Panding Contractions
			# specific
			cleanText = re.sub(r"won\'t", "will not", cleanText)
			cleanText = re.sub(r"can\'t", "can not", cleanText)
			#Panding Contractions
			# general
			cleanText = re.sub(r"n\'t", " not", cleanText)
			cleanText = re.sub(r"\'re", " are", cleanText)
			cleanText = re.sub(r"\'s", " is", cleanText)
			cleanText = re.sub(r"\'d", " would", cleanText)
			cleanText = re.sub(r"\'ll", " will", cleanText)
			cleanText = re.sub(r"\'t", " not", cleanText)
			cleanText = re.sub(r"\'ve", " have", cleanText)
			cleanText = re.sub(r"\'m", " am", cleanText)
			cleanText = ''.join([word for word in cleanText if word not in point_noise]) #Removing punctuations and numbers.
			cleanText = cleanText.lower() #Lowering case
			cleanText = "".join(word for word in cleanText if ord(word)<128) #Removing NonAscii
			cleanText = tokenizer.tokenize(cleanText) #Coverting each words to tokens
			cleanText = [lemmatizer.lemmatize(word) for word in cleanText if word not in stopwords_list] #Lemmatizing and removing stopwords
			cleanText = [word for word in cleanText if len(word) >= 2]
			# cleanText = ' '.join(cleanText)
			#return cleanText
			return cleanText
		df["message"]=df["message"].apply(preprocess)

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

# st.cache(suppress_st_warning=True,allow_output_mutation=True)
# def wordcloud_visualizer(df):
#     news = df['message'][df['sentiment']==2].str.join(' ')
#     neutral = df['message'][df['sentiment']==2].str.join(' ')
#     pro = df['message'][df['sentiment']==2].str.join(' ')
#     anti = df['message'][df['sentiment']==2].str.join(' ')
#     #Visualize each sentiment class
#     fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
#     news_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter').generate(str(news))
#     axis[0, 0].imshow(news_wordcloud)
#     axis[0, 0].set_title('News Class',fontsize=14)
#     axis[0, 0].axis("off") 
#     neutral_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(neutral))
#     axis[1, 0].imshow(neutral_wordcloud)
#     axis[1, 0].set_title('Neutral Class',fontsize=14)
#     axis[1, 0].axis("off") 
    
#     pro_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(pro))
#     axis[0, 1].imshow(pro_wordcloud)
#     axis[0, 1].set_title('Pro Class',fontsize=14)
#     axis[0, 1].axis("off") 
#     anti_wordcloud = WordCloud(width=900, height=900, background_color='white', colormap='winter', min_font_size=10).generate(str(anti))
#     axis[1, 1].imshow(anti_wordcloud)
#     axis[1, 1].set_title('Anti Class',fontsize=14)
#     axis[1, 1].axis("off")
#     st.pyplot(fig)


# def tweet_cloud(df):
# 	mask = np.array(Image.open('resources/imgs/images for streamlit/10wmt-superJumbo-v4.jpg'))
# 	words = df['message']
# 	allwords = []
# 	for wordlist in words:
# 		allwords += wordlist
# 	mostcommon = FreqDist(allwords).most_common(10000)
# 	wordcloud = WordCloud(width=1000, height=1000, mask = mask, background_color='white').generate(str(mostcommon))
# 	fig = plt.figure(figsize=(30,10), facecolor='white')
# 	plt.imshow(wordcloud, interpolation="bilinear")
# 	plt.axis('off')
# 	plt.tight_layout(pad=0)
# 	st.pyplot(fig)

def prediction_output(predict):
    if predict[0]==-1:
        output="The text has been classified as Anti, which means doesn't  believe in man-made climate change. "
        st.error("Results: {}".format(output))
    elif predict[0]==0:
        output="The text has been classified as Neutral, which means that neither support nor refute climate change theories"
        st.info("Results: {}".format(output))
    elif predict[0]==1:
        output ="The text has been classified as Pro, which means in favor of man-made climate change"
        st.success("Results: {}".format(output))
    else:
        output = "The text has been classified as News which means this tweets regarding climate change is likely  based on facts"
        st.warning("Results: {}".format(output))

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
	logo = Image.open('resources/imgs/images for streamlit/logo1.png')
	st.sidebar.image(logo, use_column_width=True)
	set_bg_hack('resources/imgs/images for streamlit/Siders analytics presentation.gif')
	st.markdown("<h1 style='color:#00ACEE'align='center'>DeKabon</h3>",unsafe_allow_html=True)
	options = ["Text Classification","Exploratory Data Analysis","Model Metrics Evaluation","About Predict","Company Profile"]
	selection = st.sidebar.selectbox("Menu", options)
	# Building out the "Information" page
	if selection == "Company Profile":
		
		# title_tag("Siders Analytics EST 2022")
		# You can read a markdown file from supporting resources folder
		st.image('resources/imgs/images for streamlit/p1n.png', use_column_width=True)
		# def st_display_pdf(pdf_files):
		# with open("resources/imgs/images for streamlit/pa2.pdf","rb") as f:
		# 	base64_pdf = base64.b64encode(f.read()).decode('utf-8')
		# pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1200" type="application/pdf"></iframe>'
		# st.markdown(pdf_display, unsafe_allow_html=True) 
		# # st_display_pdf("pa2.pdf")
		st.image('resources/imgs/images for streamlit/latest1.jpg', use_column_width=True)

			
	if selection == "About Predict":
		title_tag("Climate Change Sentiment Analysis brougth to you by siders analytics")
		st.image('resources/imgs/images for streamlit/twitter-removebg-preview.png', caption='Tweeet Attack',use_column_width=True)


		title_tag("Introduction")
		st.image('resources/imgs/images for streamlit/intro.png',use_column_width=True)
		title_tag("Problem Statement")
		
		st.image('resources/imgs/images for streamlit/problem statment.png',use_column_width=True)
		st.image('resources/imgs/images for streamlit/output-onlinegiftools.gif',use_column_width=True)
		# st.markdown("<h3 style='color:#00ACEE'>  zf5 #SIDERS ANALYTICS</h3><br/>",unsafe_allow_html=True)
		title_tag("zf5 #SIDERS ANALYTICS")
		
	# Building out the predication pages
	if selection == "Text Classification":
		with st.sidebar:
			choose = option_menu("App Models", ["Linear SVC","Stochastic Gradient Descent","Logisitic Regression Classifier"],
								icons=['tropical-storm', 'tree', 'kanban', 'bar-chart-steps','bezier', 'alt','bezier2'],
								menu_icon="app-indicator", default_index=0,
								styles={
				"container": {"padding": "5!important", "background-color": "#f1f2f6"},
				"icon": {"color": "orange", "font-size": "25px"}, 
				"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
				"nav-link-selected": {"background-color": "#585858"},		
					}
					)			
		
		if choose =="Linear SVC":
			title_tag("Linear SVC")
			tweet_text = st.text_area("Enter Text to be predicted using Linear SVC", "Type Here")
			if st.button("Predict text class with Linear SVC"):
				pred = joblib.load(open(os.path.join("resources/imgs/pkl/ridge_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
			st.info("## Linear SVC\n *The Linear SVC*, or SVC, is a linear model that can be used to solve classification and regression issues. It can handle both linear and nonlinear problems and is useful for a wide range of applications. SVM is a basic concept: As seen in the picture below, the method generates a line or hyperplane that divides the data into classes.")
			st.image('https://i.ibb.co/JQqm2vV/A-Group-2-1.png',use_column_width=True)

		if choose =="LGBMClassifier":
			title_tag("LGBMClassifier")
			tweet_text = st.text_area("Enter Text to be predicted using LGBMClassifier", "Type Here")
			if st.button("Predict text class with Decision Tree Classifier"):
				pred = joblib.load(open(os.path.join("resources/imgs/pkl/ridge_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
			st.info("## LGBMClassifier\n LightGBM is a gradient boosting framework that employs a tree-based learning method as well. LightGBM differs from other tree-based algorithms in that it develops trees vertically, whilst other algorithms grow them horizontally. This implies that Light GBM grows trees leaf-by-leaf, whereas other algorithms grow trees level-by-level.LightGBM will grow the leaf with the greatest delta loss. Leaf-wise algorithms can minimize more loss than level-wise algorithms while expanding the same leaf.The LightGBM architecture has a number of advantages.\n\n- Increased training efficiency and speed.\n- Memory use is reduced.\n- Increased precision.\n- Parallel and GPU learning are supported.\n- Capable of dealing with massive amounts of data \nThe diagrams below show how LightGBM and other boosting techniques are implemented.")
			st.image('https://i.ibb.co/7vtySS2/A-Group-2-2.png',use_column_width=True)

		if choose =="RandomForest Classifier":
			title_tag("RandomForest Classifier")
			tweet_text = st.text_area("Enter Text to be predicted using RandomForest Classifier", "Type Here")
			if st.button("Predict text class with Decision Tree Classifier"):
				pred = joblib.load(open(os.path.join("resources/imgs/pkl/ridge_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
			st.info("## RandomForest Classifier\n* Random forest is a supervised learning technique that may be used to predict and classify data. A forest is made up of a variety of different trees. Unlike decision trees, it is thought that the more trees a forest contains, the more robust it is. By building trees on random subsets, `Random Forest` eliminates overfitting.\n\nThere are four steps to the Random Forest algorithm.\n1. Chooses a collection of random samples from a dataset.\n2. For each sample, create a decision tree and extract a prediction result from each decision tree.\n3. Cast a vote for each expected outcome.\n4. As the final forecast, choose the prediction with the most votes.\n\n The figure below shows a visual depiction of a Random Forest classifier.")
			st.image('https://i.ibb.co/ZXF9744/A-Group-2.png',use_column_width=True)

				
		if choose =="Stochastic Gradient Descent":
			title_tag("Stochastic Gradient Descent")
			tweet_text = st.text_area("Enter Text to be predicted using Stochastic Gradient Classifer","Type Here")
			if st.button("Predict text class with Stochastic Gradient Classifer"):
				pred = joblib.load(open(os.path.join("resources/imgs/pkl/SGD_tfidf.pkl"),"rb"))
				predict = pred.predict([tweet_text])
				prediction_output(predict)
			st.info("## Stochastic Gradient Descent\n**Stochastic Gradient Descent (SGD)**  is a quick and easy way to fit linear classifiers and regressors to convex loss functions. Despite the fact that SGD has been around for a long time, it has only recently gained traction in the context of large-scale learning. \n The following are some of the benefits of Stochastic Gradient Descent:\n* Effectiveness.\n* Simplicity of implementation (lots of opportunities for code tuning).")
			st.image('https://i.ibb.co/V2XhNkf/A-Group-2-4.png',use_column_width=True)			

		
		elif choose =="Logisitic Regression Classifier":
			title_tag("Logisitic Regression Classifier")
			logi_text = st.text_area("Enter Text to be predicted using Logisitic Regression Classifier","Type Here")
			if st.button("Predict text class with Logisitic Regression Classifier"):
				pred = joblib.load(open(os.path.join("resources/imgs/pkl/logreg_tfidf.pkl"),"rb"))
				predict = pred.predict([logi_text])
				prediction_output(predict)
			st.info("## Logisitic Regression Classifier\n The statistical approach of **logistic regression** is used to predict binary classes. It can, for example, be utilized to solve cancer detection issues. Logistic regression classifies each data point into the best-estimated class based on its likelihood of belonging to that class.\nThe sigmoid function is used by logistic regression models to create predictions, as seen in the diagram below:")
			st.image('https://i.ibb.co/Rb9F7Zr/A-Group-2-3.png',use_column_width=True)

	if selection == "Exploratory Data Analysis":
		with st.sidebar:
			choose = option_menu("EDA visuals", ["Sentiment Class Analysis","Name Entity Recognition","Word Cloud Analysis","Popular Words Analysis"],
								icons=['emoji-smile', 'person-circle', 'file-earmark-word-fill', 'cloud-haze2-fill'],
								menu_icon="app-indicator", default_index=0,
								styles={
				"container": {"padding": "5!important", "background-color": "#f1f2f6"},
				"icon": {"color": "orange", "font-size": "25px"}, 
				"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
				"nav-link-selected": {"background-color": "#585858"},		
					}
					)

		if choose == "Sentiment Class Analysis":
			title_tag("Sentiment Class Analysis")
			st.image('resources/imgs/images for streamlit/barani.gif',use_column_width=True)
			
			
		elif choose == "Name Entity Recognition":
			title_tag('Name Entity Recognition')
			st.image('resources/imgs/images for streamlit/ner.png',use_column_width=True)
			
			

		elif choose =="Popular Words Analysis":
			title_tag("Popular Words Analysis")
			st.image('resources/imgs/images for streamlit/allf.png',use_column_width=True)
			

		elif choose == "Word Cloud Analysis":
			title_tag("Word Cloud for the entire Data set")
			st.image('resources/imgs/images for streamlit/wortwee.gif',use_column_width=True)

	if selection == "Model Metrics Evaluation":
		with st.sidebar:
			choose = option_menu("Model Evaluation", ["Linear Support Vector Classifier","Logisitic Regression Classifier","Stochastic Gradient Classifier"],
								icons=['tropical-storm', 'tree', 'kanban', 'bar-chart-steps','bezier', 'alt','bezier2'],
								menu_icon="app-indicator", default_index=0,
								styles={
				"container": {"padding": "5!important", "background-color": "#f1f2f6"},
				"icon": {"color": "orange", "font-size": "25px"}, 
				"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
				"nav-link-selected": {"background-color": "#585858"},		
					}
					)

		title_tag(selection)
		st.markdown("<h3 style='color:#00ACEE'align='center'>Performance Metrics for model evaluation</h3>",unsafe_allow_html=True)
		# components.html(
		# 	"""
		# 	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous" />
		# 	<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
		# 	<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="anonymous"></script>
		# 	<div class="d-flex justify-content-center mb-0">
		# 		<p class="font-weight-bold"></p>
		# 	</div>
		# 	"""
		# )
		st.info(	"We will evaluate our models using the the F1 Score which is the number of true instances for each label")		
		# modelselection = ["Linear Support Vector Classifier","Support Vector Classifier","Ridge Classifier","Logisitic Regression Classifier","Stochastic Gradient Classifier"]
		# modeloptions = st.selectbox("Choose Model Metrics By Model Type",modelselection)
		if choose =="Linear Support Vector Classifier":
			title_tag("Evaluation Of the Linear Support Vector Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/LinearSVC-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/LinearSVC-f1-score.png',use_column_width=True)
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
		elif choose =="Support Vector Classifier":
			title_tag("Evaluation Of the Support Vector Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Support Vector Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/SCV-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Linear Support Vector Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/SVC-f1-score.png',use_column_width=True)
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
		elif choose =="Ridge Classifier":
			title_tag("Evaluation Of the Ridge Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Ridge Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/LinearSVC-cm.png',use_column_width=True)
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
			st.image('resources/imgs/images for streamlit/LinearSVC-f1-score.png',use_column_width=True)
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
		elif choose =="Logisitic Regression Classifier":
			title_tag("Evaluation Of the Logisitic Regression Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Logisitic Regression Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/LR-cm.png',use_column_width=True)
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Logisitic Regression Classifier F1-Score predictive accuracy</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/LR-f1-score.png',use_column_width=True)
			st.markdown("""<div>
				<h5 style='color:#00ACEE'>Key Observations</h5>
				<ul>
					<li>
						The Logistic Regression Classifier performed almost as good as the LinearSVC at classifying each sentiment class with <b>Pro</b> and <b>News</b> sentiment class achieving f1 scores of 84 and 81 respetively
					</li>
				</ul>
			</div>""",unsafe_allow_html=True)
		elif choose =="Stochastic Gradient Classifier":
			title_tag("Evaluation Of the Stochastic Gradient Classifier")
			st.markdown("<h4 style='color:#00ACEE; text-align:center !important'>Stochastic Gradient Classifier Confusion Matrix</h4>",unsafe_allow_html=True)
			st.image('resources/imgs/images for streamlit/SGD-cm.png',use_column_width=True)
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
			st.image('resources/imgs/images for streamlit/SGD-f1-score.png',use_column_width=True)
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
