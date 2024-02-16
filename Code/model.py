import re
import sys
import pandas as pd

# from utils.cosine_similarity import cosine_similarity as cs
# from utils.TfidfVectorizer import TfidfVectorizer as tfv
#region
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
#endregion
import streamlit as st
import pdfplumber
from pathlib import Path

st.title("Job recommendation")

with open('../554280/stop_words.txt','r') as file:
    all_get_stop_words = file.readlines()
all_get_stop_words = [word.strip() for word in all_get_stop_words]

def extract_data(feed):
    text=''
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for page in pages:
            text+=page.extract_text(x_tolerance=2)
    return text

c1, c2 = st.columns((3,2))
cv=c1.file_uploader('Upload your CV', type='pdf')
with c2: 
    option = st.selectbox(
        'Select the Industry here',
        ('IT', 'NON-IT'))

    st.write('You selected:', option)
# argumentList = sys.argv[1:]
# options = "o:f:"
# long_options = ["option", "filename"]
# arguments, values = getopt.getopt(argumentList, options, long_options)

if cv is not None:
    with st.spinner('Wait for it...'):

        cvtext = extract_data(cv)
        csv_path1 = Path(__file__).parents[1] / 'Cleaned_Datasets/JobsIT_Dataset.csv'
        csv_path2 = Path(__file__).parents[1] / 'Cleaned_Datasets/JobsNonIT_Dataset.csv'
        csv_path3 = Path(__file__).parents[1] / 'Jobs_Data/jobs_url.csv'
        # filename, option = "input.txt", "IT"

        # for argument, value in arguments:
        #     if argument in ["-f", "--filename"]:
        #         filename = value
        #     elif argument in ["-o", "--option"]:
        #         option = value

        if option == "IT":
            path = csv_path1
        elif option == "NON-IT":
            path = csv_path2
        else:
            print("Invalid value for argument option")
            sys.exit(1)
            
        tf = TfidfVectorizer()
        jobs = pd.read_csv(path)
        # with open(f"./predictor_files/{filename}",encoding="utf8") as f:
        #     prediction_text = f.readlines()
        prediction_text = str(cvtext)
        # prediction_text = ' '.join(prediction_text)
        prediction_text = re.sub('[^a-zA-Z]', ' ', prediction_text)
        prediction_text = prediction_text.lower()
        prediction_text = prediction_text.split()
        ps = PorterStemmer()
        prediction_text = [ps.stem(word) for word in prediction_text if not word in set(all_get_stop_words)]
        prediction_text = ' '.join(prediction_text)
        tfidf_jobs = tf.fit_transform(jobs["Description"]) # Tranform return document-term weighted matrix by taking set of documents
        tfidf_prediction_text = tf.transform([prediction_text])
        # print(tfidf_prediction_text)
        # print(tfidf_jobs)
        similarity_measure = cosine_similarity(tfidf_jobs, tfidf_prediction_text)
        print("sm: \n", similarity_measure)
        labels = jobs["Query"].unique()
        similarity_scores = {label: {"sum": 0, "count": 0} for label in labels} 
        for i in range(len(similarity_measure)):
            similarity_scores[jobs["Query"][i]]["sum"] += similarity_measure[i][0]
            similarity_scores[jobs["Query"][i]]["count"] += 1

        predictions = []
        for label in similarity_scores:
            avg = similarity_scores[label]["sum"]/similarity_scores[label]["count"]
            predictions.append([avg, label])
        predictions.sort(key = lambda key: -key[0])
        output_text = f"\nTop 3 predictions for you in {option} Industry:\n" + '\n'.join("\n" + x[1] for x in predictions[:3])
        print(output_text)
        st.write("\n========================================================================================")
        st.write("Output: ")
        st.write(output_text)

        st.write("\n========================================================================================")
        joburls = pd.read_csv(csv_path3)
        outputurls = str(f"\nTop 5 Job hirings open for you in {option} Industry: ")
        st.write(outputurls)
        for word in predictions[:3]:
            temp = joburls[joburls['Title'].str.contains(word[1])]
            if temp['URL'].size != 0 :
                st.write("\nFor "+ word[1] + ": \n")
                st.write(temp['URL'].head(5)) 
        
        st.balloons()
    st.success('Done!')