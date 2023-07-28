# Import the required libraries
from flask import Flask, render_template, request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bs4 import BeautifulSoup
import requests
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000


# Load the fake news dataset
df = pd.read_csv('train.csv')
df = df.dropna()
vectorized = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorized.fit_transform(df['text'])

# Initialize Flask application
app = Flask(__name__)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


# Train the DNN model
def train_model():
    # Prepare the features and labels
    X = df['text']
    y = df['label']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Vectorize the text data
    X_train_vectorized = vectorized.fit_transform(X_train)
    # Train the DNN model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train_vectorized, y_train)
    # Evaluate the model
    X_test_vectorized = vectorized.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    print('Model Accuracy :',accuracy)
    print('Confusion Metrix :',confusion)
    return model

# Perform K-means clustering and determine optimal clusters
def determine_optimal_clusters(tfidf_matrix):
    try:
        sse = []
        silhouette_scores = []

        for k in range(2, 15):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(tfidf_matrix)
            sse.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(tfidf_matrix, kmeans.labels_))

        print('SSE Values:', sse)
        print('Silhouette Score:', silhouette_scores)

        return sse, silhouette_scores

    except Exception as e:
        print("An error occurred during cluster determination:", str(e))
        return None,None

# Function to cluster and summarize input news
def cluster_and_summarize(news_text, k, vectorizer):
    try:
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        tfidf_matrix = vectorizer.transform(df['text'])
        kmeans.fit(tfidf_matrix)

        # Assign input news to clusters
        news_vectorized = vectorizer.transform([news_text])
        cluster_label = kmeans.predict(news_vectorized)[0]

        # Summarize headlines of the assigned cluster
        cluster_headlines = df[kmeans.labels_ == cluster_label]['title']
        headlines = ' '.join(cluster_headlines)

        return headlines

    except Exception as e:
        print("An error occurred during clustering and summarization:", str(e))
        return None


# Function to perform Google search
def search_google(query, num_results):
    try:
        search_results = []
        api_key = "AIzaSyAVtMUqMQ-yRengeU2HbMXYB3uLVNPRmTM"  # Replace with your Google Custom Search API key
        cx = "973a14399ba9a4289"  # Replace with your Google Custom Search Engine ID
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&num={num_results}"
        response = requests.get(url)
        data = response.json()
        items = data.get('items', [])
        for item in items:
            search_results.append(item['link'])
        print('Top Websites Extracted by Google Search:', search_results)
        return search_results

    except Exception as e:
        print("An error occurred during Google search:", str(e))
        return []


# Function to scrape and summarize webpages
def scrape_webpages(urls):
    contents = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser',from_encoding='utf-8')
        text = soup.get_text(separator=' ')
        contents.append(text)
    return contents

# Function for extractive summarization
def extractive_summarization(texts):
    summaries = []
    for text in texts:
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        summary = ' '.join(sentences[:3])  # Extract first 3 sentences as a summary
        summaries.append(summary)
    return summaries


# Calculate cosine similarity using the loaded model
def calculate_cosine_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# Calculate Jaccard similarity
def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Function to generate final article
def generate_final_article(summaries, cosine_similarities):
    try:
        final_article = []
        for summary, similarity in zip(summaries, cosine_similarities):
            if any(score <= 0.5 for score in similarity):
                final_article.append(summary)
        return ' '.join(final_article)

    except Exception as e:
        print("An error occurred during final article generation:", str(e))
        return " "


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the input news from the user
        news_text = request.form['news']

        # Perform K-means clustering
        sse, silhouette_scores = determine_optimal_clusters(tfidf_matrix)
        if sse is None or silhouette_scores is None:
            raise ValueError("Cluster determination failed.")

        # Choose optimal clusters
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

        # Split input news into headlines and summarize using TFIDF vectorization
        headlines = cluster_and_summarize(news_text, optimal_k, vectorized)

        # Perform Google search and retrieve top 5 search results
        search_results = search_google(headlines, 5)

        # Web scraping and summarization of top 5 webpages
        content = scrape_webpages(search_results[:5])
        summaries = extractive_summarization(content)


        cosine_similarity_scores = [calculate_cosine_similarity(news_text, summary) for summary in summaries]
        jaccard_similarity_scores = [calculate_jaccard_similarity(news_text, summary) for summary in summaries]

        # Check if the news is classified as fake
        if max(cosine_similarity_scores) >= 0.5 or max(jaccard_similarity_scores) >= 0.5:
            classification = "100% Real"
        else:
            # Train the DNN model
            model = train_model()
            if model is None:
                raise ValueError("Model training failed.")
            news_vectorized = vectorized.transform([news_text])
            prediction = model.predict(news_vectorized)
            classification = 'Fake' if prediction[0] == 0 else "Maybe True"

            # Generate final article based on similarity scores
            final_article = generate_final_article(summaries, cosine_similarity_scores)

            # Render the results on the webpage
            print("Cosine Similarity Scores:", cosine_similarity_scores)
            print("Jaccard Similarity Scores:", jaccard_similarity_scores)
            print("Classification:", classification)

            result ={"Cosine Similarity Scores:", cosine_similarity_scores,"Jaccard Similarity Scores:", jaccard_similarity_scores,"Classification:", classification}

            # Render the results on the webpage
            return jsonify(result)
            #return render_template('result.html', classification=classification, final_article=final_article)

    except Exception as e:
        print("An error occurred during news detection:", str(e))
        # return render_template('error.html', error_message=str(e))


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

