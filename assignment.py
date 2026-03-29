
Steps: Pre-processing -> Feature Engineering -> Cosine Similarity -> Classification -> Clustering
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# 1. Pre-processing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
corpus = [
    "Natural Language Processing is amazing!",
    "Machine Learning is part of Artificial Intelligence.",
    "Text classification is a supervised learning task."
]

preprocessed_corpus = [preprocess(doc) for doc in corpus]
print("Preprocessed Corpus:", preprocessed_corpus)
# 2. Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_corpus)
print("\nTF-IDF Feature Matrix:\n", X.toarray())
# 3. Cosine Similarity
similarity_matrix = cosine_similarity(X)
print("\nCosine Similarity Matrix:\n", similarity_matrix)
# 4. Text Classification (Supervised)
# Example labels (0 = category A, 1 = category B)
labels = [0, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nClassification Accuracy:", accuracy_score(y_test, y_pred))
# 5. Text Clustering (Unsupervised)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
print("\nCluster Assignments:", kmeans.labels_)
