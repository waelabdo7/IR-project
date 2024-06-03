from flask import Flask, request, render_template
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_csv('data2.csv')

# Load resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    # Check if the text is NaN or not a string
    if pd.isnull(text) or not isinstance(text, str):
        return text
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [w for w in words if w not in stop_words]
    # Return the processed text
    return ' '.join(words)

def preprocess_csv_file(input_path, output_path, num_rows):
    # Read the CSV file using Pandas
    data1 = pd.read_csv(input_path, nrows=num_rows, low_memory=False)
    # Process the text in the DataFrame
    data1['processed_text'] = data1['title'].apply(preprocess)
    # Save the processed data to a new CSV file
    data1.to_csv(output_path, index=False)

# Replace NaN values with an empty string
df['title'] = df['title'].fillna('')

# Apply preprocessing to the 'title' column
df['title'] = df['title'].apply(preprocess)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 2))
# Fit and transform the 'title' column
X = vectorizer.fit_transform(df['title'])
# Create the document-term matrix
document_term_matrix = X

def process_query(query):
    # Preprocess the query
    return preprocess(query)

# Define a function to get all similar results for a query
def get_all_results(query):
    # Convert the query into a vector
    query_vector = vectorizer.transform([query])
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, document_term_matrix).flatten()
    # Get the indices of the results
    all_indices = cosine_similarities.argsort()[::-1]
    # Limit: Set a maximum number of results
    top_indices = all_indices[:1000]
    # Check for matches before returning results
    if cosine_similarities[top_indices[0]] == 0:
        return pd.DataFrame() # Return an empty DataFrame if there are no matches
    # Get the results and add a similarity column
    results = df.iloc[top_indices].copy()
    results['similarity'] = cosine_similarities[top_indices]
    return results

def get_query_suggestions(query, top_n=5):
    # Convert the query into a vector
    query_vector = vectorizer.transform([query])
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, document_term_matrix).flatten()
    # Get the indices of the top suggestions
    all_indices = cosine_similarities.argsort()[::-1]
    top_indices = all_indices[:top_n]
    # Get the suggestions
    suggestions = df.iloc[top_indices]['title'].tolist()
    return suggestions

# Initialize the Flask application
app = Flask(__name__)

# Define the main route for the application
@app.route('/', methods=['GET', 'POST'])
def index():
    suggestions = []
    if request.method == 'POST':
        # Get the query from the form
        query = request.form['query']
        # Process the query
        processed_query = process_query(query)
        # Get all results
        results = get_all_results(processed_query)
        # Get query suggestions
        suggestions = get_query_suggestions(processed_query)
        # Render the template with results and suggestions
        return render_template('index.html', query=query, results=results.to_html(), suggestions=suggestions)
    # Render the template with suggestions
    return render_template('index.html', suggestions=suggestions)

# Run the application if this is the main module
if __name__ == '__main__':
    app.run(debug=True)
