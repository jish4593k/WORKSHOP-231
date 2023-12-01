import requests
from bs4 import BeautifulSoup
import sys
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def get_results(loop, key):
    for i in range(1, loop):
        print("[+] Page %d" % (i + 1))
        url =
        response = requests.get(url)
        html = BeautifulSoup(response.text, "html.parser")

        tds = html.findAll("td", {"class": "field_domain"})
        for td in tds:
            print(td.findAll("a")[0]["title"])

def search_expired_domains(key):
    url = y}"
    response = requests.get(url)
    html = BeautifulSoup(response.text, "html.parser")
    result = html.select_one('strong').text.replace(',', '')
    print(f"[*] Total results: {result}")

    if int(result) < 25:
        return
    elif int(result) > 550:
        print("[!] Too many results, only get 550 results.")
        print("[*] 21 requests will be sent.")

        print("[+] Page 1")
        tds = html.findAll("td", {"class": "field_domain"})
        for td in tds:
            print(td.findAll("a")[0]["title"])
        get_results(21, key)

    else:
        print(f"[*] {int(result) // 25 + 1} requests will be sent.")

        print("[+] Page 1")
        tds = html.findAll("td", {"class": "field_domain"})
        for td in tds:
            print(td.findAll("a")[0]["title"])
        get_results(int(result) // 25 + 1, key)

def text_clustering_example():
    # Function to demonstrate text clustering using scikit-learn
    documents = [
        "Text clustering is an interesting task.",
        "It involves grouping similar documents together.",
        "Scikit-learn provides useful tools for this task."
    ]

    # Convert the documents to TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)

    # Print cluster labels
    print("[*] Cluster Labels:")
    print(kmeans.labels_)

def neural_network_example():
   
    print("[*] Neural Network Example:")
    # Define a simple sequential model
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Generate random input data
    X_train = np.random.random((100, 5))
    y_train = np.random.randint(2, size=(100, 1))

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)

if __name__ == "__main__":
    print("GetExpiredDomains - Search for available domain from expireddomains.net")
    print("Author: 3gstudent\n")

    if len(sys.argv) != 2:
        print('Usage:')
        print('    GetExpiredDomains.py <Search String>')
        sys.exit(0)
    
    # Search for expired domains
    search_expired_domains(sys.argv[1])
    
    # Additional Advanced Features
    print("\nAdditional Advanced Features:")
    
    # Text Clustering Example
    print("\nText Clustering Example:")
    text_clustering_example()
    
    # Neural Network Example
    print("\nNeural Network Example:")
    neural_network_example()

    print("[*] All Done")
