from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movie descriptions
movie_descriptions = ["Inception is a science fiction action film directed by Christopher Nolan.",
    "The Shawshank Redemption is a drama film based on a Stephen King novella.",
    "Pulp Fiction is a neo-noir crime film directed by Quentin Tarantin. In the film, Miles goes on an adventure with Gwen Stacy / Spider-Woman across the multiverse where he meets a new team of Spider-People known as the Spider-Society, led by Miguel O' Hara / Spider-Man 2099, but comes into conflict with them over handling a new threat."]

# Create an instance of the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the movie descriptions and transform the descriptions into TF-IDF features
tfidf_matrix = vectorizer.fit_transform(movie_descriptions)

# Get the feature names (terms)
feature_names = vectorizer.get_feature_names()

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Print the feature names
print("\nFeature Names:")
print(feature_names)
