
import os
from sklearn.feature_extraction.text import TfidfVectorizer


# we have to declare directory path where our data is present.
document_path="Enter the location to the files for feature extraction."

# We are reading all the filenames and creating a list filenames.
filenames=[os.path.join(document_path, each)
            for each in os.listdir(document_path)]
 
# We have to instantiate the Ft-Idf vectorize along with this preprocessing is built in.
# For preprocessing we tokenize based on regex pattern given below.
# We are using english set stopwords.
# We will use on 1 or 2 grams. 
# We will extract 200 features from the entire data set. 
 
vectorizer = TfidfVectorizer(input='filename', 
                             max_features=200,
                             token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b',
                             stop_words='english',
                             ngram_range=(1, 2))

# we fit the douments to the vectorizer.
tfidf_result = vectorizer.fit_transform(filenames)

# We extract the features that have been used in the Tf-Idf Vectorizer.
features=vectorizer.get_feature_names()


# We have to declare the file name and path where we store the features.
output_path="Enter the location where do you want to save the output file"

# We shall write the features into a new txt file. 
with open(output_path+"feature_list.txt", 'w') as file_handler:
    for item in features:
        file_handler.write("{}\n".format(item))

