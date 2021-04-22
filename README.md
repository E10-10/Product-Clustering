# Product-Clustering

“In this competition, you’ll apply your machine learning skills to build a model that predicts which items are the same products.” https://www.kaggle.com/c/shopee-product-matching 

Challenge: Clustering of products based on the image-pixel-data, their titles and pHashs.

Given 34250 pictures of merchandise, we try to develop algorithms to find identical products. To reach that objective, we used both visual and semantic information contained in the data set. 


This repository contains (among others) the following files:

- NLP-Notebook: In this notebook we try to use methods of Natural Language Processing (Word2Vec and TfidfVectorizer) in order to perform a good clustering of our data. Since many titles contain meaningful information about the products, this semantic insights should not be ignored when making a prediction. The most important finding of the NLP approach is that we can significantly reduce the amount of images which possibly belong to the cluster of a particular image. 
- CNN-Notebook: Here we reduced our dataset from 34.000 pictures to roughly 4.000 pictures. These 4.000 pictures belong to 200 different clusters, which means that a cluster contains 20 images on average. On this reduced data set we trained a convolutional neural network using TensorFlow. We then used the flattened model without the fully connected layers in order to generate feature vectors of each picture. Given these feature vectors, we were able to cluster the images by defining metrics on the vector space to measure similarity between the images. The reduced dataset was chosen for the sake of a lower computation time. 
- Phash-Notebook: We transform the visual pixel information contained in the images to pHashes. This enables us to measure distances between two images. Of course, the pHash Approach alone is not really suitable to make precise predictions whether two images display the same item or not. However, it turned out that when combined with NLP methods, prediction accuracy could be improved.  
- Color histogram: We created a color fingerprint of each image that clusters the colors found in each pixel to a histogram of eight different categories (the colors). The clustering approach was to define similarity via the HUE representation of pixels.
- Captation prediction: We used ten widely known classifiers (e.g. VGG16, VGG19, ResNet-50), pre-trained on ImageNet, to create new labels for the image-data. We then clustered these labels via the NLP methods described above.


In the respective notebooks you’ll find more information about our precise results. They also contain information about what possible future work on this data set could look like.
 
The data can be downloaded from: https://www.kaggle.com/c/shopee-product-matching 
