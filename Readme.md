Instructions to execute this code:

You can download this code as a zip file from this github repository or you can clone it using git clone https://github.com/priyatham1188/Dmml-Project.git

VScode:

1. Intially ,Please update the file locations or Path of train and test data within the code(Lines 15,19) as per your machine.
2. After updating the file locations run the code using py Dmml-final-code.py

Jupyter Notebook:

1. By launching the jupyter on your system please open the downaloded source code file from github
2. Update the path for the train and test data sets as per their current location in your machine
3. Save the code and Run it .



Introduction:
The aim in this challenge is to cluster the data into 16 clusters using a single cell RNA Sequencing dataset. "Single-cell RNA sequencing (scRNA-seq) is a widely used and powerful technique for analysing the whole transcriptome of many individual cells. The purpose is to develop a variety of machine learning models or strategies for detecting cell accuracy.

Data Preparation:

Feature Selection:

Variance threshold:
This technique is used to eliminate the constant or zero variance features within our dataset. When we explored the data, we found that there are many features with lot of zeroes or the features with almost zero variance which doesn’t account for model for understanding data to get good results. So first we have checked for zero variance columns/features however we didn’t find any then we decided to set a threshold value which is almost near to zero then we applied a threshold of 0.005 and able to remove 3029 features. Removal of these low variance features helped us to get the best accuracy.

Scaling:
After performing the feature selection, we have decided to scale the data using the standard scaler which further standardizes the data so that the mean and standard deviation are 0 and 1 across each feature.

Dimensionality Reduction:
PCA:
As we know this data set is a high dimensional in nature, so we have applied a linear technique such as Principal component analysis which performs a orthogonal transformation of the dataset to get principal components and this helps to choose the best variance features in lower dimensions without loosing much of the info from the dataset. In our final model we have taken 50 principal components which accounts for around 94.5% of variance with respect to the dataset. This helped us to train the model faster and achieve higher accuracy.


Model Training :

For Model training, we were flexible in trying out different models to train the data instead of sticking to one model and we feel this lead us to achieve better accuracy. The model that we have considered for this competition is Birch which gave us an accuracy of 88.53 on training dataset and 89.12 on test dataset.  Birch is a clustering algorithm which can mainly be used for large datasets as it makes fast for working. This algorithm clusters the large dataset into small clusters by summarizing the data and doesn’t alter most of the information. For the given dataset based on our pre-processing steps and the feature selection method that we used Birch helped us to achieve greater accuracy.

K-means
•	In this we have initially loaded the dataset and then we have applied k means algorithm with random state as zero and we have received the silhouette score as -0.0084 and accuracy of 63.4. 
•	Later, we considered using scaling techniques both min max and standard scaling. We got better accuracy with standard scaling and T-SNE as dimensionality reduction technique with a silhouette score as -0.0021 and accuracy of 80.15.
•	By applying the feature Selection for the dataset using K-Means with same scaling technique and PCA gave us higher accuracy compared to the pervious trial, i.e., silhouette score as 0.0013 and accuracy of 81.24.

Agglomerative:
•	In the second approach after loading the data into data frame we have scaled the data using the sklearn’s standard scaler and then we have applied the dimensionality reduction technique i.e., PCA where we chose the number of components as 4000 as it is giving the 80%varaiance ratio then we have applied this reduced dimensioned data in the k means algorithm and we have received the silhouette score as -0.002 and the accuracy of 75.3.
•	In the second attempt, using same model we have considered applying Variance Threshold feature selection. This lead to improvement in the score compared to the pervious K-Means model with a silhouette score as 0.0046 and accuracy of 83.64.

Birch:
•	In the third approach after loading the data into data frame we have normalized the data using the sklearn’s minmax scaler and then we have applied the dimensionality reduction technique i.e., PCA where we chose the number of components as 4000 as it is giving the 80%varaiance ratio then we have applied this reduced dimensioned data in the k means algorithm and we have received the silhouette score as 0.004 and the accuracy of 76.3.
•	Secondly, using the same approach we applied T-SNE as the dimensionality reduction technique which lead to decrease in the score. We achieved the silhouette score as -0.0063 and the accuracy of 71.41.
•	Later, To enhance the score we used feature selection method and applied the dimensionality reduction technique i.e., PCA where we have taken 50 principal components which accounts for around 94.5% of variance with respect to the dataset and standard scaler which helped us achieving a silhouette score of 0.0067 and the accuracy of 88.53. This is our highest accuracy we have achieved for the dataset. 


4.	Internal Evaluation:
Silhouette score:
For the internal evaluation we have used silhouette score where it ranges from   -1 to 1 which tell us how well the clustering or grouping is done on the given data set.
The silhouette score we achieved for the final result is 0.0067 for training and 0.011 for test data.


