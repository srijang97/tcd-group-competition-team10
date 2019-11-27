# ML-Group-Project
This competition involved predicting whether a set of recommendations produced by a recommendation system for 3 applications: A digital library, a blog and an e-commerce website, would result in at least one of the recommendations being clicked.
Our was one of the top-performing solutions with an F1-Score of 0.78319 and rank 3 in the in-class Kaggle Competetion.

The Machine Learning pipeline for the final solution was as follows:
1) Data Cleaning: The data was split into three parts. One had the data for the blog and e-commerce website but in the training set, only those data points were taken which were viewed by the user. The other two datasets were made from the digital library data with one containing data where the algorithm type was Content-Based filtering and the other for all the other algorithms.
2) Missing values: The Missing values were imputed using median, mode and sometimes multiple modes derived by grouping data acording to some other columns.
3) Data Normalization: Standard Scaling was used for the numerical columns as some of them deviated from Normal behaviour
4) Data Encoding: Target encoding was used in which the catgeorical values are replaced by the ratio of positive instances in the target variable for that particular category. Additionally, this ratio was normalized by the overall average according to a weight.
5) Machine Learning model: Random Forest gave the best results with 100 estimators

The code for the final solution is given in the form of a Python script as well as a Jupyter Notebook in src/jupyter_notebooks/.
The final submission file is there in results/.


