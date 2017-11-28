# TwoSigmaConnect

This is from Kaggle competition: [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries).
It is also used as the final project for Udacity Machine Learning Engineering Nanodegree.

### Problem 
The goal is to predict the interest level for a rental listing on the Renthop website. This is a typical classification problem. The possible interest levels are low, medium and high. With almost 70% listings are ranked as low, the data is skewed. So we must be very careful when the accuracy is around 70%. It may just predict every listing to be low. 

### Data
The characteristic of the project is the heterogeneous data. 
- Ordinary numerical features: price, bedrooms, bathrooms, longitude, latitude.
- High cardinality categorical features: manager id, building id, listing id, street address, display address.
- Text: description of the listings, features of the listings. 
- Image: photos of the listings.
- Time series: created time of the listings. 

We can also classify the data into
- Information about the housing condition: price, bathrooms, bedrooms, features, description, photos, building id.
- Information about the location: longitude, latitude, street address, display address.
- Information about the manager skills: manager id, created time.

### Model architecture
We design the model as a two-layer stack. The first layer transformed high-cardinality data, text, images and time series to ordinary numerical features using appropriate base estimators built on 5-fold splits.  The second layer combined the transformed results with other ordinary engineered features.

More precisely, we have six transformers in the first layer:
- ('price_pred', price_pred), do a regression of the log of price based on bathrooms, bedrooms, longitude and latitude. Then take the difference between predicted log price and the actual log price. 
- ('text_nn', text_nn), make transformations Tfidf followed by NMF of both the features and description. For description, we get 2 and 3-gram tokens and Train a 3-layer keras NN on the vectors. 
- ('photos', photos), take statistics from ImageStat of PIL library and then transform by a logistic regression.
- ('manager_i', manager), high cardinality categorical data, aggregate with expectation of the interest level. The cut off between the conditional expectation and ordinary expectation is the logistic function. 
- ('building_i', building), high cardinality categorical data, aggregate with expectation of the interest level. The cut off between the conditional expectation and ordinary expectation is the logistic function. 
- ('display_address', dis_add), high cardinality categorical data, aggregate with expectation of the interest level. The cut off between the conditional expectation and ordinary expectation is the logistic function. 

The meta estimator of the second layer is a LightGBM gradient boosting tree classifier. A gradient boosting tree classifier is usually the best choice for combining heterogeneous data. 

### Evaluation
The metric function is the log loss. Currently, the log loss on the leadboard is 0.53502, which agrees with our expectation. According to discussion board, this already reaches the best performances of single models of some of the top leaders before the leak information is provided by @KazAnova. It may be further improved by finer parameter tuning. 

### To Do in the future
It is possible to train the actual photos by transfer learning. However, the computing resources required is beyond my current laptop. Moreover, from the discussion, it seems not to benefit the model a lot so not many participants tried it. I think a more meaningful model may try to distinguish teh outdoor and indoor photos. 
