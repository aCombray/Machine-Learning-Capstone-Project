# TwoSigmaConnect

This is from Kaggle competition: [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries).
It is also used as the final project for Udacity Machine Learning Engineering Nanodegree.

### Plan
There are five parts of the model.
- The price information related to the basic information, bedrooms, bathrooms.
- The location information, includes longitude, lattitude, addresses.
- The listing information: manager id, building id and listing id.
- Text Description of the listing
- Photos.

We plan to have five different models for each of the data above and the combine their prediction to make the final prediction. I will try to use a Neural netweek with 5 nodes to combine them. 

### Finished
There are five parts of the model.
- The price information related to the basic information, bedrooms, bathrooms.
- The location information, includes longitude, lattitude, addresses.
- The listing information: manager id, building id and listing id.

We have studied the first three parts and tested the model from them on toy model (60% of the training data). The performance is around 0.58 log loss.

### To Do
The next is text and photos. 
