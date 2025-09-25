## Housing Price Prediction: Linear Regression
     This project implements Simple and Multiple Linear Regression (LR) models to predict housing prices, focusing on practical model building and interpretation.

 ## Objectives:
     The primary goals were to implement and compare the Simple and Multiple Linear Regression models. We focused on handling categorical data, evaluating performance using key metrics, and interpreting the model's coefficients.

## Key Implementation Details:
      To prepare the data, we first converted all binary categorical features (like mainroad and airconditioning) into numerical values, where 'yes' becomes 1 and 'no' becomes 0. The multi-level feature, furnishingstatus, was converted using One-Hot Encoding to create separate numerical columns, a standard technique to prevent bias in the linear model.

      The complete dataset was split into 70% for training the models and 30% for testing their performance.

## Results Summary:
  We implemented two models:

  1. Simple LR: Uses only the area of the house to predict the price.

  2. Multiple LR: Uses a comprehensive set of features, including area, bedrooms, bathrooms, stories, parking, and the processed amenity and location features.

    The Multiple Linear Regression model was significantly better at predicting house prices. Its R-squared (R2) value was ≈0.65, meaning it successfully explained 65% of the variation in price. In comparison, the Simple LR model only achieved an R2 of ≈0.28.

    The Multiple LR model's average prediction error (MAE) on the test data was about $680,000, which is much lower than the Simple LR's error of over $1,000,000.

## Key Findings:
    By examining the coefficients of the Multiple LR model, we found the features that contribute most to a higher house price:

    Area (Square Footage): Contributes positively, as expected.

    Air Conditioning: Having AC was associated with one of the largest positive impacts on price.

    Preferred Area: Houses in preferred locations commanded a significantly higher price. 