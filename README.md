# NOVOUS EatBetterNotLess


Link to Presentation: https://docs.google.com/presentation/d/18U59q65M4gWwAdbxB0iUPPr1RTWVEJv36mlMCbqDrYg/edit?usp=sharing

## Dataset
The dataset used for Novous originates from OpenFoodFacts https://world.openfoodfacts.org, a free and online database of nearly 2 million food products around the world. This data contains metadata of packaged foods such as Name, Brand, Product Image, Origin, Nutrition facts, Ingredients, Nutriscore, Nova-grade, etc.

For the purpose of this project, only packaged foods from United States were chosen to analyze their ingredients in English, and the number of items were around 190,000 products, along with 12 numerical and categorical features including

## Project Description


## Outline
1. Defining Questions
2. Data Selection & Processing
3. Explorative Data Analytics
4. Developing Models
    - Classification & Regression Models
    - OCR for Additives Detection
    - Recommendation System
5. Fine-tuning
6. Evalutation


### I. Questions 
- NOVA-grade (processing level of food) - Common ingredients/food that are classified as most processed

- Nutrition-grade - Is the food really nutritious as its label mentions?

- How to choose similar products to my favorite one?

### II. Data Pre-processing
1. Filtering unescessary columns and data
2. Dealing with missing data
    - For Recommendation system: data with missing Nutri-score and Nova-grade targets are filled in using previously built models to increase the number of items for Recommendation
    - For Classification and Regresison tasks, products with missing values are removed
3. Outliers Treatment
    - Any numerical values whose values are outside of quantile 0.99 are removed for Classification & Regression
    - However, for Data Analysis, these outliers are retained to gather insights
4. Regular Expression & String Manipulation
    - For Recommendation system, the products need to be grouped into Categories and Labels. In addition, before feeding into TfidfVectorizer, the categorical features need to undergo some Regular expression and string manipulation to clean the special characters, replace comma with space, and finally joining multiple words in the same ingredient by a hyphen. After transformation, each product's ingredients collection is represented as a string of multiple ingredients.

Sample result: ![regex](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/regex.png)

### III. Explorative Data Analysis

1. Overview of Packaged Food Quality

![Nutri-grade](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/nutri.png)

**About 70% of packaged foods from United States are heavily processed NOVA 4 This number is very consistent with most published reports of fast food in United States. Only a very smaller portion of food is unprocessed or minimally procssed.**

![Nova-grade](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/nova.png)

**Regarding Nutri-grade, nearly 50% of foods are given grade of D or E, this means most packaged food products do not bring many health benefits to consumers at all.**

2. Vegan vs. Non-Vegan 

**Does Vegan equal Healither?**

![veg_Nutri](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/veg_nutri.png)

**Are Vegan Foods less processed?**

![veg_Nova](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/vegnova.png)

**The proportions of Bad Foods (foods with low nutritional quality and high level of processing) are mainly Non-Vegan foods, with about 70% of those fall into these criterias. So why not consider going Vegan?**

3. Additives

![Additives](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/adds.png)

![Adds_catwise](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/add_cat.png)

**Most packaged foods contain at least 1 additives added for coloring, preservative, or emulsifiers. In 1 package of Dessert, there could be up to 35 additives which definitely do more harm than good to our health.**


### IV. Developing Models

1. Random Forest for NOVA-grade Classification and Nutri-score Regression

    a. Classification
Benchmark Result: 

![Class](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/classification.png)
    b. Regression
Benchmark Result:

![Regress](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/regression.png)

Random Forest was chosen as the Final model for both Classification and Regresison tasks.

2. OCR for Additives Detection

When an image of ingredients is passed through Novous, with the help of Pytesseract, it can detect the additives in E-numbers and provide the information of side effects and usages

3. Recommendation System with Content-based Filtering

Tfidf Vectorizer is used to vectorize the categorical features into numbers. Each product will be a vector with each entry is a unique ingredient of that product. Tfidf highlights the important and unique ingredients and making sure that unimportant ones (water, sugars, etc..) do not dominate

### V. Fine-Tuning

Roc Areas for each NOVA-grade class

![RocAUC](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/roc_auc.png)


|-----|Random Forest Classifier|Random Forest Regressor|
|Roc Auc| 0.94228 |         |
|MAE    |         | 0.49874 |





## Conclusion



