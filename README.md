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
  - Recommendation System
5. Fine-tuning
6. Evalutation


### 1. Questions 
- NOVA-grade (processing level of food) - Common ingredients/food that are classified as most processed

- Nutrition-grade - Is the food really nutritious as its label mentions?

- How to choose similar products to my favorite one?

### 2. Data Pre-processing
1. Filtering unescessary columns and data
2. Dealing with missing data
  - For Recommendation system: data with missing Nutri-score and Nova-grade targets are filled in using previously built models to increase the number of items for Recommendation
  - For Classification and Regresison tasks, products with missing values are removed
3. Outliers Treatment
  - Any numerical values whose values are outside of quantile 0.99 are removed for Classification & Regression
  - However, for Data Analysis, these outliers are retained to gather insights
4. Regular Expression & String Manipulation
  - For Recommendation system, the products need to be grouped into Categories and Labels. In addition, before feeding into TfidfVectorizer, the categorical features need to undergo some Regular expression and string manipulation to clean the special characters, replace comma with space, and finally joining multiple words in the same ingredient by a hyphen. After transformation, each product's ingredients collection is represented as a string of multiple ingredients.

### 3. Explorative Data Analysis

![Nutri-grade](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/nutri.png)

![Nova-grade](https://raw.githubusercontent.com/thuminhle59/Novous_EatBetterNotLess/main/imgs/nova.png)

## Conclusion



