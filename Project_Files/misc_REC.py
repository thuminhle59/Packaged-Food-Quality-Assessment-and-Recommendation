from numpy.lib.function_base import vectorize
from numpy.lib.npyio import load
import streamlit as st
import pandas as pd, seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# from wordcloud import WordCloud
from PIL import Image

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from plotly import tools


stopwords_nltk = stopwords.words("english")
new_stopwords = ['filtered','including','every','actual','equivalent', 'less','contains','actual',"concentrate","100","artificial","coloring","simple","medium","chain","flavorings","flavor",""]
stopwords_nltk.extend(new_stopwords)





######--------------------LOAD DATA------------------######

@st.cache(show_spinner=False)
def load_data():

    rec_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Data\recommendation_df.csv"
    tfidf_df = pd.read_csv(rec_path)

    df = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\eda (1).csv")

    veg_df = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\veg_df (2).csv")
    veg_df = veg_df[~veg_df.Name.duplicated()]

    additives_count = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\additives_count.csv")
    additives_count = additives_count.sort_values("Count")

    df_nova = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Saved_files\df_nova.csv")
    
    return tfidf_df, veg_df, df, additives_count, df_nova



@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    nutri_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Saved_models\best_nutri_rfr_old1.pkl"
    nova_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Saved_models\nova_rfc_11.pkl"
    
    with open(nutri_path) as nutri_open:
        nutri_model = joblib.load(nutri_path)
    with open(nova_path) as nova_open:
        nova_model = joblib.load(nova_path)

    return nutri_model, nova_model

############------------------------------ FIRST PAGE --------------------------------------#################

def about():
 
    st.markdown("<h1 style='text-align: center;'>Eat Better, Not Less </h1>", unsafe_allow_html=True)
    # h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    nutri_explain = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\nutri_explain.png")
    nutri_scale = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\nutri-scale.png")
    nova_explain = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\Nova-explain.png")

    st.markdown(f"<span style='color: #367588;font-size: 24px;font-weight: bold;'>Nutri-Score System</span>", unsafe_allow_html=True)  
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>The Nutri-Score is five-color logo with corresponding letters (A to E) for assessing the overall nutritional quality of food items. Sometimes affixed to the front of food packages, it helps consumers to choose products of better nutritional quality</h6>",unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.image(nutri_explain)
    st.write("")
    st.markdown(f"<span style='color: #367588;font-size: 18px;font-weight: bold;'>Nutri-Score Scale</span>", unsafe_allow_html=True)
    st.image(nutri_scale)

    st.write("")
    st.write("")
    st.write("")
    st.markdown(f"<span style='color: #367588;font-size: 24px;font-weight: bold;'>NOVA-Grade System</span>", unsafe_allow_html=True)  
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>The NOVA, developed by researchers at the University of Sāo Paulo in Brazil, assigns foodstuffs to four groups according to the extent and purpose of industrial food processing.</h6>",unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.image(nova_explain, width=400)

    st.caption("Epidemiological studies have demonstrated a correlation between the consumption of highly-processed foods and an increased risk of cancer, obesity and other diseases")




##############------------------------------ SECOND PAGE -------------------------------------#############

def eda():


    pages = {
        "Vegan vs. Meat": veg,
        "Additives": adds,
        "Nutrients Category-wise": varr,
        
    }
    # Widget to select your page, you can choose between radio buttons or a selectbox
    page = st.radio("(Choose an option to get redirected)", tuple(pages.keys()))
    
    # Display the selected page
    pages[page]()



def veg():
    st.title("Why not VEGAN?")

    st.markdown("")
    
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>    The following pie charts show the proportion of the Nutri and Nova gradings for Vegan and Non-Vegan foods available in the dataset </h6>",unsafe_allow_html=True)
    st.write("")
    # st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'> Does Vegan equal Healthier?</span>", unsafe_allow_html=True)
    # st.markdown("")

        # vegan nutri
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    values = veg_df[veg_df.Label=="Vegan"]["nutri_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Vegan"]["nutri_grade"].value_counts().index
    colors = ["yellowgreen","gold","lightcoral","lightskyblue","firebrick"]
    fig.add_trace(go.Pie(values=values, labels=labels, marker_colors=colors, textfont_size=14, pull=[0,0,0,0,0.2]), 1,1)
    
        # nonvegan nutri
    values = veg_df[veg_df.Label=="Non Vegan"]["nutri_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Non Vegan"]["nutri_grade"].value_counts().index
    colors = ["lightcoral","lightskyblue","yellowgreen","gold","firebrick"]
    fig.add_trace(go.Pie(values=values, labels=labels, marker_colors=colors, textfont_size=14, pull=[0,0,0,0,0.2]), 1,2)
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textinfo='percent+label')
    fig.update_layout(
        title_text="Does Vegan equal Healthier?",
        margin=dict(t=1,b=1,l=1,r=1),
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Vegan', x=0.18, y=0.5, font_size=14, showarrow=False),
                    dict(text='Non Vegan', x=0.84, y=0.5, font_size=14, showarrow=False)])

    st.plotly_chart(fig)


        # vegan nova
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    values = veg_df[veg_df.Label=="Vegan"]["nova_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Vegan"]["nova_grade"].value_counts().index
    colors = ["firebrick","gold","yellowgreen"]
    fig.add_trace(go.Pie(values=values, labels=labels,rotation=90, marker_colors=colors, textfont_size=12, pull=[0,0,0,0.5]), 1,1)
    
        # nonvegan nova
    values = veg_df[veg_df.Label=="Non Vegan"]["nova_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Non Vegan"]["nova_grade"].value_counts().index
    colors = ["firebrick","gold","yellowgreen"]
    fig.add_trace(go.Pie(values=values, labels=labels,rotation=90, marker_colors=colors, textfont_size=12, pull=[0,0,0,0.5]), 1,2)
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textinfo='percent+label')
    fig.update_layout(
        title_text="Are Vegan Foods less processed?",
        
        margin=dict(t=0,b=0,l=0,r=0),
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Vegan', x=0.18, y=0.5, font_size=14, showarrow=False),
                    dict(text='Non Vegan', x=0.84, y=0.5, font_size=14, showarrow=False)])

    st.plotly_chart(fig)

    st.title("Top Foods You Should Avoid")
    # st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Certain foods have higher proportions of specific nutrients than others. The most notable example is that meat, fish, and eggs contain higher amounts of protein than plant-based foods. Here, the data is analyzed to find the foods with the highest amounts of various nutrients, including proteins, carbohydrates, and fats. This is explained with the help of graphical representations.</h6>",unsafe_allow_html=True)
    st.markdown("")

        ### Saturated Fat ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Saturated Fat</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Saturated Fat (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Saturated fat intake may increase heart disease risk factors. The recommended daily intake of Saturated fat is less than 13 grams per 2000 Calories</h6>",unsafe_allow_html=True)
    st.markdown("")

    sat = veg_df.sort_values("Saturated fat", ascending=False)
    sat_10 = sat.head(10)
    fig = px.bar(sat_10, x="Name", y="Saturated fat", color="Label",
                            hover_data={"Label":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.update_layout(title='Top 10 Saturated Fat Rich Foods', autosize=False, width=800, height=600,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)

            ### Sodium ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Sodium</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Sodium (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Excess sodium increases blood pressure and creates an added burden on the heart. The recommended amount for a healthy person is 2,3g a day. For people with high blood pressure or diabetes, or age 51 or older, the daily recommendation is 1,5g of sodium </h6>",unsafe_allow_html=True)
    st.markdown("")

    sodium = veg_df.sort_values("Sodium", ascending=False)
    sodium_10 = sodium.head(10)
    fig = px.bar(sodium_10, x="Name", y="Sodium", color="Label",
                    hover_data={"Label":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.update_layout(title='Top 10 Sodium Rich Foods', autosize=False, width=800, height=700,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)

            ### Cholesterol ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Cholesterol</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Cholesterol (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Dietary Cholesterol has a small effect on blood cholesterol, but people with cardiovascular disease should limit to less than 300mg a day</h6>",unsafe_allow_html=True)
    st.markdown("")

    chol = veg_df.sort_values("Cholesterol", ascending=False)
    chol_10 = chol.head(10)
    fig = px.bar(chol_10, x="Name", y="Cholesterol", color="Label",
                            hover_data={"Label":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.update_layout(title='Top 10 Cholesterol Rich Foods', autosize=False, width=800, height=600,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)


            ### GOOD FOODS ###
    st.title("Better Foods for Your Health")
    st.markdown("")

        ### Fiber ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Fiber</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Fiber (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'> Daily recommended dietary Fiber intake should be 25-30g a day</h6>",unsafe_allow_html=True)
    st.markdown("")

    fiber = veg_df.sort_values("Fiber", ascending=False)
    fiber_10 = fiber.head(10)
    fig = px.bar(fiber_10, x="Name", y="Fiber", color="Label",
                            hover_data={"Label":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.update_layout(title='Top 10 Fiber Rich Foods', autosize=False, width=800, height=600,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)

             ###Reference###
    st.write("To view the sources of this data:")
    my_expander=st.beta_expander("Click Here !!")
    with my_expander:
        "[1. Foods that contain cholesterol](https://www.healthline.com/nutrition/high-cholesterol-foods#foods-to-eat)"
        "[2. Calories: Requirements](https://www.medicalnewstoday.com/articles/263028)"

    st.write("")
    vegan = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\vegan.png")
    st.image(vegan)




def varr():
    
    st.markdown("")
    
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>    The following charts show the distribution of Nutrients among different categories </h6>",unsafe_allow_html=True)
    st.write("")
    sodium_var = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\sod_var.png")
    chol_var = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\chol_var.png")
    sat_var = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\sat_var.png")
    sug_var = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\sug_var.png")
    fiber_var = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\fib_var.png")
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Cholesterol</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: grams</span>", unsafe_allow_html=True)
    st.image(chol_var, width=900)
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Sodium</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: grams</span>", unsafe_allow_html=True)
    st.caption("")
    st.image(sodium_var, width=900)
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Saturated Fat</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: grams</span>", unsafe_allow_html=True)
    st.image(sug_var, width=900)
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Fiber</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: grams</span>", unsafe_allow_html=True)
    st.image(fiber_var, width=900)



def adds():

                #### DISPLAY ###
    st.markdown("")
    st.title("Additives")

                ### Additives Category-wise###
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'> Distribution of Additives among Categories :</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Only food additives that have been deemed safe by JECFA can be used in foods that are traded internationally</h6>",unsafe_allow_html=True)
    st.write("")

    fig = plt.figure(figsize=(22,8))
    ax = sns.boxenplot(x="Category", y='Additives_count', data=df, color='#eeeeee', palette="tab10")

    # Add transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .9))

    plt.ylim(bottom=0, top=40)
    plt.show()
    st.pyplot(fig)

        #### Top 30 additives
    st.write("")
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'> Top 30 Most commonly used Additives :</span>", unsafe_allow_html=True)
    # st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>    </h6>",unsafe_allow_html=True)
    # st.write("")

    fig = px.bar(additives_count, x="Count", y="Enum", color="Type").update_xaxes(categoryorder="trace")
    fig.update_layout(autosize=False,width=600, height=600,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)

    adds_exp = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\Additives_explain - Sheet1.csv")
    with st.beta_expander("See explanation:"):
        st.table(adds_exp)

    st.write("")

         ###Reference###
    st.write("")
    st.write("")
    st.write("")
    st.write("To view the sources of this data:")
    my_expander=st.beta_expander("Click Here !!")
    with my_expander:
        "[1. Foods Additives](https://www.who.int/news-room/fact-sheets/detail/food-additives)"
        "[2. Foods Additives and E Numbers](https://dermnetnz.org/topics/food-additives-and-e-numbers)"



####################################################################################################
#############-----------------------------FOOD GRADE-------------------------------###############
####################################################################################################


def nutri_grade_convert(nutri_score):
    """Function to convert Nutri-score"""

    if nutri_score >= -15 and nutri_score <= -1:
        nutri_grade = "a"
    elif nutri_score >=0 and nutri_score <=2:
        nutri_grade = "b"
    elif nutri_score >=3 and nutri_score <= 10:
        nutri_grade = "c"
    elif nutri_score >=11 and nutri_score <=18:
        nutri_grade = "d"
    else:
        nutri_grade = "e"

    return nutri_grade


########################################################################################################
##########------------------------------FOOD-BASED--------------------------------------###############
####################################################################################################


@st.cache(allow_output_mutation=True, show_spinner=False)
def food_based_recommender(name):
    """Function to get recommendation based on selected product"""

        #Tfidf Vectorization
    # tfidf_vect = TfidfVectorizer(min_df=2, stop_words=stopwords_nltk)
    tfidf_vect = TfidfVectorizer(min_df=2)
    tfidf_matrix = tfidf_vect.fit_transform(tfidf_df["Combination"].values)

        #Compute similarities
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(tfidf_df.index, index=tfidf_df['Name'])

        #Get list of 10 similar indices
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    ndx = [i[0] for i in sim_scores]
    return tfidf_df["Name"].iloc[ndx]






########################################################################################################
##########------------------------------CRITERIA-BASED--------------------------------------###############
#########################################################################################################

                #######TF-IDF##### 
# @st.cache()
def compute_similarities(ing):
    """Function to get similar products from user preferences"""

    # Create a tf-idf matrix
    vectorizer = TfidfVectorizer(min_df=2, stop_words=stopwords_nltk)
    tfidf_matrix = vectorizer.fit_transform(tfidf_df["Combination"])

    # User side
    user_transformed = vectorizer.transform(ing)

    # Compute similarities and get top k most similar items
    cs = cosine_similarity(tfidf_matrix, user_transformed)
    sim_idx = list(cs.flatten().argsort()[::-1])[:10]

    return tfidf_df.iloc[sim_idx].drop(columns=["Combination"])






    



############--------------------------------Third page--------------------------------------##############
def food_grade():

    
    # oatmilk = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\Oatmilk_Grading.JPG")
    oats = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\Oatmilk_Grading.JPG")
    my_expander=st.beta_expander("For Demo only:")
    with my_expander:
        st.image(oats)

    st.write("")
    st.write("Please input the nutrition facts")

    st1, st2 = st.beta_columns(2)
    st3, st4 = st.beta_columns(2)
    st5, st6 = st.beta_columns(2)
    st7, st8 = st.beta_columns(2)
    st9, st10 = st.beta_columns(2)
    st11, st12 = st.beta_columns(2)

    size = st1.slider("Serving size", key="size",min_value=10, step=1, value=120, max_value=400)
    ene = st2.slider("Energy (kcal)", value=100,min_value=0, max_value=500, key= "ene")/size*100
    fat = st3.slider("Fat (g)", format="%.1f",value=5.0,min_value=0.0, max_value=10.0,step=0.01, key="fat")/size*100
    sat = st6.slider("Saturated Fat (g)",value=0.5,min_value=0.0, max_value=10.0,step=0.01, format="%.2f", key="sat")/size*100
    trans = st5.slider("Trans fat (g)",value=0.0,min_value=0.0, max_value=10.0,step=0.01, format="%.2f", key="trans")/size*100
    chol = st11.slider("Cholesterol (g)",value=0.0, min_value=0.0, max_value=1.0,step=0.0001, format="%.4f", key="chol")/size*100
    sodium = st9.slider("Sodium (g)",value=0.1, min_value=0.0, max_value=1.0,step=0.0001, format="%.4f", key="sodium")/size*100
    carb = st7.slider("Carbohydrate (g)",value=16.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="carb")/size*100
    fiber = st4.slider("Fiber (g)",value=3.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="fiber")/size*100
    pro = st8.slider("Protein (g)",value=3.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="pro")/size*100
    sug = st10.slider("Sugar (g)",value=7.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="sug")/size*100
    add = st12.slider("Additives (g)",value=3, min_value=0, max_value=50, key="add")/size*100
    
    nutrients = [chol, fat, fiber, trans, sat, carb, pro, sodium, sug, ene, add]
    nutrient_df = pd.DataFrame(columns=["Cholesterol","Fat", "Fiber", "Trans fat", "Saturated fat", "Carbohydrates", "Protein", "Sodium", "Sugars", "Energy_kcal", "Additives"])
    nutrient_df.loc[0] = nutrients
    nutri_model, nova_model = load_model()
      
    # nutri_score
    nutri_score_pred = nutri_model.predict(nutrient_df)
    nutri_grade = nutri_grade_convert(nutri_score_pred)
    # nova_grade
    nova_grade = nova_model.predict(nutrient_df)

    predict = st.button("Grade")
    if predict:
        # st.write("Calculating scores.")
        st.write("Your Food Nutrition level is:", nutri_grade)
        st.write("Your Food Processing level is:", nova_grade[0])
        pass

    














##################################------------------------Fourth page-----------------------------##########################
# @st.cache(allow_output_mutation=True)

def recommendation():
    global tfidf_df

    ingre_select = st.text_input("Your Favorite ingredients:")

    labels = ["organic","vegan","non_vegan"]
    label_select = st.multiselect("Select Label:", labels)
    label_select = " ".join([i for i in label_select])

    categories = ['snacks', 'meals', 'plant-based-foods', 'cereals', 'milk', 'pastas',
    'plant-based beverages', 'fruits', 'grains', 'dairy', 'vegetables', 'legumes', 'seafood', 'meat']

    cat_select = st.multiselect("Select Category:", categories)
    cat_select = " ".join([i for i in cat_select])    

    if ingre_select:
        user_input = ingre_select
        if label_select:
            user_input = label_select + " " + user_input
            
        if cat_select:
            user_input = cat_select + " " + user_input

        recommendations = compute_similarities([user_input])
        st.table(recommendations.sort_values(["nutri_grade", "nova_grade"]))

    else:
        if cat_select and label_select:
            user_input = cat_select + " " + label_select
            recommendations = compute_similarities([user_input])
            st.table(recommendations.sort_values("nutri_grade"))





############---------------------------------MAIN PAGE---------------------------------#############
def main():
    # st.header("MinT :D")

    pages = {
        "About": about,
        "Nutition Information": eda,
        "Food grade": food_grade,
        "Recommendation": recommendation
        }

    st.sidebar.title("Welcome to Novous")
    page = st.sidebar.radio("Choose an option to be redirected to", tuple(pages.keys()))

    # Display the selected page
    pages[page]()






    

if __name__ == "__main__":
    tfidf_df, veg_df, df, additives_count, df_nova = load_data()
    main()









