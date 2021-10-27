from io import TextIOWrapper
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
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Programming Softwares\New folder\tesseract.exe"
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from PIL import Image

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from plotly import tools


stopwords_nltk = stopwords.words("english")
new_stopwords = ['filtered','including','every','actual','equivalent', 'less','contains','actual',"concentrate","100","artificial","coloring","simple","medium","chain","flavorings","flavor",""]
stopwords_nltk.extend(new_stopwords)





####################################################################################################
#############-----------------------------LOAD DATA----------------------------------###############
####################################################################################################

@st.cache(show_spinner=False)
def load_data():

    # rec_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Data\recommendation_df.csv"
    rec_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Data\rec_df.csv"
    tfidf_df = pd.read_csv(rec_path)
    tfidf_df = tfidf_df.drop(columns=["New Ingredients"])

    df = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\eda.csv")

    veg_df = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\veg_df.csv")
    veg_df = veg_df[~veg_df.Name.duplicated()]

    additives_count = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\additives_count.csv")
    additives_count = additives_count.sort_values("Count")
    
    add_df = pd.read_csv(r"D:\CoderSchool_ML30\FINAL PROJECT\Data\OCR_additives.csv")

    return tfidf_df, veg_df, df, additives_count, add_df



@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    nutri_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Saved_models\best_nutri_rfr_old.pkl"
    nova_path = r"D:\CoderSchool_ML30\FINAL PROJECT\Saved_models\nova_rfc_11.pkl"
    
    with open(nutri_path) as nutri_open:
        nutri_model = joblib.load(nutri_path)
    with open(nova_path) as nova_open:
        nova_model = joblib.load(nova_path)

    return nutri_model, nova_model



####################################################################################################
#############-----------------------------FIRST PAGE---------------------------------###############
####################################################################################################

def about():
 
    st.markdown("<h1 style='text-align: center;'>Eat Better, Not Less </h1>", unsafe_allow_html=True)
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




####################################################################################################
#############-----------------------------SECOND PAGE--------------------------------###############
####################################################################################################

def eda():
    pages = {
        "Vegan vs. Meat": veg,
        "Additives": adds,
        "Nutrients Category-wise": varr}
    
    page = st.radio("(Choose an option to get redirected)", tuple(pages.keys()))
    
    # Display the selected page
    pages[page]()



def veg():
    st.title("Why not VEGAN?")

    st.write("")
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>    The following pie charts show the proportion of the Nutri and Nova gradings for Vegan and Non-Vegan foods available in the dataset </h6>",unsafe_allow_html=True)
    st.write("")
    st.write("")
    
        # vegan nutri
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], horizontal_spacing=0.2)
    values = veg_df[veg_df.Label=="Vegan"]["nutri_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Vegan"]["nutri_grade"].value_counts().index
    colors = ["yellowgreen","lightcoral","gold","firebrick","lightskyblue"]
    fig.add_trace(go.Pie(values=values, labels=labels, marker_colors=colors, textfont_size=14, pull=[0.2,0,0,0,0]), 1,1)
    
        # nonvegan nutri
    values = veg_df[veg_df.Label=="Non Vegan"]["nutri_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Non Vegan"]["nutri_grade"].value_counts().index
    colors = ["lightcoral","gold","firebrick","lightskyblue","yellowgreen",]
    fig.add_trace(go.Pie(values=values, labels=labels, marker_colors=colors, textfont_size=14, pull=[0.2,0,0,0,0]), 1,2)
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textinfo='percent+label')
    fig.update_layout(
        title_text="Does Vegan equal Healthier?",
        font=dict(size=16),
        margin=dict(t=0,b=0,l=0,r=0),
        annotations=[dict(text='Vegan', x=0.18, y=0.5, font_size=14, showarrow=False),
                    dict(text='Non Vegan', x=0.86, y=0.5, font_size=14, showarrow=False)])

    st.plotly_chart(fig)


        # vegan nova
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], horizontal_spacing=0.2)
    values = veg_df[veg_df.Label=="Vegan"]["nova_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Vegan"]["nova_grade"].value_counts().index
    colors = ["firebrick","gold","yellowgreen", "lightskyblue"]
    fig.add_trace(go.Pie(values=values, labels=labels,rotation=90, marker_colors=colors, textfont_size=14, pull=[0.2,0,0,0]), 1,1)
    
        # nonvegan nova
    values = veg_df[veg_df.Label=="Non Vegan"]["nova_grade"].value_counts().values
    labels = veg_df[veg_df.Label=="Non Vegan"]["nova_grade"].value_counts().index
    colors = ["firebrick","gold","yellowgreen"]
    fig.add_trace(go.Pie(values=values, labels=labels,rotation=90, marker_colors=colors, textfont_size=14, pull=[0.2,0,0]), 1,2)
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textinfo='percent+label')
    fig.update_layout(
        title_text="Are Vegan Foods less processed?",
        font=dict(size=16),
        margin=dict(t=0,b=0,l=0,r=0),
        annotations=[dict(text='Vegan', x=0.16, y=0.5, font_size=14, showarrow=False),
                    dict(text='Non Vegan', x=0.86, y=0.5, font_size=14, showarrow=False)])

    st.plotly_chart(fig)


    st.title("Top Foods You Should Avoid")
    st.markdown("")

            ### Cholesterol ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Cholesterol</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Cholesterol (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Dietary Cholesterol has a small effect on blood cholesterol, but people with cardiovascular disease should limit to less than 300mg a day</h6>",unsafe_allow_html=True)
    st.markdown("")

    mask = veg_df[veg_df.Category.isin(["Meat","Seafood"])]
    chol = mask.sort_values("Cholesterol", ascending=False)
    chol_10 = chol.head(10)
    fig = px.bar(chol_10, x="Name", y="Cholesterol", color="Category",
                            hover_data={"Category":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.add_hline(y=0.3, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(title='Top 10 Cholesterol Rich Foods', autosize=False, width=800, height=600,margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)


            ### Saturated Fat ###
    st.markdown(f"<span style='color: #000080;font-size: 24px;font-weight: bold;'>Saturated Fat</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color: #367588;font-size: 12px;font-weight: bold;'>Units: Saturated Fat (grams)</span>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>Saturated fat intake may increase heart disease risk factors. The recommended daily intake of Saturated fat is less than 13 grams per 2000 Calories</h6>",unsafe_allow_html=True)
    st.markdown("")

    mask = veg_df[veg_df.Category.isin(["Seafood", "Meat"])]
    sat = mask.sort_values("Saturated fat", ascending=False)
    sat_10 = sat.head(10)
    fig = px.bar(sat_10, x="Name", y="Saturated fat", color="Category",
                            hover_data={"Category":False, "Name":True,
                                "nutri_grade":True, "nova_grade":True})
    fig.add_hline(y=13, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(title='Top 10 Saturated Fat Rich Foods', autosize=False, width=800, height=600,margin=dict(l=40, r=40, b=40, t=40))
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
    st.markdown("<h6 style='text-align: justify;font-size:100%;font-family:Arial,sans-serif;line-height: 1.3;'>The following charts show the distribution of Nutrients among different categories </h6>",unsafe_allow_html=True)
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
#############-----------------------------THIRD PAGE---------------------------------###############
####################################################################################################

            ############------------------------Ingredients OCR------------------##############

def ocr2string(img):
    ocr_result = pytesseract.image_to_string(img)
    s = ocr_result.lower()
    s = re.sub(r"^.*:","", s)                                     # remove the term Ingredients at the start of the string
    s = re.sub(r"\n"," ", s)                                      # remove any newline character
    s = re.sub(r"(\(|\)|\[|\]|\{|\}|\~|\@|\#|\^|\&|\*)","", s)    # remove special characters
    s = re.sub(r" \x0c","", s)                                    # remove form feed
    s = re.sub(r"  "," ", s)                                      # replace double space by single space
    s = s.strip()                                                 # strip space
    s = s.strip(".")                                              # strip "."
    s = s.strip(",")                                              # strip ","
    return s


def string2additives(string):
    pattern = r"\d{3}[a-i]?"
    match_list = re.findall(pattern, string)
    try:
        mask = np.column_stack([add_df["Number"] == "E"+i for i in match_list])
        return add_df.loc[mask.any(axis=1)].reset_index(drop=True)
    except:
        pass



            ############------------------------Food Grade------------------##############

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



def food_grade():

    apps = ["Additives Detection", "Food Grading"]
    options = st.selectbox("Select application:", apps)

            ##### Additive Check ######
    if options == "Additives Detection":
        st.markdown(f"<div style='color: #2F39CB; text-align: center; font-size: 34px;font-weight: bold;'>Additives Detection</span>", unsafe_allow_html=True)
        st.write("")
        st.write("")        
        img = st.file_uploader("Please upload an image of ingredients", type=['png', 'jpg', 'jpeg'])

        if img:
            my_expander = st.beta_expander("Display Image")
            with my_expander:
                st.image(img)
        
        c1, c2, c3 = st.beta_columns(3)
        ocr = c2.button("Detect Additives")
        if img and ocr:
            # img = img.read()
            # img = tf.image.decode_image(img, channels=3).numpy()
            img = Image.open(img)
            img = np.array(img)
            add_result = string2additives(ocr2string(img))
            add_result.index += 1
            if add_result is None:
                st.warning("No additives can be found in our database. Please try another product")
                
            else:
                st.info("Your food contains these additives:")
                st.table(add_result)


                ###Reference###
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("To view the sources of this data:")
        my_expander=st.beta_expander("Click Here !!")
        with my_expander:
            "[1. Center for Science in the Public Interest](https://www.cspinet.org/eating-healthy/chemical-cuisine#banned)"
            "[2. 12 Common Food Additives](https://www.healthline.com/nutrition/common-food-additives#TOC_TITLE_HDR_7)"





            ###### Food Grading #######
    elif options == "Food Grading":
        st.markdown(f"<div style='color: #2F39CB; text-align: center; font-size: 34px;font-weight: bold;'>Food Grading</span>", unsafe_allow_html=True)
        st.write("")

        st.write("")
        st.write("Please input the nutrition facts")

        st1, st2 = st.beta_columns(2)
        st3, st4 = st.beta_columns(2)
        st5, st6 = st.beta_columns(2)
        st7, st8 = st.beta_columns(2)
        st9, st10 = st.beta_columns(2)
        st11, st12 = st.beta_columns(2)

        size = st1.number_input("Serving size", key="size",min_value=10, step=1, value=240, max_value=400)
        ene = st2.number_input("Energy (kcal)", value=120,min_value=0, max_value=500, key= "ene")/size*100
        fat = st3.number_input("Fat (g)", format="%.1f",value=5.0,min_value=0.0, max_value=100.0,step=0.01, key="fat")/size*100
        sat = st6.number_input("Saturated Fat (g)",value=0.5,min_value=0.0, max_value=10.0,step=0.01, format="%.2f", key="sat")/size*100
        trans = st5.number_input("Trans fat (g)",value=0.0,min_value=0.0, max_value=10.0,step=0.01, format="%.2f", key="trans")/size*100
        chol = st11.number_input("Cholesterol (g)",value=0.0, min_value=0.0, max_value=1.0,step=0.0001, format="%.4f", key="chol")/size*100
        sodium = st9.number_input("Sodium (g) - (1 mg = 0.001 g)",value=0.1, min_value=0.0, max_value=1.0,step=0.0001, format="%.4f", key="sodium")/size*100
        carb = st7.number_input("Carbohydrate (g)",value=16.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="carb")/size*100
        fiber = st4.number_input("Fiber (g)",value=3.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="fiber")/size*100
        pro = st8.number_input("Protein (g)",value=3.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="pro")/size*100
        sug = st10.number_input("Sugar (g)",value=7.0, min_value=0.0, max_value=100.0,step=0.01, format="%.1f", key="sug")/size*100
        add = st12.number_input("Additives (g)",value=4, min_value=0, max_value=50, key="add")/size*100
        
        # oats = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\Oats_Grading.JPG")
        oatmilk = Image.open(r"D:\CoderSchool_ML30\FINAL PROJECT\imgs\Oatmilk_Grading.JPG")
        my_expander=st.beta_expander("For Demo only:")
        with my_expander:
            st.image(oatmilk)


        c1, c2, c3, c4, c5, c6, c7 = st.beta_columns(7)
        predict = c4.button("Grade")
        
        if predict:

            nutrients = [chol, fat, fiber, trans, sat, carb, pro, sodium, sug, ene, add]
            nutrient_df = pd.DataFrame(columns=["Cholesterol","Fat", "Fiber", "Trans fat", "Saturated fat", "Carbohydrates", "Protein", "Sodium", "Sugars", "Energy_kcal", "Additives"])
            nutrient_df.loc[0] = nutrients
            nutri_model, nova_model = load_model()
            
                # nutri_score
            nutri_score_pred = nutri_model.predict(nutrient_df)
            nutri_grade = nutri_grade_convert(nutri_score_pred)
                # nova_grade
            nova_grade = nova_model.predict(nutrient_df)


            st.write("Your Food Nutrition level is:", nutri_grade)
            st.write("Your Food Processing level is:", nova_grade[0])

            if nutri_grade in ["a", "b"]:
                if nova_grade == 1:
                    st.success("You should have this more often! It's both nutritious and minimally processed")
                    st.balloons()
                elif nova_grade in [2,3,4]:
                    st.warning("Consuming a lot of processed foods might increase risk of cardiovascular diseases")

            if nutri_grade in ["c", "d", "e"] :
                if nova_grade in [3,4]:
                    st.warning("This food is good for you. Maybe you should find something healthier!")
                elif nova_grade in [1,2]:
                    st.info("This might not contain a lot of nutrition but it's not harmful to consume")

        




            ####################--------------------FOOD-BASED---------------------######################

@st.cache(allow_output_mutation=True, show_spinner=False)
def food_based_recommender(name):
    """Function to get recommendation based on selected product"""

    rec_df = tfidf_df.drop(columns=["Combination"])

        #Tfidf Vectorization
    tfidf_vect = TfidfVectorizer(min_df=2, stop_words=stopwords_nltk)
    tfidf_matrix = tfidf_vect.fit_transform(tfidf_df["Combination"].values)

        #Compute similarities
    preference = tfidf_df[tfidf_df.Name == name]["Combination"].values
    preference_transformed = tfidf_vect.transform(preference)
    cs = cosine_similarity(tfidf_matrix, preference_transformed)

    sim_idx = list(cs.flatten().argsort()[::-1])[:10]
    result = rec_df.iloc[sim_idx].sort_values(["nutri_grade","nova_grade"])
    result = result[result.Name != name]
    return result




            ##########---------------------CRITERIA-BASED----------------------------###############

def compute_similarities(ing, df):
    """Function to get similar products from user preferences"""

        # Create a tf-idf matrix
    vectorizer = TfidfVectorizer(min_df=2, stop_words=stopwords_nltk)
    tfidf_matrix = vectorizer.fit_transform(df["Combination"])

        # User side
    user_transformed = vectorizer.transform([ing])

        # Compute similarities and get top k most similar items
    cs = cosine_similarity(tfidf_matrix, user_transformed)
    sim_idx = list(cs.flatten().argsort()[::-1])[:10]

    return df.iloc[sim_idx].drop(columns=["Combination"])



def recommendation():
    global tfidf_df

    apps = ["Personalized", "Food-Based"]
    options = st.selectbox("Select application:", apps)

            ### Food-Based Recommendation ###
    if options == "Food-Based":
        prods = tfidf_df.Name
        food_select = st.selectbox("Select product:", prods)

        result = food_based_recommender(food_select)

        c1, c2, c3 = st.beta_columns(3)
        selected = c2.button("Recommend")
        if selected:
            with st.spinner("Getting Recommendations..."):
                st.table(result)


            ### Personalized Recommendation ###
    if options == "Personalized":
        ingre_select = st.text_input("Your Favorite ingredients:")

        labels = ["---Select---","Vegan","Non Vegan", "Organic"]
        label_select = st.selectbox("Select Label:", labels)

        categories = ['---Select---','Snacks', 'Meals', 'Plant-based-foods', 'Cereals', 'Milk', 'Pastas', ' Desserts',
        'Plant-based beverages', 'Fruits', 'Grains', 'Dairy', 'Vegetables', 'Legumes', 'Seafood', 'Meat', 'Noodles']

        cat_select = st.selectbox("Select Category:", categories)

        if label_select == "---Select---":
            label_select = ""
        if cat_select == "---Select---":
            cat_select = ""

        if ingre_select:
            user_input = ingre_select
            
            df = tfidf_df
            if label_select and cat_select:
                df = tfidf_df[(tfidf_df.Category == cat_select) & (tfidf_df.Label == label_select)]

            elif label_select:
                df = tfidf_df[tfidf_df.Label == label_select]
                
            elif cat_select:
                df = tfidf_df[tfidf_df.Category == cat_select]


                        ### Recommend ###
            c1, c2, c3 = st.beta_columns(3)
            selected = c2.button("Recommend")
            if selected:
                try:
                        ### Create a tf-idf matrix ###
                    vectorizer = TfidfVectorizer(min_df=2, stop_words=stopwords_nltk)
                    tfidf_matrix = vectorizer.fit_transform(df["Combination"])

                        # User side
                    user_transformed = vectorizer.transform([user_input])

                        # Compute similarities and get top k most similar items
                    cs = cosine_similarity(tfidf_matrix, user_transformed)
                    sim_idx = list(cs.flatten().argsort()[::-1])[:8]

                    recommendations = df.drop(columns=["Combination"]).iloc[sim_idx].sort_values(["nutri_grade", "nova_grade"]).reset_index(drop=True)
                    recommendations.index += 1
                    st.table(recommendations)

                except ValueError:
                    st.warning("Sorry! We can't recommend any product that match your preferences. Please try again.")






####################################################################################################
#############-----------------------------MAIN PAGE---------------------------------###############
####################################################################################################

def main():
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
    tfidf_df, veg_df, df, additives_count, add_df = load_data()
    main()









