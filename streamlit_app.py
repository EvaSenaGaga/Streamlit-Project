import streamlit as st
import pandas as pd 
import numpy as np
import os, pickle


# Application title
st.title('STORE SALES PREDICTION')

#st.set_page_config(layout= 'centered')
#App description
st.write('This app is made to predict store sales in Corporation favorita store use case.')

st.sidebar.subheader('Corporation Favorita')
st.checkbox('checkbox', value= True)

# Function to import the Machine Learning toolkit
@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object

#loading dataframe
@st.cache
def load_data(relative_path):
    train_data = pd.read_csv(relative_path, index_col=0)
    train_data['year'] = pd.to_datetime(train_data['date']).dt.year
    train_data['date'] = pd.to_datetime(train_data['date']).dt.date
    train_data.rename(columns={'type_x': 'store_type'}, inplace = True)
    return train_data

    # Function to get date features from the inputs
@st.cache()
def getDateFeatures(df, date):
    df["date"] = pd.to_datetime(df[date])
    df["day_of_week"] = df["date"].dt.dayofweek.astype(int)
    df["day_of_month"] = df["date"].dt.day.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)
    df["is_weekend"] = np.where(df["day_of_week"] > 4, 1, 0).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    df = df.drop(columns = "date")
    return df

relative_path = r"C:\Users\selas\OneDrive\Desktop\Streamlit_and_Gradio_project\Streamlit-and-Gradio-Project\train_data.csv"
train_data = load_data(relative_path)
st.write(train_data.head())
st.markdown('')

# Loading the toolkit
loaded_toolkit = load_ml_toolkit(r"C:\Users\selas\OneDrive\Desktop\Streamlit_and_Gradio_project\Streamlit-and-Gradio-Project\ML_items")
if "results" not in st.session_state:
    st.session_state["results"] = []

# Instantiating the elements of the Machine Learning Toolkit
#nn_scaler = loaded_toolkit["scaler"]
dt_model = loaded_toolkit["model"]
le_encoder = loaded_toolkit["encoder"]

# Desig of sidebar
st.sidebar.header("Information on Columns")
st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **family** identifies the type of product sold.
                    - **sales** is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **sales_date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **cluster** is a grouping of similar stores.
                    - **oil_price** is the daily oil price
                    - **holiday_type** indicates whether the day was a holiday, event day, or a workday
                    - **locale** indicates whether the holiday was local, national or regional.
                    - **transferred** indicates whether the day was a transferred holiday or not.
                    """)

#input and output section
left_col, right_col = st.columns(2)
left_col.subheader('Inputs')
left_col.write('Taking Inputs for prediction')


dataset = st.container()
features_and_output = st.container()

#form to take inputs from the user.
form = st.form(key ='information', clear_on_submit=True)

# Structuring the dataset section
with dataset:
    if dataset.checkbox("Preview the dataset"):
        dataset.write(train_data.head())
        dataset.write("For more information on the columns, Please look at the sidebar")
    dataset.write("---")

    #features to encode
categorical_features = ["family", "city", "state", "store_type", "holiday_type", "locale"]

# Defining the list of expected variables
expected_inputs = ["sales_date",  "family",  "store_nbr",  "store_type",  "city",  "state",  "onpromotion",   "holiday_type",  "locale",  "transferred"]

#boxes for forms
with form:
    col1, col2 = st.columns(2)
    sales_date = col1.date_input('Select a date', min_value= train_data['date'].min())
    store_nbr = st.selectbox('Store Number', options=(train_data['store_nbr']))
    store_type = st.radio('Store Type', ('A','B','C','D','E'))
    family = st.selectbox('Product Family', options=(train_data['family']))
    state = st.selectbox('State', options=(train_data['state']))
    city = st.selectbox('City', options=(train_data['city']))
    onpromotion = st.selectbox('Items on Promotion', options=(train_data['onpromotion']))

    if st.checkbox('is it a holiday?'):
        holiday_type = st.selectbox('Holiday type',options= set(train_data['holiday_type']))
        locale = st.selectbox('locale',options= set(train_data['locale']))
        transferred = st.radio('is the holiday transferred?',options= set(train_data['transferred']))

    else:
        holiday_type = 'Work Day'
        locale = 'National'
        transferred = 'False'

#Submit button
    submitted = st.form_submit_button(label="Submit")


    if submitted:
       with features_and_output:
        st.success("Thanks!")
        # Inputs formatting
        input_dict = {
            "sales_date": [sales_date],
            "family": [family],
            "store_nbr": [store_nbr],
            "store_type": [store_type],
            "city": [city],
            "state": [state],
            "onpromotion": [onpromotion],
            "holiday_type": [holiday_type],
            "locale": [locale],
        }

        # Converting the input into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_dataframe = input_data.copy()
        
        # Converting data types into required types
        input_data["sales_date"] = pd.to_datetime(input_data["sales_date"]).dt.date
        
        
        # Getting date features
        dataframe_processed = getDateFeatures(input_data, "sales_date")
        dataframe_processed.drop(columns=["sales_date"], inplace= True)

        # Encoding the categoricals
        encoded_categoricals = le_encoder.fit_transform(input_data[categorical_features])
        encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = le_encoder.get_feature_names_out().tolist())
        dataframe_processed = dataframe_processed.join(encoded_categoricals)
        dataframe_processed.drop(columns=categorical_features, inplace=True)

    
    
        # Making the predictions        
        dt_pred = dt_model.predict(dataframe_processed)
        dataframe_processed["sales"] = dt_pred
        input_dataframe["sales"] = dt_pred
        display = dt_pred[0]

        # Adding the predictions to previous predictions
        st.session_state["results"].append(input_dataframe)
        result = pd.concat(st.session_state["results"])


    # Displaying prediction results
    #st.success(f"**Predicted sales**: USD {display}")
    st.write('predicted sales in USD:', display)

    

 

