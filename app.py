import streamlit as st
import pandas as pd
import pickle

def load_model(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

scaler = load_model("scaler.pkl")
pca = load_model("pca.pkl")
kmeans = load_model("kmeans.pkl")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ['Dashboard', 'About Us', 'Meet the Team'])

if page == "Dashboard":
    st.title("SmartSpent Deployment")

    # upload the file
    data = st.file_uploader("Upload you transaction data", type=['csv'])

    if data is not None:
        try:
            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(data)
            df = df.dropna().reset_index(drop=True)
            st.write("Top 5 rows")
            st.write(df.head())
            

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")




    # get the customer info and the preprocess it
    start = st.text_input("Starting point")
    
    if start:
        start = int(start)
    end = st.text_input("Ending point")
    if end:
        end = int(end)




    if st.button("Get Prediction"):
       
        cust_info = df.iloc[start:end, :]
        X = cust_info.drop(columns='CUST_ID')
        scaled_cust_info = scaler.transform(X)
        reduced_cust_info = pca.transform(scaled_cust_info)

        # get prediction
        pred_cluster = kmeans.predict(reduced_cust_info)

        # result
        cust_info['CATEGORY'] = pred_cluster


        st.write("This customer belongs to cluster", cust_info[['CUST_ID', 'CATEGORY']])
elif page == "About Us":
    st.title("About Us")
    st.subheader("Welcome Brain Builders IT Firm Hub")
    st.write("Brain Builders IT Firm is Osun partnered hub for 3MTT DeepTech_Ready Skill Aquisition Program. " \
    "Brain Builders IT Firm is Osun partnered hub for 3MTT DeepTech_Ready Skill Aquisition Program," \
    " Brain Builders IT Firm is Osun partnered hub for 3MTT DeepTech_Ready Skill Aquisition Program")
