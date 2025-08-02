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
    st.write("Play with the demo app")

    input_method = st.radio("Chose your method to input the data", ["Upload file CSV", 'Manual Entry'])

    

    if input_method == "Upload file CSV": 
        # upload the file
        upload_data = st.file_uploader("Upload you transaction data", type=['csv'])

        if upload_data is not None:
            try:
                # Read the CSV file into a Pandas DataFrame
                df = pd.read_csv(upload_data)
                df = df.dropna().reset_index(drop=True)
                st.write("Top 5 rows")
                st.write(df.head())

                # get the customer info and the preprocess it
                start = st.text_input("Starting point")
                
                if start:
                    start = int(start)
                end = st.text_input("Ending point")
                if end:
                    end = int(end)


                cust_info = df.iloc[start:end, :]

                

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a CSV file to proceed.")
    elif input_method == 'Manual Entry':
        st.write("Please enter the customers detail")
        data = []

        col1, col2, col3 = st.columns(3)

        with col1:        
            CUST_ID = st.text_input("Customer ID")
            BALANCE = st.text_input("BALANCE")
        
        with col2: 
            BALANCE_FREQUENCY = st.text_input("BALANCE_FREQUENCY")
            PURCHASES = st.text_input("PURCHASES")
        
        with col3:
            ONEOFF_PURCHASES = st.text_input("ONEOFF_PURCHASES")
            INSTALLMENTS_PURCHASES = st.text_input("INSTALLMENTS_PURCHASES")
        
        if all([CUST_ID, BALANCE]):
            data.append({"CUST_ID": CUST_ID,
                        "BALANCE": float(BALANCE),
                        "BALANCE_FREQUENCY": float(BALANCE_FREQUENCY),
                        "PURCHASES": PURCHASES,
                        "ONEOFF_PURCHASES": float(ONEOFF_PURCHASES),
                        "INSTALLMENTS_PURCHASES": float(INSTALLMENTS_PURCHASES)
                        })

        df = pd.DataFrame(data)

        cust_info = df.copy()

    else:
        st.write("Enter your data")



    if st.button("Get Prediction"):
       
        
        
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
