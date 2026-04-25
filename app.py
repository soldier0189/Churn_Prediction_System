
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st
import pandas as pd
import pickle

with open("churn.pkl", "rb") as file:
    model = pickle.load(file)

def preprocess(data):
    df = pd.DataFrame([data.dict()])

    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["geography"] = df["geography"].map({"France": 0, "Spain": 1, "Germany": 2})

    df["has_cr_card"] = df["has_cr_card"].astype(int)
    df["is_active_member"] = df["is_active_member"].astype(int)

    # match training column names exactly
    df = df.rename(columns={
        "creditScore": "CreditScore",
        "geography": "Geography",
        "gender": "Gender",
        "age": "Age",
        "tenure": "Tenure",
        "balance": "Balance",
        "num_of_products": "NumOfProducts",
        "has_cr_card": "HasCrCard",
        "is_active_member": "IsActiveMember",
        "estimated_salary": "EstimatedSalary"
    })

    return df

class UserInfo(BaseModel):
    creditScore: int = Field(description="Credit Score", gt=0)
    geography: Literal["France", "Spain", "Germany"]
    gender: Literal["Male", "Female"] 
    age: int = Field(gt=0, lt=120)
    tenure: int = Field(gt=0, lt=100)
    balance: float = Field(ge=0.0)
    num_of_products: int = Field(description="No. of Products", gt=0)
    has_cr_card: bool = Field(description="Has credit card or not")
    is_active_member: bool
    estimated_salary: float = Field(ge=0)


def main():
    st.title("Bank Churn Prediction")
    creditScore = st.number_input("Credit Score")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age")
    tenure = st.number_input("Tenure")
    balance = st.number_input("Balance")
    num_of_products = st.number_input("Number of Products")
    has_cr_card = st.radio("Credit Card", [True, False])
    is_active_member = st.radio("Active member", [True, False])
    estimated_salary = st.number_input("Estimated Salary")


    if st.button("Submit"):


        user = UserInfo(
            creditScore=creditScore,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary
        )

        

        processed_data = preprocess(user)
        
        prob = model.predict_proba(processed_data)[0][1]

        if prob >= 0.5:
            st.error("Customer will CHURN ❌")
        else:
            st.success("Customer will STAY ✅")

        st.write(f"Churn Probability: {prob:.2f}")



if __name__ == "__main__":
    main()





