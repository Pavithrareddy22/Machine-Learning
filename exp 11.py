import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Step 1: Create Dataset
# --------------------------------------------------
data_dict = {
"Annual_Income":[25000,30000,40000,50000,60000,70000,45000,52000],
"Monthly_Inhand_Salary":[2000,2500,3200,4000,4800,5500,3500,4200],
"Num_Bank_Accounts":[2,3,4,3,5,4,3,4],
"Num_Credit_Card":[1,2,3,3,4,5,2,3],
"Interest_Rate":[12,10,9,8,7,6,9,8],
"Num_of_Loan":[1,1,2,2,3,2,2,1],
"Delay_from_due_date":[5,3,2,1,0,0,2,1],
"Num_of_Delayed_Payment":[2,1,1,0,0,0,1,0],
"Credit_Mix":[0,1,1,2,2,2,1,2],
"Outstanding_Debt":[5000,4000,3500,2000,1500,1000,2500,1800],
"Credit_History_Age":[2,3,4,5,6,7,4,5],
"Monthly_Balance":[500,800,1000,1500,2000,2500,1200,1600],
"Credit_Score":["Poor","Poor","Standard","Standard","Good","Good","Standard","Good"]
}

data = pd.DataFrame(data_dict)

print("Dataset Preview:\n")
print(data.head())

# --------------------------------------------------
# Step 2: Prepare Features and Target
# --------------------------------------------------
X = np.array(data[["Annual_Income","Monthly_Inhand_Salary",
                   "Num_Bank_Accounts","Num_Credit_Card",
                   "Interest_Rate","Num_of_Loan",
                   "Delay_from_due_date","Num_of_Delayed_Payment",
                   "Credit_Mix","Outstanding_Debt",
                   "Credit_History_Age","Monthly_Balance"]])

y = np.array(data["Credit_Score"])

# --------------------------------------------------
# Step 3: Split Dataset
# --------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# --------------------------------------------------
# Step 4: Train Model
# --------------------------------------------------
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# --------------------------------------------------
# Step 5: User Input for Prediction
# --------------------------------------------------
print("\nCredit Score Prediction")

a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit Cards: "))
e = float(input("Interest Rate: "))
f = float(input("Number of Loans: "))
g = float(input("Delay from Due Date: "))
h = float(input("Number of Delayed Payments: "))
i = float(input("Credit Mix (0=Bad,1=Standard,2=Good): "))
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a,b,c,d,e,f,g,h,i,j,k,l]])

prediction = model.predict(features)

print("\nPredicted Credit Score =", prediction[0])
