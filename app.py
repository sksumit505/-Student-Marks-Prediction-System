import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split)      # It give me the power to slipt the data in part so i can train and teach 
from sklearn.ensemble import(RandomForestRegressor)         #this is the ML model which i use to make predction 
import streamlit as st
from sklearn.metrics import mean_absolute_error             #to manage the mean_absolute_error
#creation of datads
def train_model():
    no_of_sample=200
    student_id=np.random.randint(0,300,no_of_sample)    #student_id
    study_hours=np.random.randint(1,12,no_of_sample)    #In hours
    sleep_hours=np.random.randint(3,8,no_of_sample)     #In hours
    Past_perf=np.random.randint(0,100,no_of_sample)     #in No. in the range of 1to 100
    Atten=np.random.randint(0,100,no_of_sample)
    # Generate upcoming_performance using a weighted realistic relationship + minor random noise
    Upcoming_per = (
        (0.4 * Past_perf) + 
        (3 * study_hours) + 
        (0.3 * Atten) + 
        (1.5 * sleep_hours) + 
     np.random.normal(0, 5, no_of_sample) # Adding slight real-world noise
        )
    # Clip values between 0 and 100 (marks scale)
    Upcoming_per = np.clip(Upcoming_per, 0, 100).astype(int)

    #Data_frame_creation
    data = pd.DataFrame({
    "student_id": student_id,
    "sleep_hours": sleep_hours,
    "past_performance": Past_perf,
    "attendance": Atten,
    "study_hours": study_hours,
    "upcoming_performance": Upcoming_per
    })
 #Droping the duplected values
    data = data.drop_duplicates(subset=["student_id"])

    #splitong the data in 80-20 ratio to teach and test the ML model
    x = data[["past_performance", "sleep_hours", "attendance", "study_hours"]]
    y = data["upcoming_performance"]              #output data
    x_train,x_test,y_train,y_test=(train_test_split(x,y,test_size=0.2,random_state=42))#Hear we are giveing the 80 percent of to to train the ai model and 
                                                                   #rest use to test it by using train_test_split

    #using this code to replace the Na value with some median
    median_vals = x_train.median()
    x_train = x_train.fillna(median_vals)
    x_test = x_test.fillna(median_vals)

    #train model

    model=RandomForestRegressor(random_state=42)
    model.fit(x_train,y_train)
    print("your data has been trained")

    prediction=model.predict(x_test)
    mae=mean_absolute_error(y_test,prediction)
    return model, round(mae, 1)

# Train the model
model, mae = train_model()

# Display Model Performance Metric
st.sidebar.header("📊 Model Status")
st.sidebar.success("Model Status: Trained Successfully!")
st.sidebar.metric(label="Mean Absolute Error (MAE)", value=f"{mae} marks")

# ---------------------------------------------------------
# 2. STREAMLIT FRONTEND INPUTS
# ---------------------------------------------------------
st.subheader("Enter Student Parameters")

col1, col2 = st.columns(2)

with col1:
    past_perf = st.slider("Past Performance (0 - 100):", min_value=0, max_value=100, value=75)
    study_hrs = st.slider("Daily Study Hours (1 - 12):", min_value=1, max_value=12, value=6)

with col2:
    atten = st.slider("Attendance Percentage (0 - 100%):", min_value=0, max_value=100, value=85)
    sleep_hrs = st.slider("Daily Sleep Hours (3 - 8):", min_value=3, max_value=8, value=7)

# ---------------------------------------------------------
# 3. PREDICTION TRIGGER
# ---------------------------------------------------------
st.markdown("---")
if st.button("🚀 Predict Upcoming Marks", use_container_width=True):
    # Format input into a DataFrame matching training features
    user_data = pd.DataFrame({
        "past_performance": [past_perf],
        "sleep_hours": [sleep_hrs],
        "attendance": [atten],
        "study_hours": [study_hrs]
    })

    # Predict result
    predicted_marks = model.predict(user_data)[0]
    predicted_marks = np.clip(predicted_marks, 0, 100)  # Bound result between 0 and 100

    # Output display
    st.balloons()
    st.markdown(f"### 🎯 Predicted Score: **{round(predicted_marks, 1)} / 100**")