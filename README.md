Taxi Fare Prediction Using Random Forest
This project predicts taxi fares based on various trip details such as pickup and dropoff locations, passenger count, and other relevant features. The model is trained using the Random Forest algorithm, and an interactive Streamlit app is created to allow users to input trip details and receive fare predictions.

Key Features:
Machine Learning Model: Trained using a custom taxi fare dataset to predict the fare for a given trip.
Model Type: Random Forest Regressor used for accurate fare predictions.
Interactive Web App: Built using Streamlit, allowing users to input trip details (e.g., pickup/dropoff coordinates, passenger count) and get an instant fare prediction.

Installation Instructions:

Clone the repository:
git clone https://github.com/SahilSelokar/taxi-fare-prediction-using-random-forest.git

Navigate to the project directory:
cd taxi-fare-prediction-using-random-forest

Install dependencies:
pip install -r requirements.txt

Train the model (if not already trained):
python train_model.py

Run the Streamlit app:
streamlit run app.py

Project Structure:
taxi-fare-prediction-using-random-forest/
│
├── app.py                # Streamlit app for fare prediction
├── train_model.py        # Script to train and save the machine learning model
├── taxi_fare_model.pkl   # Saved model
├── train.csv             # Dataset for training the model
└── requirements.txt      # Python dependencies

Technologies Used:
Python: Programming language used for model training and app development.
Random Forest Regressor: Machine learning algorithm for regression tasks.
Streamlit: Framework for building the interactive web app.
Pandas & Scikit-learn: Used for data processing and model training.

Dataset:
The dataset used for training the model contains the following columns:

pickup_longitude: Longitude of the pickup location.
pickup_latitude: Latitude of the pickup location.
dropoff_longitude: Longitude of the dropoff location.
dropoff_latitude: Latitude of the dropoff location.
passenger_count: Number of passengers.
fare_amount: The actual fare of the trip (target variable).
Contributing:
Feel free to fork the repository and submit pull requests if you have suggestions for improvements or new features.

License:
This project is open-source and available under the MIT License.
