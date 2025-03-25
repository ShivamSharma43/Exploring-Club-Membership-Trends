import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
data_path = "club_membership.csv"
df = pd.read_csv(data_path)

#The dataset has columns: 'Semester', 'Club', 'Total Members', 'Event Attendance'
# Pivot data for visualization
membership_trends = df.pivot(index='Semester', columns='Club', values='Total Members')
event_trends = df.pivot(index='Semester', columns='Club', values='Event Attendance')

#Club Membership Trends Over Semesters
plt.figure(figsize=(10, 5))
for club in membership_trends.columns:
    plt.plot(membership_trends.index, membership_trends[club], marker='o', label=club)
plt.title("Club Membership Trends Over Semesters")
plt.xlabel("Semester")
plt.ylabel("Total Members")
plt.legend()
plt.grid()
plt.show()

#Event Attendance Trends Over Semesters
plt.figure(figsize=(10, 5))
for club in event_trends.columns:
    plt.plot(event_trends.index, event_trends[club], linestyle='dashed', marker='s', label=club)
plt.title("Event Attendance Trends Over Semesters")
plt.xlabel("Semester")
plt.ylabel("Event Attendance")
plt.legend()
plt.grid()
plt.show()

#Membership Trends with Future Predictions
future_semesters = np.array([5, 6]).reshape(-1, 1)
predictions = {}
plt.figure(figsize=(12, 6))
for club in membership_trends.columns:
    X = np.array(membership_trends.index).reshape(-1, 1)
    y = membership_trends[club].values
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(future_semesters)
    predictions[club] = pred
    plt.plot(X, y, marker='o', label=club)
    plt.plot(future_semesters, pred, linestyle='dashed')
plt.title("Club Membership Trends and Predictions")
plt.xlabel("Semester")
plt.ylabel("Membership Count")
plt.legend()
plt.grid()
plt.show()

#Actual vs Predicted Attendance Scatter Plot
X_attendance = event_trends.index.values.reshape(-1, 1)
y_attendance = event_trends.mean(axis=1).values  # Averaging attendance across clubs
model_attendance = LinearRegression()
model_attendance.fit(X_attendance, y_attendance)
predicted_attendance = model_attendance.predict(X_attendance)
plt.figure(figsize=(8, 6))
plt.scatter(y_attendance, predicted_attendance, alpha=0.7)
plt.xlabel("Actual Attendance")
plt.ylabel("Predicted Attendance")
plt.title("Regression Model: Actual vs Predicted Attendance")
plt.grid()
plt.show()
