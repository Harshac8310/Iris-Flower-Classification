import joblib

# Load the saved model
model = joblib.load("iris_model.joblib")

# Measurements of a new flower
flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict the species
prediction = model.predict(flower)

# Convert the numerical label into a species name
species_names = ["setosa", "versicolor", "virginica"]
predicted_species = species_names[prediction[0]]

print("Predicted species:", predicted_species)