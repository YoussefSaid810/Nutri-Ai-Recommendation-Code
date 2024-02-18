import tkinter as tk
from tkinter import ttk
import csv
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

diet = None
tdee_Breakfast = None
tdee_Lunch = None
tdee_Dinner = None
tdee_Snack = None


# Load data
file_path = 'newfood.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Preprocess data
data['ingredients'] = data['ingredients'].apply(lambda x: x.lower().split(','))

# Create vocabulary
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
vectorizer.fit(data['ingredients'])

# Convert ingredients to bag-of-words representation
X = vectorizer.transform(data['ingredients']).toarray()

# Get user preferences
user_preferences = ['zucchini', 'olive oil', 'beef','balsamic vinegar']

# Calculate cosine similarity between user preferences and meal ingredients
user_preferences_vec = vectorizer.transform([user_preferences]).toarray()
ingredient_vectors = vectorizer.transform(data['ingredients']).toarray()
cosine_similarities = cosine_similarity(user_preferences_vec, ingredient_vectors).flatten()

# Count matching ingredients between user preferences and each meal
matching_ingredients_count = np.array([sum([pref in ingredients for pref in user_preferences]) for ingredients in data['ingredients']])

# Label meals as preferred or not preferred based on cosine similarity
preferred_threshold = 0.7  # Adjust threshold as needed
preferred_labels = (cosine_similarities >= preferred_threshold).astype(int)

# Build TensorFlow model for classification
class MealPreferenceModel(tf.keras.Model):
    def __init__(self, num_features):
        super(MealPreferenceModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')  # 2 units for binary classification (preferred or not preferred)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

# Instantiate the model
model = MealPreferenceModel(num_features=X.shape[1])

# Compile the model for classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, preferred_labels, epochs=10, batch_size=32)

# Predict preferred meals
predictions = model.predict(X)[:, 1]  # Probability of being a preferred meal

# Combine model predictions, cosine similarity, and matching ingredients count
preference_scores = predictions + cosine_similarities + matching_ingredients_count

# Append preference scores to the DataFrame
data['preference_score'] = preference_scores

# Rank meals based on preference scores
ranked_meals = data.sort_values(by='preference_score', ascending=False)

# Display ranked meals
print("\nRanked meals based on your preferences:")
for i, (index, meal) in enumerate(ranked_meals.iterrows(), start=1):
    print(f"{i}. {meal['title']} - Preference Score: {meal['preference_score']}")

ranked_meals_Bf = ranked_meals[(ranked_meals["dishTypes"] == "salad")]
ranked_meals_Lu = ranked_meals[(ranked_meals["dishTypes"] == "lunch,dinner")]
ranked_meals_Dn = ranked_meals[(ranked_meals["dishTypes"] == "lunch,dinner")]
ranked_meals_Sn = ranked_meals[(ranked_meals["dishTypes"] == "snack")]

def calculate_tdee(weight, height, age, gender, activity_level):
    global tdee_Breakfast
    global tdee_Lunch
    global tdee_Dinner
    global tdee_Snack

    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please enter 'male' or 'female'.")

    activity_levels = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55,
                       'very active': 1.725, 'extra active': 1.9}
    if activity_level.lower() in activity_levels:
        tdee = bmr * activity_levels[activity_level.lower()]
    else:
        raise ValueError("Invalid activity level.")
    tdee_Breakfast = tdee / 100 * 25
    tdee_Lunch = tdee / 100 * 40
    tdee_Dinner = tdee / 100 * 25
    tdee_Snack = tdee / 100 * 10
    return tdee

def calculate_macros(tdee):
    protein_ratio = 0.3
    fat_ratio = 0.25
    carb_ratio = 0.45

    protein = (protein_ratio * tdee) / 4
    fat = (fat_ratio * tdee) / 9
    carbs = (carb_ratio * tdee) / 4

    return protein, fat, carbs

def calculate():
    global tdee_Breakfast
    global tdee_Lunch
    global tdee_Dinner
    global tdee_Snack
    global diet
    diet = diet_var.get()
    gender = gender_var.get()
    age = int(age_entry.get())
    weight = float(weight_entry.get())
    height = float(height_entry.get())
    activity_level = activity_var.get()

    try:
        tdee = calculate_tdee(weight, height, age, gender, activity_level)
        protein, fat, carbs = calculate_macros(tdee)
        tdee_Breakfast = tdee / 100 * 25
        tdee_Dinner = tdee / 100 * 25
        tdee_Lunch = tdee / 100 * 40
        tdee_Snack = tdee / 100 * 10
        print(tdee_Breakfast, tdee_Snack, tdee_Lunch, tdee_Dinner)
        result_text.set("TDEE: {:.2f} calories per day\nProtein: {:.2f} grams per day\nFat: {:.2f} grams per day\nCarbohydrates: {:.2f} grams per day"
                        .format(tdee, protein, fat, carbs))
    except ValueError as e:
        result_text.set("Error: " + str(e))

def rec_Breakfast():
    global tdee_Breakfast
    global diet
    # Define the conditions
    condition_column = "dishTypes"
    condition_value = "salad"
    excludes =['q']
    # Ask the user about their dietary preference
    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('newfood.csv')

    # Check if the condition column exists in the DataFrame
    if condition_column not in df.columns:
        print(f"Error: {condition_column} column not found.")
    else:
        # Filter the DataFrame based on the conditions and print the result
        filtered_df = df[(df[condition_column] == condition_value) & (~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

        # Additional filter for vegan meals if the user is vegan
        if diet:
            # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
            filtered_df = filtered_df[filtered_df[diet] == True]

        while not filtered_df.empty:
            random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
            if diet:
                random_meal['weightPerServing'] = random_meal['weightPerServing']* tdee_Breakfast / random_meal['calories']
                print(random_meal[['title', 'dishTypes', diet ,'weightPerServing']],)
            elif diet is None:
                # serv = random_meal['weightPerServing'] * tdee_Breakfast / random_meal['calories']
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / random_meal['calories']
                print(random_meal[['title', 'dishTypes','weightPerServing']])
            # Ask the user if they like the meal
            user_input = input("Do you like this meal? (yes/no): ").lower()
            if user_input == 'yes':
                break  # Exit the loop if the user likes the meal
            elif user_input == 'no':
                filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
                if filtered_df.empty:
                    print(f"No more meals found for {condition_column} = {condition_value}.")
                    break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print(f"No records found for {condition_column} = {condition_value}.")



def rec_Fav_Breakfast():
    global diet
    global ranked_meals_Bf
    global tdee_Breakfast

    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Additional filter for vegan meals if the user is vegan
    if diet:
        # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
        ranked_meals_Bf = ranked_meals_Bf[ranked_meals_Bf[diet] == True]

    ranked_meals_Bf10=ranked_meals_Bf.head(10)
    print(ranked_meals_Bf10)

    while not ranked_meals_Bf10.empty:
        random_meal = ranked_meals_Bf10.sample(n=1).iloc[0]  # Select one random row from the DataFrame

        # Adjust the following code to handle DataFrame columns appropriately
        if diet:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / random_meal['calories']
            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing' , 'preference_score']], tdee_Breakfast)
        elif diet is None:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / random_meal['calories']
            print(random_meal[['title', 'dishTypes', 'weightPerServing' ,"preference_score"]])

        # Ask the user if they like the meal
        user_input = input("Do you like this meal? (yes/no): ").lower()
        if user_input == 'yes':
            break  # Exit the loop if the user likes the meal
        elif user_input == 'no':
            # ranked_meals_Bf10 = ranked_meals_Bf10.drop(index=random_meal)  # Remove the disliked meal from the DataFrame
            # ranked_meals_Bf10 = ranked_meals_Bf10.reset_index(drop=True)  # Reset the index after dropping rowsif ranked_meals_Bf10.empty:
            if ranked_meals_Bf10.empty:
                print(f"No more meals found.")
                break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")




def rec_Lunch():
    global tdee_Lunch
    global diet

    # Define the conditions
    condition_column = "dishTypes"
    condition_value = "lunch"
    excludes = ['q']
    # Ask the user about their dietary preference
    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('newfood.csv')

    # Check if the condition column exists in the DataFrame
    if condition_column not in df.columns:
        print(f"Error: {condition_column} column not found.")
    else:
        # Filter the DataFrame based on the conditions and print the result
        filtered_df = df[(df[condition_column].str.lower().str.contains('|'.join(condition_value))) & (~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

        # Additional filter for vegan meals if the user is vegan
        if diet:
            # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
            filtered_df = filtered_df[filtered_df[diet] == True]
        else:
            print("")



        while not filtered_df.empty:
            random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
            if diet:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / random_meal['calories']
                print(random_meal[['title', 'dishTypes', diet , 'weightPerServing']])
            elif diet is None:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / random_meal['calories']
                print(random_meal[['title', 'dishTypes','weightPerServing']])
            # Ask the user if they like the meal
            user_input = input("Do you like this meal? (yes/no): ").lower()
            if user_input == 'yes':
                break  # Exit the loop if the user likes the meal
            elif user_input == 'no':
                filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
            if filtered_df.empty:
                    print(f"No more meals found for {condition_column} = {condition_value}.")
                    break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print(f"No records found for {condition_column} = {condition_value}.")



def rec_Fav_Lunch():
    global diet
    global ranked_meals_Lu
    global tdee_Lunch

    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Additional filter for vegan meals if the user is vegan
    if diet:
        # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
        ranked_meals_Lu = ranked_meals_Lu[ranked_meals_Lu[diet] == True]

    ranked_meals_Lu10=ranked_meals_Lu.head(10)
    print(ranked_meals_Lu10)

    while not ranked_meals_Lu10.empty:
        random_meal = ranked_meals_Lu10.sample(n=1).iloc[0]  # Select one random row from the DataFrame

        # Adjust the following code to handle DataFrame columns appropriately
        if diet:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / random_meal['calories']
            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing' , 'preference_score']])
        elif diet is None:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / random_meal['calories']
            print(random_meal[['title', 'dishTypes', 'weightPerServing' ,"preference_score"]])

        # Ask the user if they like the meal
        user_input = input("Do you like this meal? (yes/no): ").lower()
        if user_input == 'yes':
            break  # Exit the loop if the user likes the meal
        elif user_input == 'no':
            # ranked_meals_Bf10 = ranked_meals_Bf10.drop(ranked_meals_Bf10.index[random_meal.index],axis=0)  # Remove the disliked meal from the DataFrame
            # ranked_meals_Bf10 = ranked_meals_Bf10.reset_index(drop=True)  # Reset the index after dropping rows
            if ranked_meals_Lu10.empty:
                print(f"No more meals found.")
                break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")



def rec_Dinner():
    global tdee_Dinner
    global diet
    # Define the conditions
    condition_column = "dishTypes"
    condition_value = "dinner"
    excludes = ['q']
    # Ask the user about their dietary preference
    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('newfood.csv')

    # Check if the condition column exists in the DataFrame
    if condition_column not in df.columns:
        print(f"Error: {condition_column} column not found.")
    else:
        # Filter the DataFrame based on the conditions and print the result
        filtered_df = df[(df[condition_column].str.lower().str.contains('|'.join(condition_value))) & (~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

        # Additional filter for vegan meals if the user is vegan
        if diet:
            # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
            filtered_df = filtered_df[filtered_df[diet] == True]
        else:
            print("")



        while not filtered_df.empty:
            random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
            if diet:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / random_meal['calories']
                print(random_meal[['title', 'dishTypes', diet , 'weightPerServing']])
            elif diet is None:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / random_meal['calories']
                print(random_meal[['title', 'dishTypes','weightPerServing']])
            # Ask the user if they like the meal
            user_input = input("Do you like this meal? (yes/no): ").lower()
            if user_input == 'yes':
                break  # Exit the loop if the user likes the meal
            elif user_input == 'no':
                filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
            if filtered_df.empty:
                    print(f"No more meals found for {condition_column} = {condition_value}.")
                    break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print(f"No records found for {condition_column} = {condition_value}.")



def rec_Fav_Dinner():
    global diet
    global ranked_meals_Dn
    global tdee_Dinner

    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Additional filter for vegan meals if the user is vegan
    if diet:
        # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
        ranked_meals_Dn = ranked_meals_Dn[ranked_meals_Dn[diet] == True]

    ranked_meals_Dn10=ranked_meals_Dn.head(10)
    print(ranked_meals_Dn10)

    while not ranked_meals_Dn10.empty:
        random_meal = ranked_meals_Dn10.sample(n=1).iloc[0]  # Select one random row from the DataFrame

        # Adjust the following code to handle DataFrame columns appropriately
        if diet:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / random_meal['calories']
            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing' , 'preference_score']])
        elif diet is None:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / random_meal['calories']
            print(random_meal[['title', 'dishTypes', 'weightPerServing' ,"preference_score"]])

        # Ask the user if they like the meal
        user_input = input("Do you like this meal? (yes/no): ").lower()
        if user_input == 'yes':
            break  # Exit the loop if the user likes the meal
        elif user_input == 'no':
            # ranked_meals_Bf10 = ranked_meals_Bf10.drop(ranked_meals_Bf10.index[random_meal.index],axis=0)  # Remove the disliked meal from the DataFrame
            # ranked_meals_Bf10 = ranked_meals_Bf10.reset_index(drop=True)  # Reset the index after dropping rows
            if ranked_meals_Dn10.empty:
                print(f"No more meals found.")
                break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")



def rec_Snack():
    global tdee_Snack
    global diet
    # Define the conditions
    condition_column = "dishTypes"
    condition_value = "snack"
    excludes = ['q']
    # Ask the user about their dietary preference
    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('newfood.csv')

    # Check if the condition column exists in the DataFrame
    if condition_column not in df.columns:
        print(f"Error: {condition_column} column not found.")
    else:
        # Filter the DataFrame based on the conditions and print the result
        filtered_df = df[(df[condition_column] == condition_value) & (~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

        # Additional filter for vegan meals if the user is vegan
        if diet:
            # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
            filtered_df = filtered_df[filtered_df[diet] == True]
        else:
            print("")



        while not filtered_df.empty:
            random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
            if diet:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / random_meal['calories']
                print(random_meal[['title', 'dishTypes',diet , 'weightPerServing']])
            elif diet is None:
                random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / random_meal['calories']
                print(random_meal[['title', 'dishTypes','weightPerServing']])
            # Ask the user if they like the meal
            user_input = input("Do you like this meal? (yes/no): ").lower()
            if user_input == 'yes':
                break  # Exit the loop if the user likes the meal
            elif user_input == 'no':
                filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
            if filtered_df.empty:
                    print(f"No more meals found for {condition_column} = {condition_value}.")
                    break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print(f"No records found for {condition_column} = {condition_value}.")



def rec_Fav_Snack():
    global diet
    global ranked_meals_Sn
    global tdee_Snack

    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()

    # Additional filter for vegan meals if the user is vegan
    if diet:
        # diet = input("Please write your diet: (vegan/vegetarian/glutenFree/dairyFree/sustainable/lowFodMap/ketogenic/whole30)")
        ranked_meals_Sn = ranked_meals_Sn[ranked_meals_Sn[diet] == True]

    ranked_meals_Sn10=ranked_meals_Sn.head(10)
    print(ranked_meals_Sn10)

    while not ranked_meals_Sn10.empty:
        random_meal = ranked_meals_Sn10.sample(n=1).iloc[0]  # Select one random row from the DataFrame

        # Adjust the following code to handle DataFrame columns appropriately
        if diet:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / random_meal['calories']
            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing' , 'preference_score']])
        elif diet is None:
            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / random_meal['calories']
            print(random_meal[['title', 'dishTypes', 'weightPerServing' ,"preference_score"]])

        # Ask the user if they like the meal
        user_input = input("Do you like this meal? (yes/no): ").lower()
        if user_input == 'yes':
            break  # Exit the loop if the user likes the meal
        elif user_input == 'no':
            # ranked_meals_Bf10 = ranked_meals_Bf10.drop(ranked_meals_Bf10.index[random_meal.index],axis=0)  # Remove the disliked meal from the DataFrame
            # ranked_meals_Bf10 = ranked_meals_Bf10.reset_index(drop=True)  # Reset the index after dropping rows
            if ranked_meals_Sn10.empty:
                print(f"No more meals found.")
                break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def generate_one_plan():
    global tdee_Breakfast
    global tdee_Lunch
    global tdee_Dinner
    global tdee_Snack
    global diet

    # Generate plans for each meal type (e.g., breakfast, lunch, dinner)
    for meal_type in ["salad", "lunch", "dinner", "snack"]:
        condition_column = "dishTypes"
        condition_value = meal_type
        excludes = ['q']

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('newfood.csv')

        # Check if the condition column exists in the DataFrame
        if condition_column not in df.columns:
            print(f"Error: {condition_column} column not found.")
        else:

            if condition_value == "dinner" or condition_value == "lunch":
                # Filter the DataFrame based on the conditions and print the result
                filtered_df = df[(df[condition_column] == "lunch,dinner") & (
                    ~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]
            else:
                # Filter the DataFrame based on the conditions and print the result
                filtered_df = df[(df[condition_column] == condition_value) & (
                    ~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

            # Apply the dietary preference filter
            if diet:
                filtered_df = filtered_df[filtered_df[diet] == True]

            while not filtered_df.empty:
                random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
                if diet:
                    if meal_type == "salad":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])
                    elif meal_type == "lunch":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])
                    elif meal_type == "dinner":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])
                    elif meal_type == "snack":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])

                elif diet is None:
                    if meal_type == "salad":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                    elif meal_type == "lunch":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                    elif meal_type == "dinner":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                    elif meal_type == "snack":
                        random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / \
                                                          random_meal['calories']
                        print(random_meal[['title', 'dishTypes', 'weightPerServing']])

                # Ask the user if they like the meal
                user_input = input("Do you like this meal? (yes/no): ").lower()
                if user_input == 'yes':
                    break  # Exit the loop if the user likes the meal
                elif user_input == 'no':
                    filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
                    if filtered_df.empty:
                        print(f"No more meals found for {condition_column} = {condition_value}.")
                        break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            else:
                print(f"No records found for {condition_column} = {condition_value}.")


def generate_weekly_plans():
    global tdee_Breakfast
    global tdee_Lunch
    global tdee_Dinner
    global tdee_Snack
    global diet

    # user_diet_preference = input("Are you on a specific diet? (yes/no): ").lower()
    # diet = None


    # Assuming we want to generate plans for each day of the week (e.g., breakfast, lunch, dinner)
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        print(f"\n{day} Plan:")

        # Generate plans for each meal type (e.g., breakfast, lunch, dinner)
        for meal_type in ["salad", "lunch", "dinner","snack"]:
            condition_column = "dishTypes"
            condition_value = meal_type
            excludes = ['q']

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv('newfood.csv')

            # Check if the condition column exists in the DataFrame
            if condition_column not in df.columns:
                print(f"Error: {condition_column} column not found.")
            else:

                if condition_value == "dinner" or condition_value == "lunch":
                    # Filter the DataFrame based on the conditions and print the result
                    filtered_df = df[(df[condition_column] == "lunch,dinner") & (
                        ~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]
                else:
                    # Filter the DataFrame based on the conditions and print the result
                    filtered_df = df[(df[condition_column] == condition_value) & (
                        ~df['ingredients'].str.lower().str.contains('|'.join(excludes)))]

                # Apply the dietary preference filter
                if diet:
                    filtered_df = filtered_df[filtered_df[diet] == True]


                while not filtered_df.empty:
                    random_meal = filtered_df.sample(n=1)  # Select one random row from the DataFrame
                    if diet:
                        if meal_type == "salad":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / \
                                                          random_meal['calories']
                            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])
                        elif meal_type == "lunch":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', diet , 'weightPerServing']])
                        elif meal_type == "dinner":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])
                        elif meal_type == "snack":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', diet, 'weightPerServing']])

                    elif diet is None:
                        if meal_type == "salad":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Breakfast / \
                                                          random_meal['calories']
                            print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                        elif meal_type == "lunch":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Lunch / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                        elif meal_type == "dinner":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Dinner / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', 'weightPerServing']])
                        elif meal_type == "snack":
                            random_meal['weightPerServing'] = random_meal['weightPerServing'] * tdee_Snack / \
                                                              random_meal['calories']
                            print(random_meal[['title', 'dishTypes', 'weightPerServing']])

                    # Ask the user if they like the meal
                    user_input = input("Do you like this meal? (yes/no): ").lower()
                    if user_input == 'yes':
                        break  # Exit the loop if the user likes the meal
                    elif user_input == 'no':
                        filtered_df = filtered_df.drop(random_meal.index)  # Remove the disliked meal from the DataFrame
                        if filtered_df.empty:
                            print(f"No more meals found for {condition_column} = {condition_value}.")
                            break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
                else:
                    print(f"No records found for {condition_column} = {condition_value}.")






# Create GUI
root = tk.Tk()
root.title("TDEE and Macro Nutrients Calculator")

mainframe = ttk.Frame(root, padding="20")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

gender_label = ttk.Label(mainframe, text="Gender:")
gender_label.grid(column=1, row=1, sticky=tk.W)
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(mainframe, textvariable=gender_var, values=["Male", "Female"])
gender_combobox.grid(column=2, row=1)

age_label = ttk.Label(mainframe, text="Age (years):")
age_label.grid(column=1, row=2, sticky=tk.W)
age_entry = ttk.Entry(mainframe)
age_entry.grid(column=2, row=2)

weight_label = ttk.Label(mainframe, text="Weight (kg):")
weight_label.grid(column=1, row=3, sticky=tk.W)
weight_entry = ttk.Entry(mainframe)
weight_entry.grid(column=2, row=3)

height_label = ttk.Label(mainframe, text="Height (cm):")
height_label.grid(column=1, row=4, sticky=tk.W)
height_entry = ttk.Entry(mainframe)
height_entry.grid(column=2, row=4)

activity_label = ttk.Label(mainframe, text="Activity Level:")
activity_label.grid(column=1, row=5, sticky=tk.W)
activity_var = tk.StringVar()
activity_combobox = ttk.Combobox(mainframe, textvariable=activity_var, values=["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
activity_combobox.grid(column=2, row=5)

diet_label = ttk.Label(mainframe, text="Diet:")
diet_label.grid(column=1, row=6, sticky=tk.W)
diet_var = tk.StringVar()
diet_combobox = ttk.Combobox(mainframe, textvariable=diet_var, values=["vegan","vegetarian","glutenFree","dairyFree","sustainable","lowFodMap","ketogenic","whole30","None"])
diet_combobox.grid(column=2, row=6)

calculate_button = ttk.Button(mainframe, text="Calculate", command=calculate)
calculate_button.grid(column=2, row=7)

result_text = tk.StringVar()
result_label = ttk.Label(mainframe, textvariable=result_text)
result_label.grid(column=1, row=8, columnspan=2)

calculatebf_button = ttk.Button(mainframe, text="Calculate bf", command=rec_Breakfast)
calculatelu_button = ttk.Button(mainframe, text="Calculate lu", command=rec_Lunch)
calculatedn_button = ttk.Button(mainframe, text="Calculate dn", command=rec_Dinner)
calculateSn_button = ttk.Button(mainframe, text="Calculate sn", command=rec_Snack)

calculatefavbf_button = ttk.Button(mainframe, text="Calculate fav bf", command=rec_Fav_Breakfast)
calculatefavlu_button = ttk.Button(mainframe, text="Calculate fav lu", command=rec_Fav_Lunch)
calculatefavdn_button = ttk.Button(mainframe, text="Calculate fav dn", command=rec_Fav_Dinner)
calculatefavsn_button = ttk.Button(mainframe, text="Calculate fav sn", command=rec_Fav_Snack)
# Add a button for generating one plan
generate_one_plan_button = ttk.Button(mainframe, text="Generate One Plan", command=generate_one_plan)
generate_one_plan_button.grid(column=2, row=9)

# Add a button for generating weekly plans
generate_weekly_plans_button = ttk.Button(mainframe, text="Generate Weekly Plans", command=generate_weekly_plans)
generate_weekly_plans_button.grid(column=2, row=10)
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)


root.mainloop()