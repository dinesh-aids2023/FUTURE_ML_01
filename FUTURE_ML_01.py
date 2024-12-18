import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and explore the dataset
housing = pd.read_csv('/content/train.csv')
housing.info()

# Create a dataframe with relevant features
house = pd.DataFrame().assign(
    Price=housing['price'], 
    Area=housing['area'], 
    Bedrooms=housing['bedrooms'], 
    Bathrooms=housing['bathrooms']
)

# Data Cleaning - Check for null values
print(house.isnull().sum())

# Removing null values for simplicity (if any)
house.dropna(inplace=True)

# Define features (X) and target (y)
x = house[['Area', 'Bedrooms', 'Bathrooms']]
y = house['Price']

# Outlier Treatment for Price
Q1 = house.Price.quantile(0.25)
Q3 = house.Price.quantile(0.75)
IQR = Q3 - Q1
house = house[(house.Price >= Q1 - 1.5 * IQR) & (house.Price <= Q3 + 1.5 * IQR)]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Display model coefficients
print("Intercept: ", lr.intercept_)
print("Coefficients: ", lr.coef_)

# Take user input
try:
    user_area = float(input("Enter the square footage of the house (Area): "))
    user_bedrooms = int(input("Enter the number of bedrooms: "))
    user_bathrooms = int(input("Enter the number of bathrooms: "))

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'Area': [user_area],
        'Bedrooms': [user_bedrooms],
        'Bathrooms': [user_bathrooms]
    })
    # Predict the house price
    predicted_price = lr.predict(user_input)
    print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")
except ValueError:
    print("Invalid input! Please enter numeric values for all fields.")

    #visualization
    plt.scatter(y_train, y_train)
    plt.xlabel("Real Price")
    plt.ylabel("Predicted Price")
    plt.show()