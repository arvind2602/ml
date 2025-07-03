# Airbnb Data Analysis and Price Prediction Tutorial

## 1. Data Loading & Assessment

**Explanation**  
We'll load the Airbnb dataset from a provided URL, preview its structure, and assess its data types and missing values to understand its contents, identify issues, and plan cleaning.

```python
import pandas as pd
import numpy as np

# Load the dataset
url = "https://github.com/adelnehme/python-for-spreadsheet-users-webinar/blob/master/datasets/airbnb.csv?raw=true"
df = pd.read_csv(url)

# Preview the data
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
display(df.head())

# Check data types and missing values
print("\nData Types and Missing Values:")
display(df.info())

# Summary statistics
print("\nSummary Statistics:")
display(df.describe())
```

**Why This Matters**  
- Previewing helps identify the dataset's structure (rows, columns) and content.  
- Data types reveal if columns are correctly formatted (e.g., numerical vs. categorical).  
- Missing values highlight areas needing cleaning.  
- Summary statistics provide a quick sense of distributions and potential outliers.  
- **Common Mistake**: Ignoring data types can lead to errors in analysis (e.g., treating numerical strings as objects).

## 2. Data Cleaning with Enhanced Outlier Handling

**Explanation**  
We'll handle missing values and outliers to ensure data quality. Missing values will be imputed or dropped based on context. For outliers, we'll use the Interquartile Range (IQR) method for `price`, `minimum_nights`, and `number_of_reviews`, and explore the Z-score method as an alternative for `price`. Visualizations (boxplots) will help inspect outliers before and after treatment.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Handle missing values
print("Missing Values Before Cleaning:")
display(df.isnull().sum())

# Impute missing values for numerical columns
df['price'].fillna(df['price'].median(), inplace=True)
df['minimum_nights'].fillna(df['minimum_nights'].median(), inplace=True)
df['number_of_reviews'].fillna(df['number_of_reviews'].median(), inplace=True)

# Drop rows with missing categorical data if less than 5%
df.dropna(subset=['neighbourhood', 'room_type'], inplace=True)

# Visualize outliers before treatment
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['price'])
plt.title('Price Before Outlier Treatment')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['minimum_nights'])
plt.title('Minimum Nights Before Outlier Treatment')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['number_of_reviews'])
plt.title('Number of Reviews Before Outlier Treatment')
plt.tight_layout()
plt.show()

# Method 1: IQR-based outlier handling for price, minimum_nights, and number_of_reviews
def cap_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply IQR capping
df['price'] = cap_outliers_iqr(df['price'])
df['minimum_nights'] = cap_outliers_iqr(df['minimum_nights'])
df['number_of_reviews'] = cap_outliers_iqr(df['number_of_reviews'])

# Method 2: Z-score-based outlier handling for price (alternative approach)
def cap_outliers_zscore(series, threshold=3):
    z_scores = (series - series.mean()) / series.std()
    return series.clip(lower=series[z_scores > -threshold].min(), upper=series[z_scores < threshold].max())

# Apply Z-score capping (for demonstration, applied to a copy of price)
df['price_zscore'] = cap_outliers_zscore(df['price'])

# Visualize outliers after treatment
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['price'])
plt.title('Price After IQR Treatment')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['minimum_nights'])
plt.title('Minimum Nights After IQR Treatment')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['number_of_reviews'])
plt.title('Number of Reviews After IQR Treatment')
plt.tight_layout()
plt.show()

# Compare IQR and Z-score for price
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['price', 'price_zscore']])
plt.title('Price: IQR vs. Z-score Outlier Treatment')
plt.ylabel('Price')
plt.xticks([0, 1], ['IQR Capped', 'Z-score Capped'])
plt.show()

print("\nMissing Values After Cleaning:")
display(df.isnull().sum())
print("\nPrice Distribution After IQR Outlier Treatment:")
display(df['price'].describe())
print("\nMinimum Nights Distribution After IQR Outlier Treatment:")
display(df['minimum_nights'].describe())
print("\nNumber of Reviews Distribution After IQR Outlier Treatment:")
display(df['number_of_reviews'].describe())
```

**Why This Matters**  
- **Missing Values**: Imputing with median for numerical data (`price`, `minimum_nights`, `number_of_reviews`) preserves distribution; dropping sparse categorical missing data (`neighbourhood`, `room_type`) minimizes bias.  
- **Outliers**:  
  - **IQR Method**: Caps extreme values for `price`, `minimum_nights`, and `number_of_reviews` using the 1.5*IQR rule, robust for skewed distributions.  
  - **Z-score Method**: Caps values beyond ±3 standard deviations for `price`, useful for comparison or less skewed data.  
  - **Visualizations**: Boxplots before and after treatment assess outlier handling impact.  
- **Justification**:  
  - IQR is preferred for skewed data (e.g., `price`, `minimum_nights`).  
  - Z-score is included for demonstration, suitable for normally distributed data.  
  - Capping preserves data while reducing outlier impact.  
- **Common Mistake**:  
  - Removing outliers without inspection can discard valid data (e.g., high-priced luxury listings).  
  - Applying the same method to all columns without considering distribution can distort results.

## 3. Feature Engineering

**Explanation**  
We'll create new features like `price_per_room` and `host_duration` to enhance analysis and modeling. Categorical variables like `room_type` will be encoded for machine learning.

```python
# Create price_per_room
df['price_per_room'] = df['price'] / df['bedrooms'].replace(0, 1)  # Avoid division by zero

# Calculate host_duration (assuming host_since is a date)
df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
df['host_duration'] = (pd.Timestamp.now() - df['host_since']).dt.days / 365

# Encode categorical variables
df['room_type_encoded'] = df['room_type'].map({
    'Entire home/apt': 0,
    'Private room': 1,
    'Shared room': 2
})

print("New Features Preview:")
display(df[['price_per_room', 'host_duration', 'room_type_encoded']].head())
```

**Why This Matters**  
- `price_per_room` normalizes cost by space, aiding comparisons.  
- `host_duration` captures host experience, potentially correlating with pricing or quality.  
- Encoding prepares categorical data for modeling.  
- **Common Mistake**: Failing to encode categorical variables can cause errors in machine learning algorithms.

## 4. Normalization

**Explanation**  
We'll normalize numerical features (`price`, `minimum_nights`, `number_of_reviews`, `price_per_room`, `host_duration`, `bedrooms`) using Min-Max scaling to transform them to a [0,1] range. This ensures all features contribute equally to the predictive model, as RandomForestRegressor benefits from consistent scales for feature importance interpretation. We'll visualize distributions before and after normalization to confirm the transformation.

```python
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical features for normalization
num_features = ['price', 'minimum_nights', 'number_of_reviews', 'price_per_room', 'host_duration', 'bedrooms']
num_features = [col for col in num_features if col in df.columns]  # Ensure columns exist

# Visualize distributions before normalization
plt.figure(figsize=(15, 5))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'{col} Before Normalization')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Apply Min-Max normalization
scaler = MinMaxScaler()
df[[f'{col}_normalized' for col in num_features]] = scaler.fit_transform(df[num_features])

# Visualize distributions after normalization
plt.figure(figsize=(15, 5))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[f'{col}_normalized'], bins=30, kde=True)
    plt.title(f'{col} After Normalization')
    plt.xlabel(f'{col} (Normalized)')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("Normalized Features Preview:")
display(df[[f'{col}_normalized' for col in num_features]].head())
```

**Why This Matters**  
- **Normalization**: Scales features to [0,1], ensuring equal contribution to models and improving feature importance interpretation.  
- **Min-Max Scaling**: Preserves the shape of the original distribution, suitable for non-normal data like `price`.  
- **Visualizations**: Confirm that normalization maintains distribution shapes while scaling values.  
- **Justification**: Normalization is applied to all numerical features used in modeling to ensure consistency.  
- **Common Mistake**: Not normalizing features can lead to biased feature importance in models.

## 5. Exploratory Data Analysis (EDA) with 18 Visualization Types

We'll create 18 visualizations to explore the dataset, using `matplotlib`, `seaborn`, `plotly`, and `squarify` for treemaps, including two interactive charts with `ipywidgets` and `plotly`. Each visualization includes an explanation and code.

### Visualization 1: Bar Plot
**Explanation**  
Shows the count of listings by `room_type`.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='room_type')
plt.title('Number of Listings by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Distribution of listing types (e.g., entire home, private room).  
- **Why It’s Useful**: Highlights the most common listing types, informing market trends.  
- **When to Use**: For categorical variable frequency analysis.  
- **Best Practice**: Rotate x-axis labels for readability; avoid cluttered bars.

### Visualization 2: Histogram
**Explanation**  
Displays the distribution of `price_normalized`.

```python
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='price_normalized', bins=30, kde=True)
plt.title('Normalized Price Distribution')
plt.xlabel('Price (Normalized)')
plt.ylabel('Frequency')
plt.show()
```

**Why This Matters**  
- **What it Shows**: How normalized prices are distributed, including skewness.  
- **Why It’s Useful**: Identifies common price ranges and confirms outlier handling.  
- **When to Use**: For continuous variable distributions.  
- **Common Mistake**: Using too few/many bins can obscure patterns.

### Visualization 3: Boxplot
**Explanation**  
Shows `price_normalized` distribution by `room_type`.

```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='room_type', y='price_normalized')
plt.title('Normalized Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price (Normalized)')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Price spread and outliers per room type.  
- **Why It’s Useful**: Compares pricing across categories, highlighting variability.  
- **When to Use**: To compare distributions across groups.  
- **Best Practice**: Use clear labels to avoid misinterpretation.

### Visualization 4: Heatmap
**Explanation**  
Shows correlation between normalized numerical features.

```python
plt.figure(figsize=(10, 8))
numeric_cols = [col for col in df.columns if 'normalized' in col or col == 'room_type_encoded']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Normalized Features')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Strength of relationships between normalized numerical variables.  
- **Why It’s Useful**: Identifies features for modeling (e.g., price correlations).  
- **When to Use**: To explore feature relationships.  
- **Common Mistake**: Including non-numeric data can cause errors.

### Visualization 5: Scatter Plot
**Explanation**  
Plots `price_normalized` vs. `number_of_reviews_normalized`.

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='number_of_reviews_normalized', y='price_normalized')
plt.title('Normalized Price vs. Number of Reviews')
plt.xlabel('Number of Reviews (Normalized)')
plt.ylabel('Price (Normalized)')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Relationship between price and review count.  
- **Why It’s Useful**: Tests if more reviews correlate with higher/lower prices.  
- **When to Use**: To explore relationships between two continuous variables.  
- **Best Practice**: Avoid overplotting by using transparency or sampling.

### Visualization 6: Pie Chart
**Explanation**  
Shows proportion of listings by `neighbourhood`.

```python
plt.figure(figsize=(8, 6))
df['neighbourhood'].value_counts().head(5).plot.pie(autopct='%1.1f%%')
plt.title('Top 5 Neighbourhoods by Listing Proportion')
plt.ylabel('')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Market share of top neighborhoods.  
- **Why It’s Useful**: Highlights popular areas for listings.  
- **When to Use**: For proportional data with few categories.  
- **Common Mistake**: Too many slices make pie charts unreadable.

### Visualization 7: Violin Plot
**Explanation**  
Shows `price_normalized` distribution by `room_type` with density.

```python
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='room_type', y='price_normalized')
plt.title('Normalized Price Distribution by Room Type (Violin)')
plt.xlabel('Room Type')
plt.ylabel('Price (Normalized)')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Density and spread of prices per room type.  
- **Why It’s Useful**: Combines boxplot and kernel density for richer insights.  
- **When to Use**: To compare distributions with density.  
- **Best Practice**: Ensure sufficient data to avoid misleading density estimates.

### Visualization 8: Treemap
**Explanation**  
Visualizes listing counts by `neighbourhood`.

```python
import squarify

plt.figure(figsize=(10, 8))
sizes = df['neighbourhood'].value_counts().head(10)
labels = sizes.index
squarify.plot(sizes=sizes, label=labels, alpha=0.8)
plt.title('Treemap of Listings by Neighbourhood')
plt.axis('off')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Relative size of listings per neighborhood.  
- **Why It’s Useful**: Compact visualization for hierarchical or proportional data.  
- **When to Use**: For categorical data with varying sizes.  
- **Best Practice**: Limit categories to avoid clutter.

### Visualization 9: Line Plot
**Explanation**  
Shows average `price_normalized` over `host_duration_normalized`.

```python
plt.figure(figsize=(8, 6))
avg_price = df.groupby('host_duration_normalized')['price_normalized'].mean().reset_index()
sns.lineplot(data=avg_price, x='host_duration_normalized', y='price_normalized')
plt.title('Average Normalized Price by Host Duration')
plt.xlabel('Host Duration (Normalized)')
plt.ylabel('Average Price (Normalized)')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Trend of pricing with host experience.  
- **Why It’s Useful**: Identifies if longer-hosting hosts charge differently.  
- **When to Use**: For trends over continuous variables.  
- **Best Practice**: Smooth lines with sufficient data points.

### Visualization 10: Area Plot
**Explanation**  
Shows cumulative listings by `host_since` year.

```python
plt.figure(figsize=(8, 6))
df['host_year'] = df['host_since'].dt.year
year_counts = df['host_year'].value_counts().sort_index()
year_counts.cumsum().plot.area()
plt.title('Cumulative Listings by Host Start Year')
plt.xlabel('Year')
plt.ylabel('Cumulative Listings')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Growth of listings over time.  
- **Why It’s Useful**: Tracks market expansion.  
- **When to Use**: For cumulative trends.  
- **Best Practice**: Ensure time axis is clear.

### Visualization 11: Pair Plot
**Explanation**  
Explores relationships between normalized numerical variables.

```python
sns.pairplot(df[['price_normalized', 'number_of_reviews_normalized', 'minimum_nights_normalized', 'bedrooms_normalized']].dropna())
plt.suptitle('Pair Plot of Normalized Numerical Features', y=1.02)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Scatter plots and histograms for multiple variables.  
- **Why It’s Useful**: Quick way to spot correlations and distributions.  
- **When to Use**: For multivariate exploration.  
- **Common Mistake**: Including too many variables can overwhelm.

### Visualization 12: Density Plot
**Explanation**  
Shows `price_normalized` density by `room_type`.

```python
plt.figure(figsize=(8, 6))
for room in df['room_type'].unique():
    sns.kdeplot(data=df[df['room_type'] == room]['price_normalized'], label=room)
plt.title('Normalized Price Density by Room Type')
plt.xlabel('Price (Normalized)')
plt.ylabel('Density')
plt.legend()
plt.show()
```

**Why This Matters**  
- **What it Shows**: Smoothed price distributions per room type.  
- **Why It’s Useful**: Compares distributions without binning.  
- **When to Use**: For continuous variable density.  
- **Best Practice**: Use distinct colors for clarity.

### Visualization 13: Barh Plot
**Explanation**  
Shows average `price_normalized` by top 5 `neighbourhoods`.

```python
plt.figure(figsize=(8, 6))
top_neigh = df.groupby('neighbourhood')['price_normalized'].mean().nlargest(5)
top_neigh.plot.barh()
plt.title('Average Normalized Price by Top 5 Neighbourhoods')
plt.xlabel('Average Price (Normalized)')
plt.ylabel('Neighbourhood')
plt.show()
```

**Why This Matters**  
- **What it Shows**: Highest-priced neighborhoods.  
- **Why It’s Useful**: Identifies premium areas.  
- **When to Use**: For ranked categorical comparisons.  
- **Best Practice**: Horizontal bars improve readability for long labels.

### Visualization 14: Boxen Plot
**Explanation**  
Shows `price_normalized` distribution by `room_type` with enhanced quantiles.

```python
plt.figure(figsize=(8, 6))
sns.boxenplot(data=df, x='room_type', y='price_normalized')
plt.title('Normalized Price Distribution by Room Type (Boxen)')
plt.xlabel('Room Type')
plt.ylabel('Price (Normalized)')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Detailed quantile-based price spread.  
- **Why It’s Useful**: Better for large datasets than standard boxplots.  
- **When to Use**: For detailed distribution comparisons.  
- **Best Practice**: Ensure enough data for quantile accuracy.

### Visualization 15: Swarm Plot
**Explanation**  
Shows `price_normalized` distribution for a sample by `room_type`.

```python
plt.figure(figsize=(8, 6))
sns.swarmplot(data=df.sample(100), x='room_type', y='price_normalized')
plt.title('Normalized Price Swarm Plot by Room Type (Sample)')
plt.xlabel('Room Type')
plt.ylabel('Price (Normalized)')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Individual data points by category.  
- **Why It’s Useful**: Avoids overplotting for small samples.  
- **When to Use**: For small datasets or samples.  
- **Common Mistake**: Using with large datasets causes clutter.

### Visualization 16: Stacked Bar Plot
**Explanation**  
Shows `room_type` counts by top 5 `neighbourhoods`.

```python
plt.figure(figsize=(8, 6))
top_neigh = df['neighbourhood'].value_counts().head(5).index
pivot = df[df['neighbourhood'].isin(top_neigh)].pivot_table(index='neighbourhood', columns='room_type', aggfunc='size', fill_value=0)
pivot.plot.bar(stacked=True)
plt.title('Room Type Distribution by Top 5 Neighbourhoods')
plt.xlabel('Neighbourhood')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

**Why This Matters**  
- **What it Shows**: Composition of room types per neighborhood.  
- **Why It’s Useful**: Highlights area-specific listing patterns.  
- **When to Use**: For multi-category comparisons.  
- **Best Practice**: Limit categories to avoid cluttered stacks.

### Visualization 17: Interactive Scatter Plot (Plotly)
**Explanation**  
Interactive scatter plot of `price_normalized` vs. `number_of_reviews_normalized` with `room_type` color.

```python
import plotly.express as px

fig = px.scatter(df, x='number_of_reviews_normalized', y='price_normalized', color='room_type',
                 title='Interactive Normalized Price vs. Reviews by Room Type',
                 labels={'number_of_reviews_normalized': 'Number of Reviews (Normalized)', 'price_normalized': 'Price (Normalized)'})
fig.show()
```

**Why This Matters**  
- **What it Shows**: Price vs. reviews with interactive hover and zoom.  
- **Why It’s Useful**: Allows users to explore individual data points.  
- **When to Use**: For interactive exploration of relationships.  
- **Best Practice**: Ensure color contrast for accessibility.

### Visualization 18: Interactive Filter (ipywidgets)
**Explanation**  
Interactive bar plot allowing users to select `neighbourhood` for `room_type` counts.

```python
from ipywidgets import interact, Dropdown

def plot_room_types(neighbourhood):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df[df['neighbourhood'] == neighbourhood], x='room_type')
    plt.title(f'Room Type Distribution in {neighbourhood}')
    plt.xlabel('Room Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

interact(plot_room_types, neighbourhood=Dropdown(options=df['neighbourhood'].unique(), description='Neighbourhood:'))
```

**Why This Matters**  
- **What it Shows**: Room type distribution for a user-selected neighborhood.  
- **Why It’s Useful**: Enables dynamic exploration of data subsets.  
- **When to Use**: For user-driven analysis.  
- **Best Practice**: Ensure widget options are manageable (e.g., not too many unique values).

## 6. Predictive Modeling: Airbnb Price Prediction

**Explanation**  
We'll predict `price_normalized` using a `RandomForestRegressor`, split data into train/test sets, and evaluate with MAE, RMSE, and R². We'll use normalized features to ensure consistent scale.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prepare features (using normalized columns)
features = ['bedrooms_normalized', 'minimum_nights_normalized', 'number_of_reviews_normalized', 
            'room_type_encoded', 'host_duration_normalized']
X = df[features].fillna(0)  # Fill any remaining NaNs
y = df['price_normalized']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
```

**Why This Matters**  
- **MAE**: Average prediction error in normalized units.  
- **RMSE**: Penalizes larger errors, useful for model consistency.  
- **R²**: Proportion of variance explained (0.0 to 1.0, higher is better).  
- **Interpretation**: An R² of 0.7 means 70% of price variance is explained. Lower MAE/RMSE indicates better accuracy.  
- **Common Mistake**: Not splitting data can lead to overfitting.

## 7. Interactive Explorer

**Explanation**  
We'll create an interactive tool using `ipywidgets` to let users select variables and filter listings.

```python
from ipywidgets import interact, IntSlider, Dropdown

def explore_data(neighbourhood, max_price):
    # Scale max_price to normalized range
    max_price_normalized = (max_price - df['price'].min()) / (df['price'].max() - df['price'].min())
    filtered_df = df[(df['neighbourhood'] == neighbourhood) & (df['price_normalized'] <= max_price_normalized)]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_df, x='number_of_reviews_normalized', y='price_normalized', hue='room_type')
    plt.title(f'Price vs. Reviews in {neighbourhood} (Max Price: {max_price})')
    plt.xlabel('Number of Reviews (Normalized)')
    plt.ylabel('Price (Normalized)')
    plt.show()
    display(filtered_df[['name', 'price', 'room_type', 'neighbourhood']].head())

interact(explore_data,
         neighbourhood=Dropdown(options=df['neighbourhood'].unique(), description='Neighbourhood:'),
         max_price=IntSlider(min=0, max=int(df['price'].max()), step=10, value=200, description='Max Price:'))
```

**Why This Matters**  
- **What it Does**: Filters data by neighborhood and price, showing scatter plot and table.  
- **Why It’s Useful**: Empowers users to explore specific market segments.  
- **When to Use**: For user-driven data exploration.  
- **Best Practice**: Keep widgets simple to avoid overwhelming users.

## 8. Interactive Application Concept

**Explanation**  
We'll sketch a `voila`-ready app concept where users input features to predict `price_normalized`. This could help renters estimate costs or hosts set competitive prices.

```python
from ipywidgets import FloatText, Dropdown, Button, Output
from IPython.display import display

out = Output()

def predict_price(b):
    with out:
        out.clear_output()
        input_data = pd.DataFrame({
            'bedrooms_normalized': [(bedrooms.value - df['bedrooms'].min()) / (df['bedrooms'].max() - df['bedrooms'].min())],
            'minimum_nights_normalized': [(min_nights.value - df['minimum_nights'].min()) / (df['minimum_nights'].max() - df['minimum_nights'].min())],
            'number_of_reviews_normalized': [(reviews.value - df['number_of_reviews'].min()) / (df['number_of_reviews'].max() - df['number_of_reviews'].min())],
            'room_type_encoded': [room_type.value],
            'host_duration_normalized': [(host_duration.value - df['host_duration'].min()) / (df['host_duration'].max() - df['host_duration'].min())]
        })
        pred_normalized = model.predict(input_data)[0]
        # Convert back to original price scale
        pred = pred_normalized * (df['price'].max() - df['price'].min()) + df['price'].min()
        print(f"Predicted Price: ${pred:.2f}")

bedrooms = FloatText(value=1, description='Bedrooms:')
min_nights = FloatText(value=1, description='Min Nights:')
reviews = FloatText(value=10, description='Reviews:')
room_type = Dropdown(options=[('Entire home/apt', 0), ('Private room', 1), ('Shared room', 2)], description='Room Type:')
host_duration = FloatText(value=1, description='Host Years:')
button = Button(description='Predict Price')
button.on_click(predict_price)

display(bedrooms, min_nights, reviews, room_type, host_duration, button, out)
```

--- 
