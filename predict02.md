# The Ultimate Data Visualization Guide: Mastering Airbnb Data Insights

```python
# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify  # For treemaps
import joypy  # For ridgeline plots
from wordcloud import WordCloud  # For text visualization
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set global styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
color_palette = sns.color_palette("viridis")
```

## 📁 Part 1: Data Preparation

### Why Data Preparation Matters
Before visualization, we need clean, well-structured data. This step ensures our visualizations are accurate and meaningful.

```python
# Load and prepare the Airbnb dataset
print("🔍 Loading Airbnb dataset from GitHub...")
url = 'https://github.com/adelnehme/python-for-spreadsheet-users-webinar/blob/master/datasets/airbnb.csv?raw=true'
df = pd.read_csv(url)

# Data cleaning
print("\n🧹 Cleaning data...")
# Handle missing values
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review'] = pd.to_datetime(df['last_review']).fillna(pd.Timestamp('2019-01-01'))
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float).fillna(0)

# Handle outliers
price_q1 = df['price'].quantile(0.25)
price_q3 = df['price'].quantile(0.75)
price_iqr = price_q3 - price_q1
df['price'] = df['price'].clip(price_q1 - 1.5*price_iqr, price_q3 + 1.5*price_iqr)

# Feature engineering
df['last_review_year'] = df['last_review'].dt.year
df['price_per_night'] = df['price'] / df['minimum_nights']
df['days_since_last_review'] = (pd.Timestamp.now() - df['last_review']).dt.days

# Filter relevant data
df = df[df['price'] > 0]
df = df.dropna(subset=['neighbourhood_group', 'room_type'])

print("✅ Data preparation complete!")
print(f"Final dataset shape: {df.shape}")
```

## 📊 Part 2: Visualization Gallery (20+ Types)

### 1. Bar Chart: Category Comparison
```python
plt.figure(figsize=(12, 6))
sns.barplot(x='neighbourhood_group', y='price', data=df, estimator=np.median, errorbar=None, palette='viridis')
plt.title('Median Price by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()
print("When to use: Comparing values across categories")
```

### 2. Line Chart: Time Trends
```python
trend_df = df.groupby('last_review_year')['price'].median().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='last_review_year', y='price', data=trend_df, marker='o', linewidth=2.5)
plt.title('Median Price Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
print("When to use: Showing changes over time")
```

### 3. Histogram: Distribution Analysis
```python
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue', edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.axvline(df['price'].median(), color='red', linestyle='--', label='Median')
plt.legend()
plt.show()
print("When to use: Visualizing distribution of numerical data")
```

### 4. Box Plot: Distribution Comparison
```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='neighbourhood_group', y='price', data=df, palette='Set2', showfliers=False)
plt.title('Price Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()
print("When to use: Comparing distributions across groups")
```

### 5. Scatter Plot: Relationships
```python
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='latitude', data=df.sample(1000), 
                hue='price', size='price', sizes=(10, 200), 
                palette='viridis', alpha=0.7)
plt.title('Geographical Distribution of Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Price', bbox_to_anchor=(1.05, 1))
plt.show()
print("When to use: Exploring relationships between two numerical variables")
```

### 6. Heatmap: Correlation Analysis
```python
corr_df = df[['price', 'minimum_nights', 'number_of_reviews', 
             'reviews_per_month', 'availability_365']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()
print("When to use: Visualizing relationships between multiple variables")
```

### 7. Violin Plot: Detailed Distributions
```python
plt.figure(figsize=(12, 8))
sns.violinplot(x='neighbourhood_group', y='price', data=df, 
               inner='quartile', palette='muted', cut=0)
plt.title('Price Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.show()
print("When to use: Showing detailed distribution shapes")
```

### 8. Treemap: Hierarchical Data
```python
neighborhood_counts = df.groupby(['neighbourhood_group', 'room_type']).size().reset_index(name='counts')
plt.figure(figsize=(14, 8))
squarify.plot(sizes=neighborhood_counts['counts'],
              label=[f"{row['neighbourhood_group']}\n{row['room_type']}\n{row['counts']}" 
                     for _, row in neighborhood_counts.iterrows()],
              color=sns.color_palette('Spectral', len(neighborhood_counts)),
              alpha=0.7, text_kwargs={'fontsize': 10})
plt.title('Listings Distribution by Neighborhood and Room Type')
plt.axis('off')
plt.show()
print("When to use: Visualizing hierarchical part-to-whole relationships")
```

### 9. Ridgeline Plot: Distribution Comparison
```python
plt.figure(figsize=(12, 8))
fig, axes = joypy.joyplot(df, by='neighbourhood_group', column='price',
                          figsize=(10, 6), range_style='own', overlap=1,
                          colormap=sns.color_palette("crest", as_cmap=True))
plt.title('Price Distribution Across Neighborhoods', fontsize=16)
plt.show()
print("When to use: Comparing distributions across multiple groups")
```

### 10. Bubble Chart: Three-Dimensional Relationships
```python
bubble_df = df.groupby('neighbourhood_group').agg(
    avg_price=('price', 'median'),
    avg_reviews=('number_of_reviews', 'median'),
    count=('id', 'count')
).reset_index()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='avg_reviews', y='avg_price', size='count',
                sizes=(100, 1000), hue='neighbourhood_group',
                data=bubble_df, palette='tab10', alpha=0.8)
plt.title('Neighborhood Comparison: Price vs. Reviews')
plt.xlabel('Average Number of Reviews')
plt.ylabel('Median Price ($)')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
print("When to use: Showing relationships with three dimensions of data")
```

### 11. Word Cloud: Text Data Visualization
```python
text = " ".join(name for name in df['name'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=100).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Airbnb Listing Names', fontsize=16)
plt.show()
print("When to use: Visualizing text data frequencies")
```

### 12. Facet Grid: Multi-Panel Analysis
```python
g = sns.FacetGrid(df, col='room_type', row='neighbourhood_group',
                  height=4, aspect=1.2, margin_titles=True)
g.map_dataframe(sns.scatterplot, x='number_of_reviews', y='price', alpha=0.6)
g.set_axis_labels('Number of Reviews', 'Price ($)')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Price vs. Reviews by Neighborhood and Room Type', fontsize=16)
plt.show()
print("When to use: Comparing relationships across multiple categories")
```

### 13. Interactive Map: Geospatial Analysis
```python
sample_df = df.sample(500)
fig = px.scatter_mapbox(sample_df, lat='latitude', lon='longitude',
                        color='price', size='price', hover_name='name',
                        hover_data=['room_type', 'neighbourhood_group'],
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        zoom=10)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title="Airbnb Price Distribution in NYC")
fig.show()
print("When to use: Exploring geographical patterns in data")
```

### 14. Stacked Area Chart: Composition Over Time
```python
time_data = df.groupby(['last_review_year', 'room_type']).size().unstack().fillna(0)
plt.figure(figsize=(12, 6))
plt.stackplot(time_data.index, time_data.values.T,
              labels=time_data.columns, colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Listings by Room Type Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Listings')
plt.legend(title='Room Type', loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
print("When to use: Showing composition changes over time")
```

### 15. Swarm Plot: Distribution with Individual Points
```python
plt.figure(figsize=(12, 8))
sns.swarmplot(x='neighbourhood_group', y='price', data=df.sample(500),
              hue='room_type', palette='dark', size=3, alpha=0.7)
plt.title('Price Distribution with Individual Listings')
plt.xlabel('Neighborhood')
plt.ylabel('Price ($)')
plt.legend(title='Room Type', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.show()
print("When to use: Showing distribution while preserving individual data points")
```

## 📊 Part 3: Visualization Selection Guide

### When to Use Each Visualization Type
```python
# Create a comprehensive visualization guide
viz_guide = pd.DataFrame({
    'Visualization': ['Bar Chart', 'Line Chart', 'Histogram', 'Box Plot', 
                     'Scatter Plot', 'Heatmap', 'Violin Plot', 'Treemap',
                     'Ridgeline Plot', 'Bubble Chart', 'Word Cloud', 'Facet Grid',
                     'Interactive Map', 'Area Chart', 'Swarm Plot'],
    'Best For': [
        'Comparing categories',
        'Showing trends over time',
        'Distribution of numerical data',
        'Comparing distributions + outliers',
        'Relationships between variables',
        'Matrix/correlation visualization',
        'Detailed distribution comparison',
        'Hierarchical part-to-whole',
        'Comparing distributions across groups',
        'Three-dimensional relationships',
        'Text data visualization',
        'Multi-panel comparisons',
        'Geospatial analysis',
        'Cumulative trends',
        'Distribution with individual points'
    ],
    'When to Avoid': [
        'Many categories (>10)',
        'Non-continuous data',
        'Small datasets (<30 points)',
        'Showing exact distributions',
        'Large datasets (>10k points)',
        'Small matrices (<3 variables)',
        'Simple comparisons',
        'Non-hierarchical data',
        'Too many categories (>8)',
        'More than 3 dimensions',
        'Non-text data',
        'Limited screen space',
        'Non-geospatial data',
        'Non-ordered data',
        'Large datasets (>1000 points)'
    ],
    'Key Insight': [
        'Compare magnitudes',
        'Show trends/patterns',
        'Reveal data shape',
        'Identify outliers',
        'Find correlations',
        'See multivariate patterns',
        'Understand distribution shape',
        'Visualize hierarchies',
        'Compare density distributions',
        'Show size relationships',
        'Identify common terms',
        'Compare across categories',
        'See geographic patterns',
        'Show cumulative change',
        'See individual points'
    ]
})

print("📊 Ultimate Visualization Selection Guide:")
display(viz_guide)
```

### Visualization Decision Tree
```python
print("""
🌳 How to Choose the Right Visualization:

1. What is your main goal?
   a. Compare values → Bar Chart
   b. Show distribution → Histogram, Box Plot, Violin Plot
   c. Show relationships → Scatter Plot, Bubble Chart
   d. Show composition → Treemap, Area Chart
   e. Show trends → Line Chart, Area Chart
   f. Analyze geography → Interactive Map

2. How many variables?
   a. 1 variable → Histogram, Box Plot
   b. 2 variables → Scatter Plot, Line Chart
   c. 3 variables → Bubble Chart, 3D Plot
   d. 4+ variables → Facet Grid, Parallel Coordinates

3. What type of data?
   a. Categorical → Bar Chart, Treemap
   b. Numerical → Histogram, Box Plot
   c. Geographical → Interactive Map
   d. Temporal → Line Chart, Area Chart
   e. Text → Word Cloud

4. How much data?
   a. Small dataset → Swarm Plot, Box Plot
   b. Large dataset → Histogram, Heatmap
""")
```

## 🤖 Part 4: Predictive Modeling

### Why Build Predictive Models?
Models help us understand key drivers and predict future outcomes based on data patterns.

```python
# Prepare data for modeling
print("\n⚙️ Preparing data for price prediction model...")
model_df = df[['neighbourhood_group', 'room_type', 'latitude', 'longitude', 
               'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
               'calculated_host_listings_count', 'availability_365', 'price']].dropna()

X = model_df.drop('price', axis=1)
y = model_df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                    'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
categorical_features = ['neighbourhood_group', 'room_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_names = numeric_features + list(model.named_steps['preprocessor']
                                       .named_transformers_['cat']
                                       .get_feature_names_out(categorical_features))

importances = model.named_steps['regressor'].feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Top 10 Price Predictors')
plt.show()
```

## 🎮 Part 5: Interactive Price Predictor

```python
# Create input widgets
print("\n🎮 Creating Interactive Price Predictor...")

neighbourhood_dropdown = widgets.Dropdown(
    options=df['neighbourhood_group'].unique(),
    description='Neighborhood:',
    value='Manhattan'
)

room_type_dropdown = widgets.Dropdown(
    options=df['room_type'].unique(),
    description='Room Type:',
    value='Private room'
)

latitude_slider = widgets.FloatSlider(
    value=40.72,
    min=df['latitude'].min(),
    max=df['latitude'].max(),
    step=0.01,
    description='Latitude:',
    readout_format='.3f'
)

longitude_slider = widgets.FloatSlider(
    value=-74.00,
    min=df['longitude'].min(),
    max=df['longitude'].max(),
    step=0.01,
    description='Longitude:',
    readout_format='.3f'
)

min_nights_input = widgets.IntText(
    value=2,
    description='Min Nights:',
    min=1,
    max=365
)

availability_slider = widgets.IntSlider(
    value=180,
    min=0,
    max=365,
    description='Availability:'
)

calculate_button = widgets.Button(description="Predict Price", button_style='success')
output = widgets.Output()

# Set default values
defaults = {
    'number_of_reviews': df['number_of_reviews'].median(),
    'reviews_per_month': df['reviews_per_month'].median(),
    'calculated_host_listings_count': df['calculated_host_listings_count'].median()
}

# Prediction function
def predict_price(button):
    with output:
        clear_output()
        # Create input dictionary
        input_data = {
            'neighbourhood_group': neighbourhood_dropdown.value,
            'room_type': room_type_dropdown.value,
            'latitude': latitude_slider.value,
            'longitude': longitude_slider.value,
            'minimum_nights': min_nights_input.value,
            'number_of_reviews': defaults['number_of_reviews'],
            'reviews_per_month': defaults['reviews_per_month'],
            'calculated_host_listings_count': defaults['calculated_host_listings_count'],
            'availability_365': availability_slider.value
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        predicted_price = model.predict(input_df)[0]
        
        # Display results
        print("📋 Your Listing Details:")
        display(input_df)
        print(f"\n💲 Predicted Price: ${predicted_price:.2f} per night")
        
        # Show location on map
        fig = px.scatter_mapbox(
            input_df,
            lat="latitude",
            lon="longitude",
            zoom=12,
            height=400,
            width=700,
            color_discrete_sequence=["red"],
            title="Your Listing Location"
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()

# Connect button to function
calculate_button.on_click(predict_price)

# Display the widget panel
print("🔮 Adjust the features and click 'Predict Price' to see estimated value")
widgets_box = widgets.VBox([
    widgets.HBox([neighbourhood_dropdown, room_type_dropdown]),
    widgets.HBox([latitude_slider, longitude_slider]),
    widgets.HBox([min_nights_input, availability_slider]),
    calculate_button,
    output
])
display(widgets_box)

# Location reference guide
print("\n🗺️ NYC Location Reference:")
print("Manhattan: 40.78, -73.96")
print("Brooklyn: 40.65, -73.94")
print("Queens: 40.74, -73.79")
print("Bronx: 40.85, -73.87")
print("Staten Island: 40.58, -74.15")
```

## 📚 Part 6: Key Takeaways & Resources

### Data Visualization Best Practices
```python
print("""
🌟 10 Visualization Best Practices:

1. Know your audience: Tailor complexity to viewers
2. Choose the right chart: Match visualization to your question
3. Simplify: Remove unnecessary elements (chartjunk)
4. Use color strategically: Highlight important information
5. Label clearly: Always include titles and axis labels
6. Tell a story: Sequence visualizations logically
7. Highlight insights: Use annotations to emphasize key points
8. Maintain proportions: Avoid distorting data relationships
9. Provide context: Include comparisons and benchmarks
10. Iterate: Create multiple versions and get feedback

🛠️ Essential Python Visualization Libraries:
- Matplotlib: Foundation for all plotting
- Seaborn: High-level statistical visualizations
- Plotly: Interactive and web-based visualizations
- Geopandas: Geospatial data visualization
- WordCloud: Text data visualization

📚 Recommended Learning Resources:
1. Python Graph Gallery: https://www.python-graph-gallery.com/
2. From Data to Viz: https://www.data-to-viz.com/
3. Plotly Documentation: https://plotly.com/python/
4. Seaborn Tutorials: https://seaborn.pydata.org/tutorial.html
5. Matplotlib Cheat Sheet: https://matplotlib.org/cheatsheets/
""")
```

### Final Project Challenge
```python
print("""
🏆 Final Data Visualization Challenge:

Using the techniques learned in this guide:
1. Find a dataset on Kaggle (https://www.kaggle.com/datasets)
2. Perform comprehensive EDA with at least 5 different visualization types
3. Identify 3 key insights about the data
4. Build one interactive visualization
5. Create a predictive model for an important outcome

Share your project on GitHub with #DataVizMaster!
""")
```

### Quick Reference Guide
```python
# Create a quick reference table
quick_ref = pd.DataFrame({
    'Question Type': [
        'Comparison', 
        'Distribution', 
        'Relationship', 
        'Composition', 
        'Trend',
        'Geospatial',
        'Text Analysis'
    ],
    'Recommended Visualizations': [
        'Bar Chart, Treemap',
        'Histogram, Box Plot, Violin Plot, Ridgeline',
        'Scatter Plot, Bubble Chart, Heatmap',
        'Treemap, Pie Chart (sparingly), Stacked Bar',
        'Line Chart, Area Chart',
        'Interactive Map, Scatter Map',
        'Word Cloud, Bar Chart'
    ],
    'Best Library': [
        'Seaborn',
        'Seaborn/Matplotlib',
        'Plotly/Seaborn',
        'Matplotlib/Squarify',
        'Matplotlib',
        'Plotly/Geopandas',
        'WordCloud'
    ]
})

print("\n📋 Quick Reference Guide:")
display(quick_ref)
```
