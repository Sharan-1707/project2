import pandas as pd
import matplotlib.pyplot as plt
import os

def run_eda():
    # Define file paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "EDA_data.csv")
    
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n" + "=" * 60)
    print("2. IDENTIFYING MISSING VALUES")
    print("=" * 60)
    missing_counts = df.isnull().sum()
    print("Missing value counts per column:")
    print(missing_counts)
    
    # 3. Handling Missing Values
    print("\n" + "=" * 60)
    print("3. DATA IMPUTATION / CLEANING")
    print("=" * 60)
    
    # Age - impute with median
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    print(f"- Imputed missing Age values using median age: {median_age:.1f}")
    
    # Spending - impute with mean
    mean_spending = df['Spending'].mean()
    df['Spending'] = df['Spending'].fillna(mean_spending)
    print(f"- Imputed missing Spending values using mean spending: ${mean_spending:.2f}")
    
    # Visits_Per_Month - impute with median
    median_visits = df['Visits_Per_Month'].median()
    df['Visits_Per_Month'] = df['Visits_Per_Month'].fillna(median_visits)
    print(f"- Imputed missing Visits_Per_Month values using median visits: {median_visits:.1f}")
    
    print("\nVerify missing values after imputation:")
    print(df.isnull().sum())
    
    # 4. Outlier Detection
    print("\n" + "=" * 60)
    print("4. OUTLIER DETECTION")
    print("=" * 60)
    # Detect Age outliers (> 100)
    age_outliers = df[df['Age'] > 100]
    print(f"Found {len(age_outliers)} age outliers (> 100 years):")
    if not age_outliers.empty:
        print(age_outliers[['Customer_ID', 'Age', 'City', 'Spending']])
        
    # Detect Spending outliers (using IQR method)
    q1 = df['Spending'].quantile(0.25)
    q3 = df['Spending'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    spending_outliers = df[(df['Spending'] < lower_bound) | (df['Spending'] > upper_bound)]
    print(f"\nFound {len(spending_outliers)} spending outliers (outside range ${lower_bound:.2f} to ${upper_bound:.2f}):")
    if not spending_outliers.empty:
        print(spending_outliers[['Customer_ID', 'Age', 'City', 'Spending']])
        
    # 5. Descriptive Statistics
    print("\n" + "=" * 60)
    print("5. DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe())
    
    # 6. Correlation Analysis
    print("\n" + "=" * 60)
    print("6. CORRELATION ANALYSIS (Numeric Columns)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_cols.corr()
    print(correlation_matrix)
    
    # 7. Visualizations
    print("\n" + "=" * 60)
    print("7. GENERATING VISUAL PLOTS (using Matplotlib)")
    print("=" * 60)
    
    # Set default style parameters
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Plot 1: Spending Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['Spending'].to_numpy(), bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title("Distribution of Spending")
    plt.xlabel("Spending Amount ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    spending_plot_path = os.path.join(script_dir, "spending_distribution.png")
    plt.savefig(spending_plot_path)
    plt.close()
    print(f"- Saved spending distribution plot to: {spending_plot_path}")
    
    # Plot 2: Correlation Heatmap using plt.imshow
    plt.figure(figsize=(7, 6))
    corr_data = correlation_matrix.to_numpy()
    cols = correlation_matrix.columns.tolist()
    
    im = plt.imshow(corr_data, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation Coefficient")
    
    # Set labels
    plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
    plt.yticks(range(len(cols)), cols)
    
    # Add annotation values in cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_data[i, j]
            # Choose text color based on cell background intensity
            text_color = "white" if abs(val) > 0.5 else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
            
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_plot_path = os.path.join(script_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_plot_path)
    plt.close()
    print(f"- Saved correlation heatmap to: {heatmap_plot_path}")
    
    # Plot 3: Age Boxplot (showing outliers)
    plt.figure(figsize=(8, 4))
    plt.boxplot(df['Age'].to_numpy(), vert=False, patch_artist=True, 
                boxprops=dict(facecolor="lightgreen", color="darkgreen"),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='orange', markersize=8))
    plt.title("Box Plot of Age (Detecting Outliers)")
    plt.xlabel("Age")
    plt.tight_layout()
    age_plot_path = os.path.join(script_dir, "age_boxplot.png")
    plt.savefig(age_plot_path)
    plt.close()
    print(f"- Saved age boxplot to: {age_plot_path}")
    
    # Plot 4: City-wise Average Spending Bar Chart
    plt.figure(figsize=(10, 5))
    city_spending = df.groupby('City')['Spending'].mean().reset_index().sort_values(by='Spending', ascending=False)
    
    cities = city_spending['City'].to_numpy()
    avg_spend = city_spending['Spending'].to_numpy()
    
    # Generate bar chart
    bars = plt.bar(cities, avg_spend, color="#4F46E5", edgecolor="black", alpha=0.8)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f"${height:.0f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
                 
    plt.title("Average Spending by City")
    plt.xlabel("City")
    plt.ylabel("Average Spending ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    city_plot_path = os.path.join(script_dir, "city_spending.png")
    plt.savefig(city_plot_path)
    plt.close()
    print(f"- Saved city spending plot to: {city_plot_path}")
    
    print("\nEDA Completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
