import pandas as pd
import numpy as np
import os

# Sample sales pitches grouped by semantic similarity
healthcare_pitches = [
    "What specific benefits are you looking for?",
    "What are your healthcare coverage needs?",
    "Do you have any upcoming procedures that need coverage?",
    "Do you currently reside in a nursing home, assisted living, or at home?",
    "Are there any specific benefits you are looking for?",
    "Do you make your own health care decisions?",
    "Do you have any upcoming procedures?",
    "What kind of healthcare coverage are you interested in?",
    "Which specific benefits matter most to you?",
    "Are you looking for any particular healthcare benefits?",
]

pricing_pitches = [
    "What is your budget for monthly premiums?",
    "How much are you currently paying for healthcare?",
    "What would you consider a reasonable monthly cost?",
    "Is cost a major factor in your decision?",
    "How much are you willing to spend each month?",
    "What's your comfort level for out-of-pocket expenses?",
    "Are you concerned about deductible amounts?",
    "How important is pricing in your decision?",
    "Would you prefer lower premiums with higher deductibles?",
    "What price point works best for your situation?",
]

current_coverage_pitches = [
    "Who is your current healthcare provider?",
    "How long have you been with your current provider?",
    "What do you like about your current coverage?",
    "What would you change about your current plan?",
    "Are you satisfied with your current healthcare provider?",
    "What aspects of your current coverage work well for you?",
    "Why are you looking to change your current coverage?",
    "How would you rate your current provider?",
    "What features of your current plan do you want to keep?",
    "Does your current plan cover all your healthcare needs?",
]

def create_sample_data():
    # Combine all pitches
    all_pitches = healthcare_pitches + pricing_pitches + current_coverage_pitches
    
    # Create sales stages
    stages = ["Needs Analysis", "Presentation", "Closing", "Follow-up"]
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    data = {
        'phrase': all_pitches,
        'stage': np.random.choice(stages, size=len(all_pitches)),
        'freq': np.random.randint(10, 100, size=len(all_pitches)),
    }
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Generate success count and rate
    df['success'] = np.random.randint(0, df['freq'], size=len(df))
    df['success_rate'] = df['success'] / df['freq']
    
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save to Excel
    excel_path = 'data/sample_sales_pitches.xlsx'
    df.to_excel(excel_path, index=False)
    
    print(f"Sample data created and saved to {excel_path}")
    return excel_path

if __name__ == "__main__":
    create_sample_data() 