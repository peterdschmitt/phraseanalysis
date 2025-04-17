# Sales Pitch Semantic Analysis Tool

This Streamlit application helps analyze and group sales pitches based on semantic similarity, tracking their frequency and success probability.

## Features

- Import Excel files containing sales pitches with metrics
- Automatically group similar sentences using semantic analysis
- Track frequency and success probability for each pitch
- Visualize success probability distributions
- Manually create and modify groups
- Identify the best-performing pitch in each group

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Excel File Format

Your Excel file should contain the following columns:
- `sentence`: The sales pitch text
- `frequency`: How often the pitch has been used
- `success_probability`: The success rate of the pitch (0-1)

## Usage

1. Upload your Excel file using the file uploader
2. Adjust the number of semantic groups using the slider
3. View automatically generated groups in the "Automatic Groups" tab
4. Create and manage custom groups in the "Manual Regrouping" tab
5. Move sentences between groups as needed

## Notes

- The semantic grouping uses the BERT model for natural language understanding
- Groups are limited to 50 sentences maximum
- The "best" sentence in each group is determined by combining frequency and success probability 