# College Teaching Self-Efficacy Scale (CTSES) Analysis

## Project Overview
This project analyzes College Teaching Self-Efficacy Scale (CTSES) data by converting ratings from a 6-point to 12-point Likert scale and exploring the relationships between teaching confidence and frequency of teaching behaviors.

## Dataset
- **44 teaching activities** across various domains
- **Dual ratings** for each activity: Confidence and Frequency
- **Original scale**: 6-point Likert scale  
- **Converted scale**: 12-point Likert scale using formula: (rating × 2) - 1

## Analysis Features

### Data Transformation
- Converts string representations of rating lists to Python lists
- Applies mathematical transformation to expand scale range
- Handles multiple ratings per item by calculating averages
- Creates separate records for confidence and frequency analyses

### Statistical Analysis
- **Descriptive statistics** by rating type (confidence vs. frequency)
- **Gap analysis** identifying items where confidence exceeds frequency and vice versa
- **Category-based analysis** grouping items into teaching domains:
  - Course Planning
  - Student Engagement  
  - Classroom Management
  - Assessment
  - Feedback and Reflection
  - Student Support
  - Content Expertise

### Visualizations
1. **Distribution Histograms**: Side-by-side comparison of confidence and frequency rating patterns
2. **Category Heatmap**: Average ratings across teaching domains with color-coded intensity
3. **Scatter Plot**: Confidence vs. frequency relationship with gap analysis and outlier identification

## Key Findings

### Rating Distributions
- **Confidence ratings**: Cluster around 7-9 on 12-point scale (moderate to high self-efficacy)
- **Frequency ratings**: More variable distribution (1-11 range) with multiple peaks

### Category Analysis
- **Highest confidence**: Feedback and Reflection (8.33), Classroom Management (8.00)
- **Lowest confidence**: Course Planning (6.38)
- **Frequency patterns**: Generally align with confidence but show notable gaps in some areas

### Confidence-Frequency Relationship
- Strong positive correlation between confidence and practice frequency
- Several outliers identified (Items 34, 35, 36) showing significant confidence-frequency gaps
- Reference line analysis reveals where confidence exceeds actual practice

## Technical Implementation

### Python Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Mathematical operations and statistical calculations
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Advanced statistical visualizations
- **ast**: Converting string representations to Python objects

### Skills Demonstrated
- Data cleaning and transformation
- Statistical analysis and descriptive statistics
- Multi-panel visualization creation
- Categorical data analysis
- Gap analysis and comparative statistics
- Publication-quality plot formatting

## Files Generated
- `ctses-data.csv` - Original dataset
- `ctses_12point_likert_scale.csv` - Transformed dataset with 88 items (44 × 2 rating types)
- `ctses_confidence_frequency_comparison.csv` - Item-by-item comparison analysis
- `ctses_category_analysis.csv` - Category-level summary statistics
- Visualization files:
  - `ctses_12point_distribution.png` - Distribution histograms
  - `confidence_vs_frequency.png` - Relationship scatter plot
  - `category_heatmap.png` - Categorical analysis heatmap

## Code Structure
The analysis follows a logical progression:
1. Data input and DataFrame creation
2. String-to-list conversion for rating data
3. Scale transformation and statistical calculation
4. Data restructuring for dual-rating analysis
5. Comparative analysis and gap identification
6. Categorical grouping and thematic analysis
7. Comprehensive visualization suite

## Usage
Run `CTSES_ScD.py` to reproduce the complete analysis. The script is designed to be executed sequentially in a Jupyter notebook environment or as a standalone Python script.

## Applications
This analysis approach could be applied to:
- Other psychometric scale transformations
- Educational assessment data analysis
- Self-efficacy research in various domains
- Confidence-behavior gap analysis in professional development

## Future Enhancements
- Interactive dashboard development with Streamlit
- Statistical significance testing for category differences
- Correlation analysis with demographic variables
- Longitudinal analysis if repeated measures available
