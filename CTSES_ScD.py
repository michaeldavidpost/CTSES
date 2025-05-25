#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[3]:


# create a dictionary with the data


# In[24]:


data = {
    "Item": list(range(1, 45)), # items 1-44
    "Activity": [
        "Specify the learning goals that I expect my students to attain?",
        "Actively engage my students in the learning activities that I include in my teaching plan/syllabus?",
        "Create a positive classroom climate for learning?",
        "Reflect on my teaching practice with the aim of making appropriate improvements?",
        "Develop different assessment methods depending on the learning goals I want to check in my students?",
        "Evaluate the effectiveness of my own teaching in light of my students' feedback to me?",
        "Promote student participation in my classes?",
        "Use different evaluation methods?",
        "Prepare the teaching materials I will use?",
        "Ensure that my students resolve the difficulties they encounter while learning?",
        "Promote a positive attitude towards learning in my students?",
        "Adapt my teaching practices in response to my students' evaluations of my teaching?",
        "Evaluate accurately my students' academic capabilities?",
        "Decide on the most appropriate evaluation method for a particular course?",
        "Ensure that my students consider themselves capable of learning the material in my class?",
        "Employ systematic methods that permit me to assess my own teaching?",
        "Give my students feedback about their progress?",
        "Clearly identify my course objectives?",
        "Maintain high academic expectations?",
        "Use information derived from my own self-reflection to improve my teaching?",
        "Adequately grade my students' exams and assignments?",
        "Adapt to the needs of my students (motivation, interest, prior knowledge, etc.) when planning my courses?",
        "Think of my students as active learners, which is to say knowledge constructors rather than information receivers?",
        "Provide support and encouragement to students who are having difficulty learning?",
        "Update my knowledge of the subject I am teaching?",
        "Provide my students with detailed feedback about their academic progress?",
        "Modify and adapt my syllabus if my students' needs require it?",
        "Permit my students to prepare and/or develop some of the course units?",
        "Calmly handle any problems that may arise in the classroom?",
        "Develop my teaching skills using various means (attending conferences, reading about pedagogy, talking to other professionals…)?",
        "Use formative assessment to gather information about my students' academic progress?",
        "Encourage my students to ask questions during class?",
        "Make students aware that I have a personal investment in them and in their learning?",
        "Evaluate the degree to which my course objectives have been met?",
        "Design the structure and content of each class?",
        "Let students take initiative for their own learning?",
        "Show my students respect through my actions?",
        "Be flexible in my teaching even if I must alter my plans?",
        "Make students aware of the relevance of what they are learning?",
        "Master the material that I cover in class?",
        "Promote my students' confidence in themselves?",
        "Make my students feel that their academic success is due to their own efforts?",
        "Spend the time necessary to plan my classes?",
        "Select the appropriate materials for each class?"
    ],
    "Confidence_Ratings": [
        "[4]", "[4]", "[4]", "[5]", "[4]", "[5]", "[5]", "[4, 5]", "[5]", "[5]",
        "[5]", "[5]", "[4]", "[3]", "[4]", "[3]", "[4]", "[4, 5]", "[4]", "[5]",
        "[5]", "[3]", "[4]", "[4]", "[5]", "[4]", "[3]", "[2]", "[4]", "[5]",
        "[3, 4]", "[5]", "[5]", "[5]", "[3, 4]", "[3]", "[6]", "[4]", "[4]", "[5, 6]",
        "[4]", "[3]", "[4]", "[3, 4]"
    ],
    "Frequency_Ratings": [
        "[4]", "[4]", "[5]", "[6]", "[4]", "[4, 5]", "[4, 5]", "[4]", "[5]", "[4, 5]",
        "[5]", "[6]", "[3]", "[3]", "[3]", "[2]", "[4]", "[5]", "[5]", "[6]",
        "[5]", "[3]", "[3]", "[3, 4]", "[6]", "[3, 4]", "[3, 4]", "[1]", "[4, 5]", "[5]",
        "[3, 4]", "[6]", "[5]", "[3]", "[1]", "[1]", "[6]", "[3, 4]", "[4]", "[4, 5]",
        "[3]", "[2, 3]", "[4]", "[2, 3]"
    ]
}


# In[26]:


# create a dataframe from the dictionary
df = pd.DataFrame(data)

#diplay the first few rows to verify
df.head()


# In[28]:


# define the file path where you want to save the CSV
file_path = 'ctses-data.csv'

# save the dataframe as a csv file
df.to_csv(file_path, index=False)

print(f"CSV file has been saved to: {file_path}")


# In[30]:


# read the CSV file back in to verify it was saved correctly
verification_df = pd.read_csv(file_path)

# display the first few rows
verification_df.head()


# In[32]:


# convert string representations of lists to actual python lists
import ast

# create a copy of the dataframe to work with
working_df = verification_df.copy()

# convert string lists to actual python lists
working_df['Confidence_Ratings'] = working_df['Confidence_Ratings'].apply(ast.literal_eval)
working_df['Frequency_Ratings'] = working_df['Frequency_Ratings'].apply(ast.literal_eval)

# show a sample to verify the conversion worked
working_df.head()


# In[36]:


# define function to convert 6-point ratings to 12-point scale
def convert_to_12point(ratings_list):
    """Convert a list of 6-point scales to a 12-point scale by formula (rating*2)-1"""
    converted = [(rating * 2) - 1 for rating in ratings_list]
    return np.mean(converted) # return average if multiple ratings

# apply the conversion
working_df['Confidence_12point'] = working_df['Confidence_Ratings'].apply(convert_to_12point)
working_df['Frequency_12point'] = working_df['Frequency_Ratings'].apply(convert_to_12point)

# display the results
working_df[['Item', 'Activity', 'Confidence_12point', 'Frequency_12point']].head(10)


# In[54]:


# create a new dataframe with separate rows for confidence and frequency items
likert_rows = []

# first, add all confidence items
for idx, row in working_df.iterrows():
    confidence_ratings = row['Confidence_Ratings']
    # simply use string conversion without any conditional
    original_ratings_str = str(row['Confidence_Ratings']).replace("array", "").replace("dtype=object", "")
    
    likert_rows.append({
        'Item': f"{row['Item']}a",
        'Prompt': f"Confidence: {row['Activity']}",
        'Rating_12point': row['Confidence_12point'],
        'Original_Ratings': row['Confidence_12point'], 
        'Type': 'Confidence'
    })

# then, add all frequency items
for idx, row in working_df.iterrows():
    frequency_ratings = row['Frequency_Ratings']
    # Simply use string conversion without any conditional
    original_ratings_str = str(row['Frequency_Ratings']).replace("array", "").replace("dtype=object", "")
    
    likert_rows.append({
        'Item': f"{row['Item']}b",
        'Prompt': f"Frequency: {row['Activity']}",
        'Rating_12point': row['Frequency_12point'],
        'Original Ratings': row['Frequency_Ratings'], 
        'Type': 'Frequency'
    })

# create the final 12-point likert scale dataframe
likert_df = pd.DataFrame(likert_rows)

# show the first few rows of the new format
print(f"Original number of items: {len(working_df)}")
print(f"New number of items in 12-point scale: {len(likert_df)}")
likert_df.head(10)


# In[56]:


# Let's see what types we're working with
for idx, row in working_df.head().iterrows():
    print(f"Row {idx}:")
    print(f"  Confidence_Ratings type: {type(row['Confidence_Ratings'])}")
    print(f"  Confidence_Ratings value: {row['Confidence_Ratings']}")
    print(f"  Frequency_Ratings type: {type(row['Frequency_Ratings'])}")
    print(f"  Frequency_Ratings value: {row['Frequency_Ratings']}")
    print("-" * 50)


# In[60]:


# Create a new DataFrame with separate rows for confidence and frequency items
likert_rows = []

# First, add all confidence items - WITHOUT the problematic column
for idx, row in working_df.iterrows():
    likert_rows.append({
        'Item': f"{row['Item']}a",
        'Prompt': f"Confidence: {row['Activity']}",
        'Rating_12point': row['Confidence_12point'],
        'Type': 'Confidence'
    })

# Then, add all frequency items - WITHOUT the problematic column
for idx, row in working_df.iterrows():
    likert_rows.append({
        'Item': f"{row['Item']}b", 
        'Prompt': f"Frequency: {row['Activity']}",
        'Rating_12point': row['Frequency_12point'],
        'Type': 'Frequency'
    })

# Create the DataFrame first
likert_df = pd.DataFrame(likert_rows)

# Then add the Original_Ratings column separately
confidence_ratings = [repr(row['Confidence_Ratings']) for _, row in working_df.iterrows()]
frequency_ratings = [repr(row['Frequency_Ratings']) for _, row in working_df.iterrows()]
all_ratings = confidence_ratings + frequency_ratings

# Add as a new column
likert_df['Original_Ratings'] = all_ratings

# Show the first few rows
likert_df.head(10)


# In[64]:


# save the 12-point likert scale data to csv
likert_df.to_csv('ctses_12point_likert.csv', index=False)
print("12-point Likert scale data saved to: ctses_12point_likert.csv")


# In[66]:


# calculate basic statistics grouped by type (confidence vs frequency)
stats_by_type = likert_df.groupby('Type')['Rating_12point'].agg([
    'mean', 'median', 'std', 'min', 'max'
])

print("Statistics by item type:")
print(stats_by_type)

# calculate the range of ratings used
likert_df['Rating_12point'].describe()


# In[74]:


# create visualizations of the 12-point scale data
import matplotlib.pyplot as plt

# set up the figure
plt.figure(figsize=(12, 8))

# plot confidence ratings
confidence_data = likert_df[likert_df['Type'] == 'Confidence']

# modified histogram parameters
plt.subplot(2, 1, 1)
# create bins explicitly centered on each integer from 1-12
bins = np.arange(0.5, 12.5, 1)  # Creates bins at 0.5, 1.5, 2.5, ..., 11.5, 12.5
plt.hist(confidence_data['Rating_12point'], bins=bins,
         alpha=0.7, color='blue', edgecolor='black')
plt.title('Confidence Ratings Distribution (12-point scale)', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
# set x-ticks at integer values
plt.xticks(range(1, 13))
# restrict x-limits to see only the rating range
plt.xlim(0.5, 12.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)


# plot frequency ratings
frequency_data = likert_df[likert_df['Type'] == 'Frequency']

# modified histogram parameters
plt.subplot(2, 1, 2)
# create bins explicitly centered on each integer from 1-12
bins = np.arange(0.5, 12.5, 1)  # Creates bins at 0.5, 1.5, 2.5, ..., 11.5, 12.5
plt.hist(frequency_data['Rating_12point'], bins=bins,
         alpha=0.7, color='green', edgecolor='black')
plt.title('Frequency Ratings Distribution (12-point scale)', fontsize=14)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
# set x-ticks at integer values
plt.xticks(range(1, 13))
# restrict x-limits to see only the rating range
plt.xlim(0.5, 12.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ctses_12point_distribution.png')
plt.show()


# In[85]:


# create a dataframe to compare confidence and frequency for each item
comparison_df = pd.DataFrame()

# extract item numbers (without the 'a' or 'b' suffix)
confidence_items = confidence_data['Item'].str.replace('a', '', regex=False)
frequency_items = frequency_data['Item'].str.replace('b', '', regex=False)

# make sure we're comparing the same items
assert all(confidence_items.values == frequency_items.values)

# create the comparison DataFrame
comparison_df['Item'] = confidence_items
comparison_df['Activity'] = confidence_data['Prompt'].str.replace('Confidence: ', '', regex=False)
comparison_df['Confidence_Rating'] = confidence_data['Rating_12point'].values
comparison_df['Frequency_Rating'] = frequency_data['Rating_12point'].values
comparison_df['Gap'] = comparison_df['Confidence_Rating'] - comparison_df['Frequency_Rating']

# sort by the gap between confidence and frequency
comparison_df = comparison_df.sort_values('Gap', ascending=False)

# display items with largeest positive gaps (more confidence than frequent)
print("Items with HIGHER confidence than frequency:")
print(comparison_df.head(5)[['Item', 'Activity', 'Confidence_Rating', 'Frequency_Rating', 'Gap']])

# display items with largest negative gaps (more frequent than confident)
print("\nItems with HIGHER frequency than confidence:")
print(comparison_df.tail(5)[['Item', 'Activity', 'Confidence_Rating', 'Frequency_Rating', 'Gap']])

# display items with smallest gaps (confidence matches frequency)
middle_index = len(comparison_df) // 2
print("\nItems where confidence and frequency are most similar:")
print(comparison_df.iloc[middle_index-2:middle_index+3][['Item', 'Activity', 'Confidence_Rating', 'Frequency_Rating', 'Gap']])


# In[89]:


# create a scatter plot of confidence vs. frequency
plt.figure(figsize=(10,8))
plt.scatter(comparison_df['Confidence_Rating'], comparison_df['Frequency_Rating'],
            alpha=0.7, s=80, c=comparison_df['Gap'], cmap='coolwarm')

# add reference line (y=x) where confidence equals frequency
plt.plot([1, 11], [1, 11], 'k--', alpha=0.5)

plt.colorbar(label='Gap (Confidence - Frequency)')
plt.xlabel('Confidence Rating (12-point scale)', fontsize=12)
plt.ylabel('Frequency Rating (12-point scale)', fontsize=12)
plt.title('Relationship Between Teaching Confidence and Frequency', fontsize=14)
plt.grid(alpha=0.3)

# set equal axes to make the reference line meaningful
plt.axis('equal')
plt.xlim(0, 12)
plt.ylim(0, 12)

# add some item labels for points with large gaps
for i, row in comparison_df.iterrows():
    if abs(row['Gap']) > 3: # label points with large gaps
        plt.annotate(row['Item'],
                     (row['Confidence_Rating'], row['Frequency_Rating']), 
                     xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('confidence_vs_frequency.png')
plt.show()


# In[97]:


# optional: group items by themes/categories
# this requires manual categorization based on your knowledge of the ctses

# example grouping (you would customize this based on your understanding of the scale)
categories = {
    'Course Planning': [1, 5, 18, 22, 27, 35, 43, 44], 
    'Student Engagement': [2, 7, 15, 23, 32, 36, 39, 41], 
    'Classroom Management': [3, 29, 37, 38], 
    'Assessment': [8, 13, 14, 17, 21, 26, 31, 34], 
    'Feedback and Reflection': [4, 6, 12, 16, 20, 30], 
    'Student Support': [10, 11, 24, 33, 42], 
    'Content Expertise': [9, 19, 25, 28, 40]
}

# create a new column for category
comparison_df['Category'] = 'Uncategorized'
for category, items in categories.items():
    comparison_df.loc[comparison_df['Item'].astype(int).isin(items), 'Category'] = category

# create a pivot table of average ratings by category
category_pivot = comparison_df.pivot_table(
    index='Category', 
    values=['Confidence_Rating', 'Frequency_Rating', 'Gap'],
    aggfunc='mean'
).sort_values('Gap', ascending=False)

print("Average ratings by category:")
print(category_pivot)

# create a heatmap of categories
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(category_pivot[['Confidence_Rating', 'Frequency_Rating']], 
            annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Average Ratings by Teaching Category', fontsize=14)
plt.tight_layout()
plt.savefig('category_heatmap.png')
plt.show()


# In[101]:


# save the 12-point likert scale data
likert_df.to_csv('ctses_12point_likert_scale.csv', index=False)

# save the comparison data
comparison_df.to_csv('ctses_confidence_frequency_comparison.csv', index=False)

# save category analysis if performed
if 'Category' in comparison_df.columns:
    category_pivot.to_csv('ctses_category_analysis.csv')

    print("All files have been saved successfully!")


# In[ ]:




