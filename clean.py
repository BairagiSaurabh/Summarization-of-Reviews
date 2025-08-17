import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from pathlib import Path

def downloads():
    """Download required NLTK data"""
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def setup_cleaning():
    """Setup lemmatizer and stopwords for cleaning"""
    downloads()
    
    lemma = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    
    # Remove specific stopwords that we want to keep
    stopwords_to_keep = [
        'not', 'but', 'because', 'against', 'between', 'up', 'down', 
        'in', 'out', 'once', 'before', 'after', 'few', 'more', 'most', 
        'no', 'nor', 'same', 'some'
    ]
    
    for word in stopwords_to_keep:
        if word in all_stopwords:
            all_stopwords.remove(word)
    
    return lemma, all_stopwords

def clean_aspect_spacy(reviews, lemma, all_stopwords):
    """
    This function removes punctuations, stopwords and other non alpha numeric characters.
    We expand the contractions and replace some words by an empty string
    """
    if pd.isna(reviews) or reviews == "":
        return ""
    
    statement = reviews.lower().strip()
    statement = statement.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("*****", " ") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("doesn't", "does not")
    statement = re.sub('[^a-zA-Z]', ' ', statement)  # replacing whatever isn't letters by an empty string
    statement = statement.split()  # forming list of words in a given review
    final_statement = [lemma.lemmatize(word) for word in statement if not word in set(all_stopwords)]
    final_statement_ = ' '.join(final_statement)  # joining the words and forming the review again without stopwords
    return final_statement_

def clean_csv_reviews(csv_path):
    """
    Clean reviews in a CSV file and save as clean_{original_name}.csv
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        str: Path to the cleaned CSV file
    """
    print(f"Processing: {csv_path}")
    
    # Setup cleaning components
    lemma, all_stopwords = setup_cleaning()
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    # Find review text column (look for common names)
    review_column = None
    possible_names = ['review_text', 'review', 'text', 'Review', 'Review_Text', 'reviews']
    
    for col in df.columns:
        if col in possible_names:
            review_column = col
            break
    
    if review_column is None:
        print(f"Could not find review column. Available columns: {list(df.columns)}")
        print("Please specify which column contains the review text.")
        return None
    
    print(f"Found review column: '{review_column}'")
    
    # Clean the reviews
    print("Cleaning reviews...")
    df[review_column] = df[review_column].apply(
        lambda x: clean_aspect_spacy(x, lemma, all_stopwords)
    )
    
    # Create output filename
    input_path = Path(csv_path)
    output_filename = f"clean_{input_path.name}"
    output_path = input_path.parent / output_filename
    
    # Save cleaned CSV
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Cleaned CSV saved as: {output_path}")
        
        # Show some stats
        non_empty_reviews = df[df[review_column].str.strip() != ''].shape[0]
        print(f"Successfully cleaned {non_empty_reviews} non-empty reviews")
        
        return str(output_path)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None

def process_all_csvs_in_directory(directory="."):
    """
    Process all CSV files in the given directory
    
    Args:
        directory (str): Directory path (default: current directory)
    """
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv') and not file.startswith('clean_'):
            csv_files.append(os.path.join(directory, file))
    
    if not csv_files:
        print("No CSV files found in the directory")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    print("\nProcessing files...")
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        result = clean_csv_reviews(csv_file)
        if result:
            print(f"✓ Successfully processed: {csv_file}")
        else:
            print(f"✗ Failed to process: {csv_file}")

# Example usage for specific files from your image
def clean_specific_files():
    """Clean the specific CSV files visible in your image"""
    files_to_clean = [
        "iphone_15.csv",
        "jbl_earbuds.csv", 
        "macbook.csv",
        "motorola_moto_G85.csv"
    ]
    
    print("Cleaning specific files from your directory...")
    for file in files_to_clean:
        if os.path.exists(file):
            print(f"\n{'='*50}")
            result = clean_csv_reviews(file)
            if result:
                print(f"✓ Successfully cleaned: {file}")
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    # Method 1: Clean all CSV files in current directory
    print("Option 1: Process all CSV files in current directory")
    print("Option 2: Process specific files from your image")
    
    choice = input("Enter 1 or 2 (or press Enter for option 1): ").strip()
    
    if choice == "2":
        clean_specific_files()
    else:
        process_all_csvs_in_directory()
    
    print("\n" + "="*50)
    print("Cleaning complete!")