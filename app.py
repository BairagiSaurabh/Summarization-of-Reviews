import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import bs4
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
from dateutil.relativedelta import relativedelta
import en_core_web_sm
from aspect_extraction import apply_extraction
from scrape import scrape_amazon_reviews, extract_asin_from_url
from transformers import pipeline

# Load pipeline
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

st.set_page_config(page_title='Product Summarization', layout='wide')
st.title('Product Review Summarisation')

# Available CSV files for database option
DATABASE_FILES = {
    "iPhone 15": "iphone_15.csv",
    "JBL Earbuds": "jbl_earbuds.csv", 
    "MacBook": "macbook.csv",
    "Motorola Moto G85": "motorola_moto_G85.csv"
}

def load_data_from_database(selected_product):
    """Load data from selected CSV file"""
    try:
        file_path = DATABASE_FILES[selected_product]
        df_amazon = pd.read_csv(file_path)
        df_amazon.dropna(inplace=True)
        return df_amazon
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def scrape_fresh_data(url, page_number, pages_to_extract):
    """Scrape fresh data from Amazon"""
    try:
        with st.spinner("Scraping data from Amazon... This may take a few minutes."):
            df_amazon = scrape_amazon_reviews(url, page_number, pages_to_extract)
            if df_amazon is not None and not df_amazon.empty:
                df_amazon.dropna(inplace=True)
                return df_amazon
            else:
                st.error("No data was scraped. Please check the URL and try again.")
                return None
    except Exception as e:
        st.error(f"Error scraping data: {str(e)}")
        return None

def process_reviews(df):
    """Main processing function for reviews"""
    
    def split_review(text):
        """Split review into multiple sentences based on conjunctions"""
        delimiters = ".", "but", "and", "also"
        regex_pattern = '|'.join(map(re.escape, delimiters))
        splitted = re.split(regex_pattern, text)
        return splitted

    @st.cache_data
    def downloads():
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    downloads()
    
    # Data Cleaning
    lemma = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    
    # Remove important words from stopwords
    important_words = ['not', 'but', 'because', 'against', 'between', 'up', 'down', 
                      'in', 'out', 'once', 'before', 'after', 'few', 'more', 'most', 
                      'no', 'nor', 'same', 'some']
    
    for word in important_words:
        if word in all_stopwords:
            all_stopwords.remove(word)

    def clean_aspect_spacy(reviews):
        """Clean and preprocess reviews"""
        statement = reviews.lower().strip()
        contractions = {
            "won't": "will not", "cannot": "can not", "can't": "can not",
            "n't": " not", "what's": "what is", "it's": "it is",
            "'ve": " have", "i'm": "i am", "'re": " are",
            "he's": "he is", "she's": "she is", "*****": " ",
            "%": " percent ", "₹": " rupee ", "$": " dollar ",
            "€": " euro ", "'ll": " will", "doesn't": "does not"
        }
        
        for contraction, expansion in contractions.items():
            statement = statement.replace(contraction, expansion)
            
        statement = re.sub('[^a-zA-Z]', ' ', statement)
        statement = statement.split()
        final_statement = [lemma.lemmatize(word) for word in statement if word not in set(all_stopwords)]
        return ' '.join(final_statement)

    def get_splitted_reviews(df):
        """Split reviews and create new dataframe"""
        reviews = []
        dates = []
        raw_reviews = []

        for i, review in enumerate(df["Review"].values):
            review_split = split_review(review)
            review_split_ = [x for x in review_split if len(x.split()) >= 3]
            duplicate_dates = [str(df["Date"].values[i]) for _ in range(len(review_split_))]
            raws = [x for x in review_split if len(x.split()) >= 3]
            
            reviews.extend(review_split_)
            dates.extend(duplicate_dates)
            raw_reviews.extend(raws)

        reviews_ = [clean_aspect_spacy(text) for text in reviews]
        data = pd.DataFrame({"Date": dates, "Review": reviews_, "Raw_Review": raw_reviews})
        return data

    def extract_aspects(reviews, nlp):
        """Extract aspects from reviews"""
        aspect_list = reviews.apply(lambda row: apply_extraction(row, nlp), axis=1)
        return list(aspect_list)

    def add_data(data, aspect_list):
        """Add aspects and descriptions to dataframe"""
        rev_ = []
        dates_ = []
        aspects_ = []
        description_ = []
        raw_r = []

        for i, j in enumerate(aspect_list):
            if len(list(j.values())[0]) != 0:
                length = len(list(j.values())[0])
                rev_a = [data["Review"].values[i] for k in range(length)]
                dates_a = [data["Date"].values[i] for k in range(length)]
                raw_r_a = [data["Raw_Review"].values[i] for k in range(length)]
                aspects_a = [list(j.values())[0][h]["noun"] for h in range(length)]
                descrip_a = [list(j.values())[0][h]["adj"] for h in range(length)]

                rev_.extend(rev_a)
                dates_.extend(dates_a)
                raw_r.extend(raw_r_a)
                aspects_.extend(aspects_a)
                description_.extend(descrip_a)
            else:
                rev_.append(data["Review"].values[i])
                dates_.append(data["Date"].values[i])
                raw_r.append(data["Raw_Review"].values[i])
                aspects_.append('neutral')
                description_.append('neutral')

        return pd.DataFrame({
            "Date": dates_, "Review": rev_, "Aspect": aspects_, 
            "Description": description_, "Raw_Review": raw_r
        })
    
    def star_to_sentiment(label):
        if "1" in label or "2" in label:
            return "Negative"
        elif "3" in label:
            return "Neutral"
        else:
            return "Positive"

    def sentiment_scores(sentence):
        """Calculate sentiment scores"""
        prediction = pipe(sentence)[0]   # e.g. {'label': '1 star', 'score': 0.7}
        sentiment = star_to_sentiment(prediction['label'])
        stars = int(prediction['label'].split()[0])

        return sentiment, stars


    def create_final_dataframe(data):
        """Create final dataframe with sentiments"""
        sentiment_ = []
        compound = []

        for review in data["Review"].values:
            sentiment, stars = sentiment_scores(review)
            sentiment_.append(sentiment)
            compound.append(stars)

        data["Sentiment"] = sentiment_
        data["Score"] = compound
        data["Date"] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', inplace=True)
        data["Year"] = pd.DatetimeIndex(data['Date']).year
        data["Month"] = pd.DatetimeIndex(data['Date']).month

        return data.reset_index().drop(["index"], axis=1)

    # Process the reviews
    with st.spinner("Processing reviews... This may take a few minutes."):
        df1 = get_splitted_reviews(df)
        
        # Load spacy model
        nlp = spacy.load("en_core_web_sm")
        reviews_train = df1[["Review"]]
        aspect_list_train = extract_aspects(reviews_train, nlp)
        
        df2 = add_data(df1, aspect_list_train)
        final_df = create_final_dataframe(df2)
        
    return final_df

def display_analysis(dfinal):
    """Display the analysis results"""
    
    # Get top aspects
    top = dfinal["Aspect"].value_counts()[1:15]
    asp = list(dict(top).keys())

    def streamlit_menu():
        with st.sidebar:
            selected = option_menu(
                menu_title="Aspects",
                options=asp,
                menu_icon="cast",
                default_index=0,
            )
        return selected

    if asp:  # Only show menu if aspects exist
        select = streamlit_menu()

        # Create aspect sentiment bar chart
        asp_bar = []
        asp_score = []
        for k in asp:
            a1 = dfinal.groupby(by="Aspect")
            a2 = a1.get_group(k)
            a3 = a2["Score"].mean()
            asp_bar.append(k)
            asp_score.append(a3)

        df_bar = pd.DataFrame({"Aspect": asp_bar, "Score": asp_score})
        fig = px.bar(df_bar, x="Score", y="Aspect", title="Sentiments for Aspects", 
                    color="Score", orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        st.success("More score 👉🏻 Positive Review 😄")
        st.success("Less score 👉🏻 Negative Review 😡")

        def show_senti(data_, senti):
            data_show = data_[data_["Sentiment"] == str(senti)]
            data_show_imp = data_show[["Raw_Review", "Sentiment"]]
            data_display = data_show_imp.drop_duplicates(subset=["Raw_Review"])
            return data_display.reset_index().drop(["index"], axis=1).head(15)

        def pie_plot(data_, select):
            data_pos = data_[data_["Sentiment"] == "Positive"]
            data_neg = data_[data_["Sentiment"] == "Negative"]
            data_neu = data_[data_["Sentiment"] == "Neutral"]
            count = [
                round((data_pos.shape[0] * 100) / data_.shape[0]),
                round((data_neg.shape[0] * 100) / data_.shape[0]),
                round((data_neu.shape[0] * 100) / data_.shape[0])
            ]
            labels_ = ["Positive", "Negative", "Neutral"]
            fig = go.Figure(go.Pie(labels=labels_, values=count, hoverinfo="label+percent", 
                                 textinfo="value", title=f"Pie chart for {select}"))
            st.plotly_chart(fig, use_container_width=True)

        def line_plot(data, select):
            fig = px.line(data, x="Date", y="Score", 
                         title=f"Sentiment for {select} across timeline")
            fig.update_traces(line_color="purple")
            st.plotly_chart(fig, use_container_width=True)

        def wordcloud_plot(data, select):
            wc_data = dict(data["Description"].value_counts())
            if wc_data:
                wc = WordCloud().fit_words(wc_data)
                st.image(wc.to_array(), use_column_width=True, 
                        caption=f"Wordcloud for {select}")

        # Show analysis for selected aspect
        aspects = [x for x, value in enumerate(dfinal["Review"].values) if str(select) in value]
        data_ = dfinal.iloc[aspects]

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Positive Reviews"):
                st.table(show_senti(data_, 'Positive'))

        with col2:
            if st.button("Negative Reviews"):
                st.table(show_senti(data_, 'Negative'))

        pie_plot(data_, select)
        line_plot(data_, select)
        wordcloud_plot(data_, select)

def main():
    """Main application function"""
    
    # Data source selection
    st.header("📊 Data Source Selection")
    
    data_source = st.radio(
        "Choose your data source:",
        ["Use Database (Existing Products)", "Scrape Fresh Data"],
        horizontal=True
    )
    
    df_amazon = None
    
    if data_source == "Use Database (Existing Products)":
        st.subheader("📱 Available Products")
        
        # Display available products in a nice format
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📱 iPhone 15", use_container_width=True):
                st.session_state.selected_product = "iPhone 15"
        
        with col2:
            if st.button("🎧 JBL Earbuds", use_container_width=True):
                st.session_state.selected_product = "JBL Earbuds"
        
        with col3:
            if st.button("💻 MacBook", use_container_width=True):
                st.session_state.selected_product = "MacBook"
        
        with col4:
            if st.button("📱 Motorola Moto G85", use_container_width=True):
                st.session_state.selected_product = "Motorola Moto G85"
        
        # Load selected product data
        if 'selected_product' in st.session_state:
            st.success(f"Selected: {st.session_state.selected_product}")
            df_amazon = load_data_from_database(st.session_state.selected_product)
            
            if df_amazon is not None:
                st.info(f"Loaded {len(df_amazon)} reviews from database")
                st.dataframe(df_amazon.head(), use_container_width=True)
    
    else:  # Scrape Fresh Data
        st.subheader("🔍 Scrape Fresh Reviews")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            url = st.text_input("🔗 Amazon Product URL", 
                               placeholder="https://amazon.com/product...")
        
        with col2:
            page_number = st.number_input("📄 Starting Page Number", 
                                        min_value=1, value=1, step=1)
        
        with col3:
            pages_to_extract = st.number_input("📚 Number of Pages to Extract", 
                                             min_value=1, value=1, step=1)
        
        if st.button("🚀 Start Scraping", type="primary"):
            if url:
                df_amazon = scrape_fresh_data(url, page_number, pages_to_extract)
                if df_amazon is not None:
                    st.success(f"Successfully scraped {len(df_amazon)} reviews!")
                    st.dataframe(df_amazon.head(), use_container_width=True)
            else:
                st.error("Please enter a valid Amazon product URL")
    
    # Process and analyze data
    if df_amazon is not None and not df_amazon.empty:
        if st.button("🔍 Analyze Reviews", type="primary"):
            try:
                final_df = process_reviews(df_amazon)
                st.session_state.dfinal = final_df
                st.success("✅ Analysis completed successfully!")
                
                # Show basic statistics
                st.subheader("📈 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Reviews", len(final_df))
                
                with col2:
                    positive_count = len(final_df[final_df["Sentiment"] == "Positive"])
                    st.metric("Positive Reviews", positive_count)
                
                with col3:
                    negative_count = len(final_df[final_df["Sentiment"] == "Negative"])
                    st.metric("Negative Reviews", negative_count)
                
                with col4:
                    unique_aspects = len(final_df["Aspect"].unique())
                    st.metric("Unique Aspects", unique_aspects)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Display analysis results
    if 'dfinal' in st.session_state and st.session_state.dfinal is not None:
        st.header("🎯 Aspect-Based Sentiment Analysis")
        display_analysis(st.session_state.dfinal)

# Custom CSS for better appearance
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: visible;}
footer:after {
    content: 'Creator : Saurabh Bairagi';
    display: block;
    position: relative;
    color: white;
    padding: 0px;
    top: 3px;
}
.stButton > button {
    width: 100%;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()