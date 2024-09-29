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

# import sys
# sys.path.append(r"c:\users\rajpo\appdata\local\programs\python\python311\lib\site-packages")

#import warnings

#warnings.filterwarnings("ignore")
st.set_page_config(page_title='Product Summarization')
st.title('Product Review Summarisation')


### 1. Extract Data

#dfinal = 0



### 1. Extract Data

# dfinal = 0

def main_file(webpage, page_number, pages_to_extract):
    # global dfinal
    def amazon_data(webpage, page_number, pages_to_extract):
        """
        Given a URL,page number and number of pages to extract; this function extracts review, date, summary
        and creates a dataframe

        """

        webpage = webpage[:-1]
        amazon_review = []
        amazon_date = []

        def scrape_data_amazon(webpage, page_number, pages_to_extract):
            head = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
  'Accept-Language': 'en-US,en;q=0.9',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
}
            if '&page=' in webpage:
                webpage = webpage
            else:
                webpage = webpage + '&page='
            next_page = webpage + str(page_number)
            response = requests.get(str(next_page),headers=head) # headers=headers
                                
            #st.success(response)
            soup = BeautifulSoup(response.content, "html.parser")
            # print(soup)
            soup_review = soup.findAll("div", {"class": "ZmyHeo"})
            # soup_summary = soup.findAll("a",{"class":"a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold"})
            soup_date = soup.find_all(lambda tag: tag.name == 'p' and tag.get('class') == ['_2NsDsF'])  # 10 reviews
            # print(soup_review)
            for x in range(len(soup_review)):
                amazon_review.append(soup_review[x].text.replace('READ MORE', '').strip())
                amazon_date.append(soup_date[x].text.strip())
                # amazon_summary.append(soup_summary[x].text.strip())

            # Generating the next page url
            if page_number < pages_to_extract:
                page_number = page_number + 1
                # print(page_number)
                scrape_data_amazon(webpage, page_number, pages_to_extract)

        scrape_data_amazon(webpage, page_number, pages_to_extract)
        data_amazon = {'Date': amazon_date, 'Review': amazon_review}
        df_amazon = pd.DataFrame(data_amazon, columns=['Date', 'Review'])

        def get_date_amazon(text):
            if 'months ago' in text.lower() or 'month ago' in text.lower():
                number = int(re.search(r'\d+', text).group())
                current_date = datetime.date.today()
                months_ago = current_date - relativedelta(months=number)
                formatted_result = months_ago.strftime('%B, %Y')
                return formatted_result

            elif 'days ago' in text.lower() or 'day ago' in text.lower() or 'today' in text.lower():
                res = datetime.date.today()
                return res.strftime('%B, %Y')


            else:
                return text

        df_amazon["Date"] = [get_date_amazon(x) for x in df_amazon["Date"].values]
        df_amazon.dropna(inplace=True)

        #st.success(f"Scrapped len : {len(df_amazon)}")
        #print(f"Scrapped Reviews : {len(df_amazon)}")

        return df_amazon

    # final_df = 0
    df = amazon_data(str(url), int(page), int(extract))

    ### 2. Split Reviews

    def split_review(text):
        """
        This function splits the review into multiple sentences based on the following conjunctions

        """
        delimiters = ".", "but", "and", "also"
        regex_pattern = '|'.join(map(re.escape, delimiters))  # applying the above delimiters
        splitted = re.split(regex_pattern, text)  # splitting the review
        return splitted  # this returns a list of multiple reviews

    @st.cache(suppress_st_warning=True)
    def downloads():
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')



    downloads()
    ### 4. Data Cleaning
    lemma = WordNetLemmatizer()  # applying lemmatization to return dictionary form of words

    all_stopwords = stopwords.words('english')  # this consists all the stopwords, which will be removed later.

    # Removing the following words from list containing stopwords
    all_stopwords.remove('not')
    all_stopwords.remove('but')
    all_stopwords.remove('because')
    all_stopwords.remove('against')
    all_stopwords.remove('between')
    all_stopwords.remove('up')
    all_stopwords.remove('down')
    all_stopwords.remove('in')
    all_stopwords.remove('out')
    all_stopwords.remove('once')
    all_stopwords.remove('before')
    all_stopwords.remove('after')
    all_stopwords.remove('few')
    all_stopwords.remove('more')
    all_stopwords.remove('most')
    all_stopwords.remove('no')
    all_stopwords.remove('nor')
    all_stopwords.remove('same')
    all_stopwords.remove('some')

    def clean_aspect_spacy(reviews):
        """
        This function removes punctuations, stopwords and other non alpha numeric characters.
        We expand the contractions and replace some words by an empty string

        """
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

    ### 5. Form Dataframe again
    def get_splitted_reviews(df):
        """
        This function applies the above defined splitting function and forms a dataframe again

        """
        reviews = []  # this will contain our reviews
        dates = []  # this will contain our dates
        raw_reviews = []

        for i, j in enumerate(df["Review"].values):  # for each review
            review_split = split_review(j)  # apply the splitting the function
            review_split_ = [x for x in review_split if
                             len(x.split()) >= 3]  # review containing less than 3 words are removed
            duplicate_dates = [str(df["Date"].values[i]) for h in
                               range(len(review_split_))]  # repeat the dates for splitted reviews
            raws = [x for x in review_split if len(x.split()) >= 3]
            reviews.extend(review_split_)  # add reviews to list
            dates.extend(duplicate_dates)  # add dates to list
            raw_reviews.extend(raws)

        reviews_ = [clean_aspect_spacy(text) for text in reviews]  # applying the cleaning function

        data = pd.DataFrame({"Date": dates, "Review": reviews_, "Raw_Review": raw_reviews})  # create new dataframe
        return data

    df1 = get_splitted_reviews(df)

    ### 6. Aspect Extraction
    def apply_extraction(row, nlp):
        """
        This function extracts aspect and its corresponding description from the review by
        applying 7 different rules of pos tagging

        """

        prod_pronouns = ['it', 'this', 'they', 'these']
        review_body = row['Review']
        doc = nlp(review_body)

        rule1_pairs = []
        rule2_pairs = []
        rule3_pairs = []
        rule4_pairs = []
        rule5_pairs = []
        rule6_pairs = []
        rule7_pairs = []

        for token in doc:
            A = "999999"
            M = "999999"
            if token.dep_ == "amod" and not token.is_stop:
                M = token.text
                A = token.head.text

                # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
                M_children = token.children
                for child_m in M_children:
                    if (child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + M
                        break

                # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
                A_children = token.head.children
                for child_a in A_children:
                    if (child_a.dep_ == "det" and child_a.text == 'no'):
                        neg_prefix = 'not'
                        M = neg_prefix + " " + M
                        break

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict1 = {"noun": A, "adj": M, "rule": 1}
                rule1_pairs.append(dict1)

            # print("--- SPACY : Rule 1 Done ---")

            # -----------------------------------------------------------------------------------------------------------------------------
            # # SECOND RULE OF DEPENDANCY PARSE -
            # # M - Sentiment modifier || A - Aspect
            # Direct Object - A is a child of something with relationship of nsubj, while
            # M is a child of the same something with relationship of dobj
            # Assumption - A verb will have only one NSUBJ and DOBJ
            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if ((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict2 = {"noun": A, "adj": M, "rule": 2}
                rule2_pairs.append(dict2)

            # print("--- SPACY : Rule 2 Done ---")
            # -----------------------------------------------------------------------------------------------------------------------------

            ## THIRD RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Adjectival Complement - A is a child of something with relationship of nsubj, while
            ## M is a child of the same something with relationship of acomp
            ## Assumption - A verb will have only one NSUBJ and DOBJ
            ## "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "acomp" and not child.is_stop):
                    M = child.text

                # example - 'this could have been better' -> (this, not better)
                if (child.dep_ == "aux" and child.tag_ == "MD"):
                    neg_prefix = "not"
                    add_neg_pfx = True

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M
                # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict3 = {"noun": A, "adj": M, "rule": 3}
                rule3_pairs.append(dict3)
                # rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))
            # print("--- SPACY : Rule 3 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## FOURTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
            # M is a child of the same something with relationship of advmod

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if ((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "advmod" and not child.is_stop):
                    M = child.text
                    M_children = child.children
                    for child_m in M_children:
                        if (child_m.dep_ == "advmod"):
                            M_hash = child_m.text
                            M = M_hash + " " + child.text
                            break
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict4 = {"noun": A, "adj": M, "rule": 4}
                rule4_pairs.append(dict4)
                # rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )

            # print("--- SPACY : Rule 4 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## FIFTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Complement of a copular verb - A is a child of M with relationship of nsubj, while
            # M has a child with relationship of cop

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            buf_var = "999999"
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "cop" and not child.is_stop):
                    buf_var = child.text
                    # check_spelling(child.text)

            if (A != "999999" and buf_var != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict5 = {"noun": A, "adj": token.text, "rule": 5}
                rule5_pairs.append(dict5)
                # rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))

            # print("--- SPACY : Rule 5 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## SIXTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)

            children = token.children
            A = "999999"
            M = "999999"
            if (token.pos_ == "INTJ" and not token.is_stop):
                for child in children:
                    if (child.dep_ == "nsubj" and not child.is_stop):
                        A = child.text
                        M = token.text
                        # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict6 = {"noun": A, "adj": M, "rule": 6}
                rule6_pairs.append(dict6)

                # rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))

            # print("--- SPACY : Rule 6 Done ---")

            # ------------------------------------------------------------------------------------------------------------------------------

            ## SEVENTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## ATTR - link between a verb like 'be/seem/appear' and its complement
            ## Example: 'this is garbage' -> (this, garbage)

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if ((child.dep_ == "attr") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict7 = {"noun": A, "adj": M, "rule": 7}
                rule7_pairs.append(dict7)
                # rule7_pairs.append((A, M,sid.polarity_scores(M)['compound'],7))

        # print("--- SPACY : All Rules Done ---")

        # ------------------------------------------------------------------------------------------------------------------------------

        aspects = []

        aspects = rule1_pairs + rule2_pairs + rule3_pairs + rule4_pairs + rule5_pairs + rule6_pairs + rule7_pairs

        dic = {"aspect_pairs": aspects}
        return dic

    def extract_aspects(reviews, nlp):
        """
        Applying the aspect extraction function and returning a dictionary
        with key = Aspect & value = Description

        """
        print("Entering Apply function!")
        aspect_list = reviews.apply(lambda row: apply_extraction(row, nlp),
                                    axis=1)  # going through all the rows in the dataframe
        return aspect_list

    nlp = spacy.load("en_core_web_sm")
    reviews_train = df1[["Review"]]
    aspect_list_train = extract_aspects(reviews_train, nlp)
    aspect_list_train = list(aspect_list_train)

    ### 7. Add aspects to dataframe

    def add_data(data, aspect_list):
        """
        This function adds aspect and the description to the dataframe such that if one review has multiple aspects,
        then the reviews are repeated such that each row consists of single aspect and description

        """
        rev_ = []  # list containing reviews
        dates_ = []  # list containing dates
        aspects_ = []  # list containing aspects
        description_ = []  # list containing description
        raw_r = []

        #print(aspect_list)
        for i, j in enumerate(aspect_list):
            length = len(list(j.values())[0])
            if len(list(j.values())[0]) != 0:
                rev_a = [data["Review"].values[i] for k in range(len(list(j.values())[0]))]
                dates_a = [data["Date"].values[i] for k in range(len(list(j.values())[0]))]
                raw_r_a = [data["Raw_Review"].values[i] for k in range(len(list(j.values())[0]))]
                aspects_a = [list(j.values())[0][h]["noun"] for h in range(length)]
                descrip_a = [list(j.values())[0][h]["adj"] for h in range(length)]

                rev_.extend(rev_a)
                dates_.extend(dates_a)
                raw_r.extend(raw_r_a)
                aspects_.extend(aspects_a)
                description_.extend(descrip_a)

            # if a review doesn't contain any aspect then we add "neutral" for both the aspect & description column

            else:
                rev_.append(data["Review"].values[i])
                dates_.append(data["Date"].values[i])
                raw_r.append(data["Raw_Review"].values[i])
                aspects_.append('neutral')
                description_.append('neutral')

        data_ = pd.DataFrame(
            {"Date": dates_, "Review": rev_, "Aspect": aspects_, "Description": description_, "Raw_Review": raw_r})
        return data_

    df2 = add_data(df1, aspect_list_train)

    ### 8. Sentiments

    def sentiment_scores(sentence):
        senti = SentimentIntensityAnalyzer()
        sentiment_dict = senti.polarity_scores(sentence)  # this line returns the polarity for sentence

        if sentiment_dict['compound'] >= 0.05:  # if the compound score is >= 0.05 then the review is positive
            return ("Positive"), sentiment_dict['pos'], sentiment_dict['compound']

        elif sentiment_dict['compound'] <= - 0.05:  # if the compound score is <= -0.05 then the review is negative
            return ("Negative"), sentiment_dict['neg'], sentiment_dict['compound']

        else:
            return ("Neutral"), sentiment_dict['neu'], sentiment_dict[
                'compound']  # if compound score is in between 0.05 and -0.05 then the review is neutral

    ### 9. Final frame:
    # global final_df
    def date_df(data):
        sentiment_ = []  # this list carries the sentiment for the review
        compound = []  # this consists of score based on which the sentiments are defined

        for u in data["Review"].values:  # applying the above defined function by iterating through each review
            a, b, c = sentiment_scores(u)
            sentiment_.append(a)
            compound.append(c)

        data["Sentiment"] = sentiment_  # adding the sentiment to dataframe
        data["Score"] = compound  # adding the score to dataframe
        data["Date"] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', inplace=True)  # sorting the date in ascending order

        data["Year"] = pd.DatetimeIndex(data['Date']).year  # extracting year from date
        data["Month"] = pd.DatetimeIndex(data['Date']).month  # extracting month from date

        dfinal_ = data.reset_index().drop(["index"], axis=1)  # resetting the index
        return dfinal_

    print(f"Final number of reviews {len(df2)}")
    return date_df(df2)


url = st.text_input("Paste the URL here")
page = st.text_input("Enter the page number")
extract = st.text_input("Enter the number of pages to be searched")

try:
    if 'dfinal' not in st.session_state:
        st.session_state.dfinal = 0

    if st.button("Get Summary"):
        dfinal = main_file(str(url), int(page), int(extract))
        #st.write(len(dfinal))
        #st.table(dfinal.head(10))
        st.session_state.dfinal = dfinal
        # st.dataframe(st.session_state.dfinal)

    ### 10. Aspect:
    top = st.session_state.dfinal["Aspect"].value_counts()[1:15]
    asp = list(dict(top).keys())


    def streamlit_menu():
        with st.sidebar:
            selected = option_menu(
                menu_title="Aspects",  # required
                options=asp,  # required
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected


    select = streamlit_menu()

    asp_bar = []
    asp_score = []
    for k in asp:
        a1 = st.session_state.dfinal.groupby(by="Aspect")
        a2 = a1.get_group(k)
        a3 = a2["Score"].mean()
        asp_bar.append(k)
        asp_score.append(a3)

    df_bar = pd.DataFrame({"Aspect": asp_bar, "Score": asp_score})
    fig = px.bar(df_bar, x="Score", y="Aspect", title="Sentiments for Aspects", color="Score", orientation='h')
    st.plotly_chart(fig)
    st.success("More score 👉🏻 Positive Review 😄")
    st.success("Less score 👉🏻 Negative Review 😡")


    def show_senti(data_, senti):
        data_show = data_[data_["Sentiment"] == str(senti)]
        data_show_imp = data_show[["Raw_Review", "Sentiment"]]
        data_display = data_show_imp.drop_duplicates(subset=["Raw_Review"])
        data_display_ = data_display.reset_index().drop(["index"], axis=1)
        return data_display_.head(15)


    def pie_plot(data_, select):
        data_pos = data_[data_["Sentiment"] == "Positive"]
        data_neg = data_[data_["Sentiment"] == "Negative"]
        data_neu = data_[data_["Sentiment"] == "Neutral"]
        count = [round((data_pos.shape[0] * 100) / data_.shape[0]), round((data_neg.shape[0] * 100) / data_.shape[0]),
                 round((data_neu.shape[0] * 100) / data_.shape[0])]
        labels_ = ["Positive", "Negative", "Neutral"]
        fig = go.Figure(go.Pie(labels=labels_, values=count, hoverinfo="label+percent", textinfo="value",
                               title="Pie chart for {}".format(select)))
        st.plotly_chart(fig)


    def line_plot(data, select):
        fig = px.line(data, x="Date", y="Score", title=("Sentiment for {} across timeline".format(select)))
        fig.update_traces(line_color="purple")
        st.plotly_chart(fig)


    def worcloud_plot(data, select):
        wc_data = dict(data["Description"].value_counts())
        wc = WordCloud().fit_words(wc_data)
        st.image(wc.to_array(), use_column_width=True, caption="Wordcloud for {}".format(select))


    def show_aspects(data, aspect_name):
        if select == aspect_name:
            aspects = [x for x, value in enumerate(data["Review"].values) if str(select) in value]
            data_ = data.iloc[aspects]

            if st.button("Positive Reviews"):
                st.table(show_senti(data_, 'Positive'))

            if st.button("Negative Reviews"):
                st.table(show_senti(data_, 'Negative'))

            pie_plot(data_, select)
            line_plot(data_, select)
            worcloud_plot(data_, select)


    for x in asp:
        show_aspects(st.session_state.dfinal, x)

except TypeError:
    st.spinner(text="Waiting to get the link...")

hide_streamlit_style = """
            <style>
            #MainMenu {
            visibility: hidden;
            }
            footer{
            visibility: visible;
            }
            footer:after{
            content: 'Creator : Saurabh Bairagi';
            display:block;
            postion:relative;
            color:white;
            padding:0px;
            top:3px
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# streamlit run main_file.py --client.showErrorDetails=false
