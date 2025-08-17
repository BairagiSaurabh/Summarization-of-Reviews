import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urlparse, parse_qs

def extract_asin_from_url(input_url):
    """Extract ASIN from Amazon product review URL"""
    # Pattern to match ASIN in the URL
    asin_pattern = r'/product-reviews/([A-Z0-9]{10})/'
    match = re.search(asin_pattern, input_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Could not extract ASIN from URL")

def scrape_amazon_reviews(input_url, max_pages=5, delay=1):
    """
    Scrape Amazon reviews with pagination
    
    Args:
        input_url (str): Amazon product reviews URL
        max_pages (int): Maximum number of pages to scrape
        delay (float): Delay between requests in seconds
    
    Returns:
        pandas.DataFrame: DataFrame containing all reviews
    """
    
    # Extract ASIN from the input URL
    asin = extract_asin_from_url(input_url)
    print(f"Extracted ASIN: {asin}")
    
    # API endpoint for reviews
    api_url = "https://www.amazon.in/hz/reviews-render/ajax/reviews/get/ref=cm_cr_arp_d_paging_btm_next_2"
    
    # Headers for the request
    headers = {
        'accept': 'text/html,*/*',
        'accept-language': 'en-US,en;q=0.9',
        'anti-csrftoken-a2z': 'hFMg9G0oM2AA2A4Bim6CHsqOt8IcYcEF1sQeSaA5PuYiAAAAAGigZjAAAAAB',
        'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
        'referer': f'https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'Cookie': 'session-id=""; i18n-prefs=INR; lc-acbin=en_IN; ubid-acbin=258-8412160-0541739; id_pkel=n1; id_pk=""; x-acbin="MJlzP0Nlcx3rUygSeobkXf6p@w3PXdSPwZ3L6JvrMG?Vqa@C4bFv71YW90lx?MhS"; at-acbin=Atza|IwEBII7306Y3blNBye1uso_2ZQ3rsCAHqXZsT4wWJ6XL0jxRmzllkwuI6OhYwxms3QZ7cZ6le9vUqe0t85_zK_FOGrDiInqwz-D1vaVUmiXTpLZC6cYyDIAQmny-iR4iGEZPKgY0hIwcxpVc7advLocIUsZU79CXG4bKXPuna9fywOFxNUD1YBs5Axg5XPeaj7a7jtt3te-8hurGuAjAjfXngcvUHhDgKfdvvVpYyprrCwzyYw; sess-at-acbin=""; sst-acbin=""; session-id-time=2082787201l; session-token=""; csm-hit=tb:s-P6MT8V8Y3NYM4W70CVNF|1755280455367&t:1755280455559&adb:adblk_no; rxc=ALGUyUc2XAos2shtsB8; session-id=523-6839149-3250444; session-id-time=2082787201l; session-token=""'
    }
    
    all_reviews = []
    
    for page_num in range(1, max_pages + 1):
        print(f"Scraping page {page_num}...")
        
        # Payload for the current page
        payload = f"sortBy=&reviewerType=all_reviews&formatType=&mediaType=&filterByStar=&filterByAge=&pageNumber={page_num}&filterByLanguage=&filterByKeyword=&shouldAppend=undefined&deviceType=desktop&canShowIntHeader=undefined&reftag=cm_cr_arp_d_paging_btm_next_2&pageSize=10&asin={asin}&scope=reviewsAjax0"
        
        try:
            # Make the request
            response = requests.post(api_url, headers=headers, data=payload)
            
            if response.status_code != 200:
                print(f"Failed to fetch page {page_num}. Status code: {response.status_code}")
                continue
            
            # Parse the response
            chunks = [chunk.strip() for chunk in response.text.split("&&&") if chunk.strip()]
            
            if len(chunks) < 7:
                print(f"Unexpected response format on page {page_num}")
                continue
                
            parsed = [json.loads(chunk) for chunk in chunks]
            
            # Extract HTML content (usually in index 6)
            html_content = None
            for item in parsed:
                if len(item) > 2 and isinstance(item[2], str) and 'review-text' in item[2]:
                    html_content = item[2]
                    break
            
            if not html_content:
                print(f"No review content found on page {page_num}")
                continue
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract reviews
            page_reviews = extract_review_data(soup)
            
            if not page_reviews:
                print(f"No reviews found on page {page_num}. Stopping pagination.")
                break
            
            # Add page number to each review
            for review in page_reviews:
                review['page_number'] = page_num
            
            all_reviews.extend(page_reviews)
            print(f"Found {len(page_reviews)} reviews on page {page_num}")
            
            # Add delay between requests
            if page_num < max_pages:
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error scraping page {page_num}: {str(e)}")
            continue
    
    # Create DataFrame
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        print(f"\nTotal reviews scraped: {len(df)}")
        return df
    else:
        print("No reviews were scraped.")
        return pd.DataFrame()

def extract_review_data(soup):
    """Extract review text and dates using the working method"""
    try:
        # Find all spans with review text
        reviews = soup.find_all("span", class_="a-size-base review-text review-text-content")
        review_texts = [review.get_text(strip=True) if review else "" for review in reviews]
        
        # Find all spans with review dates
        dates = soup.find_all("span", class_="a-size-base a-color-secondary review-date")
        date_texts = []
        
        for date in dates:
            try:
                full_date_text = date.get_text(strip=True) if date else ""
                # Extract just the date part from "Reviewed in India on 18 July 2025"
                if " on " in full_date_text:
                    date_only = full_date_text.split(" on ")[-1]
                else:
                    date_only = full_date_text
                date_texts.append(date_only)
            except:
                date_texts.append("")
    except:
        review_texts = []
        date_texts = []
    
    # Match reviews with dates (pad with empty strings if lengths don't match)
    max_length = max(len(review_texts), len(date_texts)) if review_texts or date_texts else 0
    
    # Pad shorter list with empty strings
    while len(review_texts) < max_length:
        review_texts.append("")
    while len(date_texts) < max_length:
        date_texts.append("")
    
    # Convert to list of dictionaries for consistency with DataFrame
    reviews_data = []
    for i in range(max_length):
        try:
            reviews_data.append({
                'Review': review_texts[i] if i < len(review_texts) else "",
                'Date': date_texts[i] if i < len(date_texts) else "",
            })
        except:
            reviews_data.append({
                'Review': "",
                'Date': "",
            })
    
    return reviews_data

# # Main execution
# if __name__ == "__main__":
#     # Input parameters
#     # input_url = "https://www.amazon.in/product-reviews/B0D9LS6MVV/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2" (motorola)
#     # input_url = "https://www.amazon.in/product-reviews/B0CHX3TW6X/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2" #(iphone 15)
#     # input_url = "https://www.amazon.in/product-reviews/B08N5W4NNB/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2" # (macbook)
#     input_url = "https://www.amazon.in/product-reviews/B0DHL63KWK/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=2" # (JBL Earbuds)
#     pages_to_scrape = 15  # Number of pages to scrape
    
#     # Scrape reviews
#     reviews_df = scrape_amazon_reviews(input_url, max_pages=pages_to_scrape, delay=2)
    
#     if not reviews_df.empty:
#         # Display basic statistics
#         print(f"\nDataFrame shape: {reviews_df.shape}")
#         # Save to CSV
#         output_filename = f"jbl_earbuds.csv"
#         reviews_df.to_csv(output_filename, index=False, encoding='utf-8')
#         print(f"\nReviews saved to: {output_filename}")
    
#     else:
#         print("No reviews were scraped successfully.")