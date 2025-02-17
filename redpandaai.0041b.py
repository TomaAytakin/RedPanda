import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
import time
from datetime import datetime
import csv
import ccxt
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import tweepy

# ----------------------------------- CONFIGURATION -----------------------------------
# -------- API Keys and IDs --------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")

if not GEMINI_API_KEY:
    print("Error: Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    exit()
if not TELEGRAM_BOT_TOKEN:
    print("Error: Telegram Bot Token not found. Set TELEGRAM_BOT_TOKEN environment variable.")
    exit()
if not TELEGRAM_CHANNEL_ID:
    print("Error: Telegram Channel ID not found. Set TELEGRAM_CHANNEL_ID environment variable.")
    exit()

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-pro')
TELEGRAM_BOT = Bot(token=TELEGRAM_BOT_TOKEN)

# -------- Pump.fun Scraping --------
PUMP_FUN_NEW_TOKENS_URL = "https://pump.fun/"

# -------- Scanning, Analysis Intervals, and Thresholds --------
SCAN_INTERVAL_SECONDS = 60
SEEN_TOKENS = set()

MIN_HOLDERS_THRESHOLD = 100
MIN_MCAP_THRESHOLD = 20000
MAX_BUNDLES_THRESHOLD = 3

# -------- Data Storage and Learning --------
DATA_STORAGE_FILE = "token_analysis_history.csv"
SUCCESS_METRIC_PERCENT_INCREASE = 1.00
PERFORMANCE_TRACKING_WINDOW_HOURS = 10
PRICE_CHECK_INTERVAL_SECONDS = 60 * 15

# -------- Crypto Price Data --------
CRYPTO_PRICE_DATA_EXCHANGE = 'gateio'
PRICE_DATA_SYMBOL_FORMAT = '{}/USDT'
PRICE_DATA_LOOKBACK_MINUTES = 60

# -------- Trending Meta Configuration --------
CURRENT_TRENDING_METAS = [
    "ExampleMeta1",
    "ExampleMeta2"
]

# -------- Twitter API Configuration --------
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

if not TWITTER_BEARER_TOKEN:
    print("Error: Twitter API Bearer Token not found. Set the TWITTER_BEARER_TOKEN environment variable.")
    exit()

# -------- Signal Categories --------
SIGNAL_CATEGORIES = ["Fast Pump & Dump", "Long Hold ( > 2 Hours)", "Small Profit (50-100%) Play", "Not Recommended"] # Define signal categories

# ----------------------------------- FUNCTIONS -----------------------------------

def scrape_pump_fun_new_tokens(url):
    """... (unchanged) ..."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        token_elements = soup.find_all('div', class_='token-card')

        new_tokens = []
        for element in token_elements:
            token_name_element = element.find('a', class_='token-name-link')
            holders_element = element.find('span', class_='token-holders')
            mcap_element = element.find('span', class_='token-mcap')
            bundles_element = element.find('span', class_='token-bundles')

            if token_name_element and holders_element and mcap_element:
                token_name = token_name_element.text.strip()
                token_link = "https://pump.fun" + token_name_element['href']
                holders_text = holders_element.text.strip()
                mcap_text = mcap_element.text.strip()

                try:
                    holders = int(holders_text.split(' ')[0].replace(',', ''))
                    mcap_value = float(mcap_text.replace('$', '').replace('k', '').replace('m', '').strip())
                    mcap_multiplier = 1000 if 'k' in mcap_text.lower() else (1000000 if 'm' in mcap_text.lower() else 1)
                    mcap = mcap_value * mcap_multiplier
                except ValueError:
                    print(f"Warning: Could not parse holders or MCAP for {token_name}. Skipping.")
                    continue

                bundles = 0
                if bundles_element:
                    bundles_text = bundles_element.text.strip()
                    try:
                        bundles = int(bundles_text.split(' ')[0])
                    except ValueError:
                        print(f"Warning: Could not parse bundle count for {token_name}. Defaulting to 0.")

                new_tokens.append({
                    'name': token_name,
                    'link': token_link,
                    'holders': holders,
                    'mcap': mcap,
                    'bundles': bundles
                })

        return new_tokens

    except requests.exceptions.RequestException as e:
        print(f"Error scraping pump.fun: {e}")
        return []
    except Exception as e:
        print(f"Error parsing pump.fun HTML: {e}")
        return []


def send_telegram_message(message):
    """... (unchanged) ..."""
    try:
        TELEGRAM_BOT.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=message, parse_mode='Markdown')
        print("Telegram message sent successfully.")
    except TelegramError as e:
        print(f"Error sending Telegram message: {e}")


def scan_twitter_for_trends():
    """... (unchanged) ..."""
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

        trends_result = client.get_trends_place(1)
        trending_hashtags = []
        if trends_result.data:
            for trend in trends_result.data:
                if trend['name'].startswith('#') and any(keyword in trend['name'].lower() for keyword in ['memecoin', 'pumpfun', 'crypto', 'token', 'coin']):
                    trending_hashtags.append(trend['name'])
            print(f"Twitter Trending Hashtags (Meme Coin Related): {trending_hashtags}")

        recent_tweets = []
        search_query = "#memecoin OR #pumpfun OR memecoin OR pumpfun new token"
        tweet_search_result = client.search_recent_tweets(
            query=search_query,
            tweet_fields=["created_at", "lang", "author_id"],
            max_results=10
        )

        if tweet_search_result.data:
            for tweet in tweet_search_result.data:
                if tweet.lang == 'en':
                    recent_tweets.append(tweet.text)
            print(f"Recent Relevant Tweets (English, first 10): {recent_tweets[:3] if recent_tweets else 'No relevant recent tweets found.'}")

        return {
            'trending_hashtags': trending_hashtags,
            'recent_tweets': recent_tweets
        }

    except tweepy.TweepyException as e:
        print(f"Error accessing Twitter API (for general trends): {e}")
        return {
            'trending_hashtags': [],
            'recent_tweets': []
        }
    except Exception as e:
        print(f"Error processing Twitter API data (for general trends): {e}")
        return {
            'trending_hashtags': [],
            'recent_tweets': []
        }


def scan_twitter_for_token_mentions(token_name):
    """... (unchanged) ..."""
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

        search_query = f"${token_name} OR #{token_name} OR {token_name} OR $ {' '.join(token_name.split())}"

        tweet_search_result = client.search_recent_tweets(
            query=search_query,
            tweet_fields=["created_at", "lang", "author_id"],
            max_results=15
        )

        token_tweets = []
        if tweet_search_result.data:
            for tweet in tweet_search_result.data:
                if tweet.lang == 'en':
                    token_tweets.append(tweet.text)
            print(f"Recent Tweets mentioning Token '{token_name}' (first 10): {token_tweets[:3] if token_tweets else 'No recent tweets found.'}")

        return token_tweets

    except tweepy.TweepyException as e:
        print(f"Error accessing Twitter API (for token mentions of {token_name}): {e}")
        return []
    except Exception as e:
        print(f"Error processing Twitter API data (for token mentions of {token_name}): {e}")
        return []



def analyze_token_with_gemini(token_name, token_link, holders, mcap, bundles, twitter_trends, token_twitter_mentions):
    """
    Analyzes token with Gemini, now categorizing signals (Fast Pump & Dump, Long Hold, Small Profit Play).
    """
    trending_metas_str = ", ".join(CURRENT_TRENDING_METAS)
    trending_hashtags_str = ", ".join(twitter_trends.get('trending_hashtags', []))
    recent_tweets_str = "\n".join(twitter_trends.get('recent_tweets', [])[:3])
    token_tweets_str = "\n".join(token_twitter_mentions[:3])

    prompt = f"""
    Analyze the new cryptocurrency token '{token_name}' launched on pump.fun (link: {token_link}).

    **Crucial Parameters to Consider:**
    * **Holders:** {holders}
    * **Market Cap (MCAP):** ${mcap:,.0f}
    * **Bundles:** {bundles}

    **Current Trending Meme Coin Metas:** [{trending_metas_str}]
    **Current Twitter Trends (Meme Coin Related Hashtags):** [{trending_hashtags_str}]
    **Example Recent Tweets (General Meme Coin Trends):**
    {recent_tweets_str if recent_tweets_str else "No recent relevant tweets found in initial Twitter scan."}
    **Recent Twitter Mentions of Token '{token_name}' (Example Tweets):**
    {token_tweets_str if token_tweets_str else "No recent Twitter mentions of this token found in initial scan."}

    **Desired Output: Token Signal Category & Analysis.**  Categorize this token into ONE of the following signal categories:
    * **{SIGNAL_CATEGORIES[0]} (Fast Pump & Dump):**  Expect rapid initial pump, short holding time (minutes to < 2 hours), high volatility, quick profit-taking crucial.
    * **{SIGNAL_CATEGORIES[1]} (Long Hold > 2 Hours):**  Potential for sustained growth beyond initial pump, holding for > 2 hours possible (but still meme coin timeframe, manage risk).
    * **{SIGNAL_CATEGORIES[2]} (Small Profit 50-100% Play):** More moderate, potentially less volatile, aim for 50-100% profit, could be slower growth than pump & dumps.
    * **{SIGNAL_CATEGORIES[3]} (Not Recommended):**  Avoid this token, low potential, high risk of loss.

    Consider ALL parameters above, trending metas, Twitter trends (general & token-specific) to categorize the token signal AND provide analysis justifying the categorization.

    Factors to guide categorization (consider how these factors might point to different signal categories):
    * Token Name/Ticker & Meta Alignment
    * Initial Holders, MCAP, Bundles (suggest initial momentum?)
    * General Crypto Sentiment & Meme Coin Trend Strength
    * Twitter Trend Signals (general meme coin trends & token-specific buzz)

    **Analyze and categorize the token. Provide:**
    1.  **Predicted Signal Category:** Choose ONE category from [{', '.join(SIGNAL_CATEGORIES)}].
    2.  **Analysis (2-3 sentences) Justifying Category:** Explain reasoning behind the chosen category based on the factors considered.

    **Important: HIGH-RISK, NOT financial advice. Categories are probabilistic assessments, manage risk accordingly.**

    Format for Telegram (Markdown):

    *Token: {token_name}*
    *Signal Category:* [Predicted Category Here]
    *Holders:* {holders}  *MCAP:* ${mcap:,.0f}  *Bundles:* {bundles}
    *Trending Meta Alignment:* [Assess]  *General Twitter Trend:* [Assess]  *Token-Specific Buzz:* [Assess]
    [Gemini Analysis Justifying Category.]
    """

    try:
        response = MODEL.generate_content(prompt)
        gemini_analysis = response.text

        # --- Attempt to extract predicted category from Gemini's response (can be improved with more robust parsing if needed) ---
        predicted_category = "Unknown" # Default category if extraction fails
        for category in SIGNAL_CATEGORIES:
            if category in gemini_analysis: # Simple keyword check - improve parsing for robustness if needed
                predicted_category = category
                break


        telegram_message = f"*New Token Analysis*\n\n*Token:* {token_name}\n*Signal Category:* *{predicted_category}*\n*Holders:* {holders}  *MCAP:* ${mcap:,.0f}  *Bundles:* {bundles}\n*Trending Meta Alignment:*\n[Assess]\n*General Twitter Trend Signal:*\n[Assess]\n*Token-Specific Twitter Buzz:*\n[Assess]\n*Link:* {token_link}\n\n*Gemini Analysis (Category Justification):*\n{gemini_analysis}\n\n*Disclaimer:* NOT financial advice. Meme coins are high risk. Signal category is probabilistic."
        send_telegram_message(telegram_message)

        return gemini_analysis, predicted_category # Return both analysis text and predicted category

    except Exception as e:
        error_message = f"Error analyzing token '{token_name}' with Gemini: {e}"
        print(error_message)
        return error_message, "Error" # Return "Error" category in case of exception


def record_analysis_data(token_data, gemini_analysis, prediction_timestamp, call_outcome=None, signal_category="Unknown"): # Added signal_category
    """Records token analysis data to CSV, now including signal_category and call outcome."""
    header_exists = os.path.exists(DATA_STORAGE_FILE)
    with open(DATA_STORAGE_FILE, mode='a', newline='') as csvfile:
        fieldnames = ['token_name', 'token_link', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'prediction_timestamp', 'call_outcome', 'signal_category'] # Added 'signal_category'
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not header_exists:
            writer.writeheader()

        writer.writerow({
            'token_name': token_data['name'],
            'token_link': token_data['link'],
            'holders': token_data['holders'],
            'mcap': token_data['mcap'],
            'bundles': token_data['bundles'],
            'gemini_analysis': gemini_analysis,
            'prediction_timestamp': prediction_timestamp.isoformat(),
            'call_outcome': call_outcome,
            'signal_category': signal_category # Record predicted signal category
        })



async def get_token_price_data(token_name, prediction_timestamp):
    """... (unchanged) ..."""
    try:
        exchange_id = CRYPTO_PRICE_DATA_EXCHANGE
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()

        symbol = PRICE_DATA_SYMBOL_FORMAT.format(token_name.upper())

        initial_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=PRICE_DATA_LOOKBACK_MINUTES)
        if not initial_ohlcv:
            print(f"No initial price data found for {symbol} on {exchange_id} at prediction time.")
            return None

        initial_df = pd.DataFrame(initial_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        initial_price = initial_df['close'].iloc[-1]

        if initial_price == 0:
            print(f"Error: Initial price for {symbol} is 0. Cannot calculate performance.")
            return None

        target_price = initial_price * (1 + SUCCESS_METRIC_PERCENT_INCREASE)
        end_time = prediction_timestamp + datetime.timedelta(hours=PERFORMANCE_TRACKING_WINDOW_HOURS)
        is_good_call = False
        max_price_reached = initial_price

        current_time = datetime.now()
        while current_time <= end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                if ohlcv:
                    current_price = ohlcv[0][4]
                    max_price_reached = max(max_price_reached, current_price)

                    if current_price >= target_price:
                        is_good_call = True
                        print(f"  Token {token_name} reached target price ({SUCCESS_METRIC_PERCENT_INCREASE*100}%) - Good Call!")
                        break

                    print(f"  Token {token_name}: Current Price: {current_price:.6f}, Target Price: {target_price:.6f}, Max Reached: {max_price_reached:.6f}, Time left: {end_time - current_time}")
                else:
                    print(f"Warning: No price data returned from exchange for {symbol} at {current_time}.")

            except ccxt.ExchangeError as e:
                print(f"Exchange error while tracking {token_name}: {e}")
            except Exception as e:
                print(f"Error during price tracking for {token_name}: {e}")

            await asyncio.sleep(PRICE_CHECK_INTERVAL_SECONDS)
            current_time = datetime.now()

        final_percent_change = (max_price_reached - initial_price) / initial_price if initial_price != 0 else 0
        print(f"  Token {token_name} Performance Tracking завершено. Max Price Reached: {max_price_reached:.6f}, Percent Change: {final_percent_change:.4f}%, Good Call: {is_good_call}")

        return {'is_good_call': is_good_call, 'percent_change': final_percent_change, 'max_price_reached': max_price_reached, 'initial_price': initial_price}



def evaluate_prediction_performance():
    """Evaluates prediction performance, now analyzing performance by signal category."""
    try:
        if not os.path.exists(DATA_STORAGE_FILE):
            print("No historical data available for evaluation yet.")
            return

        df = pd.read_csv(DATA_STORAGE_FILE)
        if df.empty:
            print("No data in historical file to evaluate.")
            return

        good_calls_df = df[df['call_outcome'] == 'Good']
        bad_calls_df = df[df['call_outcome'] == 'Bad']

        prompt = f"""
        Evaluate performance of my token analysis script based on historical data, including 'call_outcome' and *predicted 'signal_category'*.

        Current Trending Meme Coin Metas: [{', '.join(CURRENT_TRENDING_METAS)}]
        Signal Categories used for prediction: [{', '.join(SIGNAL_CATEGORIES)}]

        Here are examples of *Good Calls*:
        (Tokens that achieved >= {SUCCESS_METRIC_PERCENT_INCREASE*100}% increase, showing *predicted signal category*):
        [Start of Good Call Data Example]
        {good_calls_df.head(3).to_string(columns=['token_name', 'signal_category', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'call_outcome'], index=False)}
        [End of Good Call Data Example]

        Here are examples of *Bad Calls*:
        (Tokens that did NOT achieve >= {SUCCESS_METRIC_PERCENT_INCREASE*100}% increase, showing *predicted signal category*):
        [Start of Bad Call Data Example]
        {bad_calls_df.head(3).to_string(columns=['token_name', 'signal_category', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'call_outcome'], index=False)}
        [End of Bad Call Data Example]

        Analyze data, considering *predicted signal category*, *current trending metas*, *general Twitter trend signals*, AND *token-specific Twitter buzz* to understand prediction performance *for each signal category*.

        For EACH SIGNAL CATEGORY ({', '.join(SIGNAL_CATEGORIES)}), analyze the following:

        1.  **Performance by Category:**  What is the 'Good Call' rate (percentage of 'Good Calls') for each *predicted signal category*? Are some categories significantly more accurate than others? Is the 'Not Recommended' category effectively filtering out bad performers?
        2.  **Characteristics of Categories:** For each signal category, analyze the typical characteristics of tokens predicted in that category.  Specifically, examine:
            *   Typical ranges or patterns for Holders, MCAP, Bundles.
            *   Common themes or alignments with *current trending metas*.
            *   Typical *general Twitter trend signals* and *token-specific Twitter buzz* levels.
            *   Are there distinct patterns in 'gemini_analysis' text for each category?  What kind of analysis leads Gemini to categorize a token as 'Fast Pump & Dump' vs. 'Long Hold' vs. 'Small Profit' vs. 'Not Recommended'?
        3.  **Category Prediction Accuracy Improvement:** Based on your analysis of performance by category and the characteristics of each category, suggest specific improvements to the `analyze_token_with_gemini` prompt to improve the *accuracy of category predictions*. How can we better guide Gemini to distinguish between 'Fast Pump & Dump', 'Long Hold', 'Small Profit Play', and 'Not Recommended' signals?  Are there specific keywords, phrases, or instructions to add to the prompt to improve category prediction?
        4.  **Refine Twitter Scanning (Category-Specific)?** Should we refine `scan_twitter_for_token_mentions` or `scan_twitter_for_trends` functions to capture different *types* of Twitter signals that might be more relevant for *specific signal categories*? (e.g., are 'Fast Pump & Dump' tokens associated with very short bursts of intense Twitter hype, while 'Long Hold' tokens show more sustained community discussion?)
        5.  **Update Trending Metas List:** ... (unchanged) ...
        6.  **Prompt Refinement for Categories, Metas & Twitter:** Suggest specific prompt improvements to better incorporate 'trending metas', 'general Twitter trend signals', *'token-specific Twitter buzz'*, AND *signal category prediction* factors in `analyze_token_with_gemini`.

        Focus on actionable recommendations to improve 'Good Call' rate *within each signal category* and improve the *accuracy of signal category predictions*.
        """

        response = MODEL.generate_content(prompt)
        gemini_feedback = response.text
        print("\n--- Gemini Feedback on Prediction Performance (Signal Category Analysis): ---\n")
        print(gemini_feedback)

        # --- **Action Required: Manual Review and Update** ---
        print("\n--- **Action Required: Review Gemini's Feedback Above.**")
        print("--- **1. MANUALLY UPDATE the CURRENT_TRENDING_METAS list in the script.**")
        print("--- **2. MANUALLY REFINE the `analyze_token_with_gemini` prompt (especially for category prediction).** ---")
        print("--- **3. Consider refining Twitter scanning for category-specific signals.** ---")


    except Exception as e:
        print(f"Error evaluating prediction performance (Signal Category Analysis): {e}")



async def main():
    """... (rest of main function) ..."""
    print("Starting pump.fun token scanner, Gemini analyzer, and Telegram relay (with Signal Category Learning)...")
    while True:
        print(f"\n--- Scanning for new tokens at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        new_tokens = scrape_pump_fun_new_tokens(PUMP_FUN_NEW_TOKENS_URL)

        if new_tokens:
            print(f"Found {len(new_tokens)} new token(s) scraped from pump.fun.")

            filtered_tokens = []
            for token in new_tokens:
                if token['holders'] >= MIN_HOLDERS_THRESHOLD and token['mcap'] >= MIN_MCAP_THRESHOLD and token['bundles'] <= MAX_BUNDLES_THRESHOLD:
                    filtered_tokens.append(token)
                else:
                    print(f"Token '{token['name']}' filtered out - Holders: {token['holders']}, MCAP: ${token['mcap']:.0f}, Bundles: {token['bundles']} - Below thresholds.")

            if filtered_tokens:
                print(f"Analyzing {len(filtered_tokens)} token(s) meeting criteria.")
                for token in filtered_tokens:
                    if token['name'] not in SEEN_TOKENS:
                        print(f"\n-- Analyzing New Token: {token['name']} --")
                        print(f"  Link: {token['link']}, Holders: {token['holders']}, MCAP: ${token['mcap']:.0f}, Bundles: {token['bundles']}")

                        twitter_trends_data = scan_twitter_for_trends()
                        token_twitter_mentions_data = scan_twitter_for_token_mentions(token['name'])

                        prediction_timestamp = datetime.now()
                        gemini_analysis_result, predicted_category = analyze_token_with_gemini( # Capture predicted_category
                            token['name'], token['link'], token['holders'], token['mcap'], token['bundles'], twitter_trends_data, token_twitter_mentions_data
                        )

                        print("\n-- Gemini Analysis: --")
                        print(gemini_analysis_result)
                        print(f"-- Predicted Signal Category: {predicted_category} --")


                        record_analysis_data(token, gemini_analysis_result, prediction_timestamp, signal_category=predicted_category) # Record with signal_category

                        SEEN_TOKENS.add(token['name'])

                        asyncio.create_task(track_token_performance_and_update(token['name'], prediction_timestamp))

                    else:
                        print(f"Token '{token['name']}' already analyzed, skipping.")
            else:
                print("No new tokens met holder, MCAP, and bundle criteria.")


        else:
            print("No new tokens found on pump.fun.")

        if datetime.now().minute % 30 == 0:
            print("\n--- Evaluating Prediction Performance (Signal Category Analysis) ---")
            evaluate_prediction_performance()

        print(f"Waiting {SCAN_INTERVAL_SECONDS} seconds before next scan...")
        time.sleep(SCAN_INTERVAL_SECONDS)


async def track_token_performance_and_update(token_name, prediction_timestamp):
    """... (unchanged) ..."""
    performance_data = await get_token_price_data(token_name, prediction_timestamp)

    if performance_data:
        is_good_call = performance_data['is_good_call']
        call_outcome = "Good" if is_good_call else "Bad"

        print(f"\n--- Performance Tracking завершено for Token: {token_name} ---")
        print(f"  Call Outcome: {call_outcome}")

        update_call_outcome_in_csv(token_name, call_outcome)
    else:
        print(f"\n--- Performance Tracking failed to get data for Token: {token_name} ---")
        update_call_outcome_in_csv(token_name, "DataError")


def update_call_outcome_in_csv(token_name, call_outcome):
    """... (unchanged) ..."""
    try:
        df = pd.read_csv(DATA_STORAGE_FILE)
        df.loc[df['token_name'] == token_name, 'call_outcome'] = call_outcome
        df.to_csv(DATA_STORAGE_FILE, index=False)
        print(f"  CSV updated for Token: {token_name} - Call Outcome set to: {call_outcome}")

    except Exception as e:
        print(f"Error updating CSV with call outcome for {token_name}: {e}")



if __name__ == "__main__":
    asyncio.run(main())

#========================================================================================================
#TOMA AYTAKIN (EMPRESS)
#REGISTERED CODE IN WEBFLOW #2005910000058681 12/14/2025 11:41:30 UTC
#========================================================================================================