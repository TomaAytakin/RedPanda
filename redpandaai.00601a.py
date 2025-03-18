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
import discord
import openai
from PIL import Image, ImageDraw, ImageFont
import logging
from io import BytesIO

# ----------------------------------- CONFIGURATION -----------------------------------
# -------- API Keys and IDs --------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # OpenAI API Key
SOLSCAN_API_KEY = os.environ.get("SOLSCAN_API_KEY") # Solscan Pro API Key - NEW

if not GEMINI_API_KEY:
    print("Error: Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    exit()
if not TELEGRAM_BOT_TOKEN:
    print("Error: Telegram Bot Token not found. Set TELEGRAM_BOT_TOKEN environment variable.")
    exit()
if not TELEGRAM_CHANNEL_ID:
    print("Error: Telegram Channel ID not found. Set TELEGRAM_CHANNEL_ID environment variable.")
    exit()
if not DISCORD_BOT_TOKEN:
    print("Error: Discord Bot Token not found. Set DISCORD_BOT_TOKEN environment variable.")
    exit()
if not DISCORD_CHANNEL_ID:
    print("Error: Discord Channel ID not found. Set DISCORD_CHANNEL_ID environment variable.")
    exit()
if not OPENAI_API_KEY:
    print("Warning: OpenAI API key not found. OpenAI analysis will be disabled.")
    USE_OPENAI = False # Flag to disable OpenAI if API key is missing
else:
    openai.api_key = OPENAI_API_KEY
    USE_OPENAI = True
if not SOLSCAN_API_KEY: # Check for Solscan API Key - NEW
    print("Error: Solscan API key not found. Set SOLSCAN_API_KEY environment variable.")
    exit()


genai.configure(api_key=GEMINI_API_KEY)
MODEL_GEMINI = genai.GenerativeModel('gemini-pro')
TELEGRAM_BOT = Bot(token=TELEGRAM_BOT_TOKEN)
DISCORD_BOT = discord.Client(intents=discord.Intents.default())

# -------- Pump.fun Scraping --------
PUMP_FUN_NEW_TOKENS_URL = "https://pump.fun/board?coins_sort=created_timestamp"

# -------- Scanning, Analysis Intervals, and Thresholds --------
SCAN_INTERVAL_SECONDS = 60
SEEN_TOKENS = set()

MIN_HOLDERS_THRESHOLD = 100 # This threshold will now be based on Solscan API data
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

# -------- Image Configuration --------
BACKGROUND_IMAGE_PATH = "background_image.png"  # Path to your background image - MAKE SURE TO REPLACE THIS WITH YOUR ACTUAL IMAGE FILE NAME
FONT_PATH = "arial.ttf" # Path to a font file (e.g., Arial) - you might need to adjust this based on your system fonts
DEFAULT_FONT_SIZE = 24
HEADER_FONT_SIZE = 36
TEXT_COLOR = (255, 255, 255) # White color for text
HEADER_COLOR = (255, 255, 255) # White color for header

# -------- Logging Configuration --------
LOG_FILE = "token_scanner.log"
LOG_ADMIN_DISCUSSION_FILE = "admin_discussion_log.txt" # Log file for admin discussions
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Script started")


# ----------------------------------- FUNCTIONS -----------------------------------

def get_solscan_holder_count(token_address):
    """
    Retrieves token holder count from Solscan Pro API.
    Requires SOLSCAN_API_KEY environment variable to be set.
    """
    api_key = SOLSCAN_API_KEY
    headers = {
        'accept': 'application/json',
        'token': api_key,
    }
    params = {
        'tokenAddress': token_address,
        'page': 1,
        'page_size': 10 # We only need total count, pagination not crucial for this call
    }

    try:
        response = requests.get('https://pro-api.solscan.io/v2.0/token/holders', headers=headers, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        total_holders = data.get('data', {}).get('total')
        if total_holders is not None:
            logging.info(f"Solscan API - Holders for {token_address}: {total_holders}")
            return total_holders
        else:
            logging.warning(f"Solscan API - Holder count not found in response for {token_address}")
            return None # Indicate holder count not found in response

    except requests.exceptions.RequestException as e:
        logging.error(f"Solscan API Request Error for {token_address}: {e}")
        return None # Indicate API request error
    except json.JSONDecodeError as e: # Handle potential JSON decode errors
        logging.error(f"Solscan API JSON Decode Error for {token_address}: {e}")
        return None # Indicate JSON decode error
    except Exception as e:
        logging.error(f"Error fetching holder count from Solscan API for {token_address}: {e}")
        return None # Indicate general error


def scrape_pump_fun_new_tokens(url):
    """
    Scrapes new tokens from pump.fun and now fetches holder count from Solscan API.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        token_elements = soup.find_all('div', class_='token-card')

        new_tokens = []
        for element in token_elements:
            token_name_element = element.find('a', class_='token-name-link')
            mcap_element = element.find('span', class_='token-mcap')
            bundles_element = element.find('span', class_='token-bundles')
            token_address_element = element.find('a', href=lambda href: href and "solscan.io/token" in href) # Find link to Solscan


            if token_name_element and mcap_element and token_address_element:
                token_name = token_name_element.text.strip()
                token_link = "https://pump.fun" + token_name_element['href']
                mcap_text = mcap_element.text.strip()
                token_address_url = token_address_element['href']
                token_address = token_address_url.split('/')[5] if len(token_address_url.split('/')) >= 6 else None # Extract token address from Solscan URL

                if not token_address:
                    logging.warning(f"Could not extract token address from Solscan URL for {token_name}. Skipping token.")
                    continue # Skip if token address extraction fails

                holders = get_solscan_holder_count(token_address) # Get holders from Solscan API - NEW

                if holders is None: # Skip token if holder count retrieval fails - NEW
                    logging.warning(f"Could not retrieve holder count from Solscan API for {token_name} (Address: {token_address}). Skipping token.")
                    continue


                try:
                    mcap_value = float(mcap_text.replace('$', '').replace('k', '').replace('m', '').strip())
                    mcap_multiplier = 1000 if 'k' in mcap_text.lower() else (1000000 if 'm' in mcap_text.lower() else 1)
                    mcap = mcap_value * mcap_multiplier
                except ValueError:
                    logging.warning(f"Could not parse MCAP for {token_name}. Skipping.")
                    continue

                bundles = 0
                if bundles_element:
                    bundles_text = bundles_element.text.strip()
                    try:
                        bundles = int(bundles_text.split(' ')[0])
                    except ValueError:
                        logging.warning(f"Could not parse bundle count for {token_name}. Defaulting to 0.")

                new_tokens.append({
                    'name': token_name,
                    'link': token_link,
                    'holders': holders, # Holders from Solscan API - NEW
                    'mcap': mcap,
                    'bundles': bundles,
                    'address': token_address # Add token address - NEW
                })

        return new_tokens

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping pump.fun: {e}")
        return []
    except Exception as e:
        logging.error(f"Error parsing pump.fun HTML: {e}")
        return []


def send_telegram_message(image_path, text_message=None):
    """... (unchanged) ..."""
    try:
        if image_path:
            with open(image_path, 'rb') as photo: # Open the image in binary read mode
                TELEGRAM_BOT.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=photo)
            logging.info("Telegram message sent successfully (image).")
        elif text_message:
            TELEGRAM_BOT.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=text_message, parse_mode='Markdown')
            logging.info("Telegram message sent successfully (text).")
        else:
            logging.warning("send_telegram_message called without image_path or text_message.")

    except TelegramError as e:
        logging.error(f"Error sending Telegram message: {e}")


async def send_discord_message(image_path, text_message=None):
    """... (unchanged) ..."""
    try:
        channel = DISCORD_BOT.get_channel(int(DISCORD_CHANNEL_ID))
        if not channel:
            logging.error(f"Discord channel ID {DISCORD_CHANNEL_ID} not found.")
            return

        if image_path:
            with open(image_path, 'rb') as f:
                picture = discord.File(f)
                await channel.send(file=picture)
            logging.info("Discord message sent successfully (image).")
        elif text_message:
            await channel.send(text_message)
            logging.info("Discord message sent successfully (text).")
        else:
            logging.warning("send_discord_message called without image_path or text_message.")

    except discord.errors.DiscordException as e:
        logging.error(f"Error sending Discord message: {e}")


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
            logging.info(f"Twitter Trending Hashtags (Meme Coin Related): {trending_hashtags}")

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
            logging.info(f"Recent Relevant Tweets (English, first 10): {recent_tweets[:3] if recent_tweets else 'No relevant recent tweets found.'}")

        return {
            'trending_hashtags': trending_hashtags,
            'recent_tweets': recent_tweets
        }

    except tweepy.TweepyException as e:
        logging.error(f"Error accessing Twitter API (for general trends): {e}")
        return {
            'trending_hashtags': [],
            'recent_tweets': []
        }
    except Exception as e:
        logging.error(f"Error processing Twitter API data (for general trends): {e}")
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
            logging.info(f"Recent Tweets mentioning Token '{token_name}' (first 10): {token_tweets[:3] if token_tweets else 'No recent tweets found.'}")

        return token_tweets

    except tweepy.TweepyException as e:
        logging.error(f"Error accessing Twitter API (for token mentions of {token_name}): {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing Twitter API data (for token mentions of {token_name}): {e}")
        return []


def analyze_token_with_gemini(token_name, token_link, holders, mcap, bundles, twitter_trends, token_twitter_mentions):
    """
    Analyzes token with Gemini, now categorizing signals.
    """
    trending_metas_str = ", ".join(CURRENT_TRENDING_METAS)
    trending_hashtags_str = ", ".join(twitter_trends.get('trending_hashtags', []))
    recent_tweets_str = "\n".join(twitter_trends.get('recent_tweets', [])[:3])
    token_tweets_str = "\n".join(token_twitter_mentions[:3])

    prompt = f"""
    Analyze the new cryptocurrency token '{token_name}' launched on pump.fun (link: {token_link}).

    **Crucial Parameters to Consider:**
    * **Holders (Solscan API):** {holders}
    * **Market Cap (MCAP):** ${mcap:,.0f}
    * **Bundles:** {bundles}

    **Data Source for Holders:** Holder count is now fetched from Solscan API for increased reliability.

    **Current Trending Meme Coin Metas:** [{trending_metas_str}]
    **Current Twitter Trends (Meme Coin Related Hashtags):** [{trending_hashtags_str}]
    **Example Recent Tweets (General Meme Coin Trends):**
    {recent_tweets_str if recent_tweets_str else "No recent relevant tweets found in initial Twitter scan."}
    **Recent Twitter Mentions of Token '{token_name}' (Example Tweets):**
    {token_tweets_str if token_tweets_str else "No recent Twitter mentions of this token found in initial scan."}

    **Desired Output: Token Signal Category & Analysis.**  Categorize this token into ONE of the following signal categories:
    * **{SIGNAL_CATEGORIES[0]} (Fast Pump & Dump):**  Expect rapid initial pump, short holding time (minutes to < 2 hours), high volatility, quick profit-taking crucial.
    * **{SIGNAL_CATEGORIES[1]} (Long Hold > 2 Hours):**  Potential for sustained growth beyond initial pump, holding for > 2 hours possible (but still meme coin timeframe, manage risk).
    * **{SIGNAL_CATEGORIES[2]} (Small Profit 50-100%) Play):** More moderate, potentially less volatile, aim for 50-100% profit, could be slower growth than pump & dumps.
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
    *Holders (Solscan API):* {holders}  *MCAP:* ${mcap:,.0f}  *Bundles:* {bundles}
    *Trending Meta Alignment:* [Assess]  *General Twitter Trend:* [Assess]  *Token-Specific Buzz:* [Assess]
    [Gemini Analysis Justifying Category.]
    """

    try:
        response = MODEL_GEMINI.generate_content(prompt)
        gemini_analysis = response.text

        # --- Attempt to extract predicted category from Gemini's response ---
        predicted_category = "Unknown" # Default category if extraction fails
        for category in SIGNAL_CATEGORIES:
            if category in gemini_analysis:
                predicted_category = category
                break

        return gemini_analysis, predicted_category

    except Exception as e:
        error_message = f"Error analyzing token '{token_name}' with Gemini: {e}"
        logging.error(error_message)
        return error_message, "Error" # Return "Error" category in case of exception


def analyze_token_with_openai(token_name, token_link, holders, mcap, bundles, twitter_trends, token_twitter_mentions):
    """Analyzes token with OpenAI (GPT-3.5-turbo), categorizing signals."""
    if not USE_OPENAI:
        return "OpenAI analysis disabled due to missing API key.", "Disabled"

    trending_metas_str = ", ".join(CURRENT_TRENDING_METAS)
    trending_hashtags_str = ", ".join(twitter_trends.get('trending_hashtags', []))
    recent_tweets_str = "\n".join(twitter_trends.get('recent_tweets', [])[:3])
    token_tweets_str = "\n".join(token_twitter_mentions[:3])


    prompt = f"""
    Analyze the new cryptocurrency token '{token_name}' launched on pump.fun (link: {token_link}).

    **Crucial Parameters to Consider:**
    * **Holders (Solscan API):** {holders}
    * **Market Cap (MCAP):** ${mcap:,.0f}
    * **Bundles:** {bundles}

    **Data Source for Holders:** Holder count is now fetched from Solscan API for increased reliability.

    **Current Trending Meme Coin Metas:** [{trending_metas_str}]
    **Current Twitter Trends (Meme Coin Related Hashtags):** [{trending_hashtags_str}]
    **Example Recent Tweets (General Meme Coin Trends):**
    {recent_tweets_str if recent_tweets_str else "No recent relevant tweets found in initial Twitter scan."}
    **Recent Twitter Mentions of Token '{token_name}' (Example Tweets):**
    {token_tweets_str if token_tweets_str else "No recent Twitter mentions of this token found in initial scan."}

    **Desired Output: Token Signal Category & Analysis.**  Categorize this token into ONE of the following signal categories:
    * **{SIGNAL_CATEGORIES[0]} (Fast Pump & Dump):**  Expect rapid initial pump, short holding time (minutes to < 2 hours), high volatility, quick profit-taking crucial.
    * **{SIGNAL_CATEGORIES[1]} (Long Hold > 2 Hours):**  Potential for sustained growth beyond initial pump, holding for > 2 hours possible (but still meme coin timeframe, manage risk).
    * **{SIGNAL_CATEGORIES[2]} (Small Profit 50-100%) Play):** More moderate, potentially less volatile, aim for 50-100% profit, could be slower growth than pump & dumps.
    * **{SIGNAL_CATEGORIES[3]} (Not Recommended):**  Avoid this token, low potential, high risk of loss.

    Consider ALL parameters above, trending metas, Twitter trends (general & token-specific) to categorize the token signal AND provide analysis justifying the categorization.

    Factors to guide categorization:
    * Token Name/Ticker & Meta Alignment
    * Initial Holders, MCAP, Bundles
    * General Crypto Sentiment & Meme Coin Trend Strength
    * Twitter Trend Signals

    **Analyze and categorize. Provide:**
    1. **Predicted Signal Category:** Choose ONE category from [{', '.join(SIGNAL_CATEGORIES)}].
    2. **Analysis (2-3 sentences):** Justify category choice.

    **Important: HIGH-RISK, NOT financial advice.**
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cryptocurrency analyst specializing in meme coins on pump.fun."},
                {"role": "user", "content": prompt}
            ]
        )
        openai_analysis = response['choices'][0]['message']['content']

        predicted_category = "Unknown"
        for category in SIGNAL_CATEGORIES:
            if category in openai_analysis:
                predicted_category = category
                break

        return openai_analysis, predicted_category

    except Exception as e:
        error_message = f"Error analyzing token '{token_name}' with OpenAI: {e}"
        logging.error(error_message)
        return error_message, "Error"


def create_signal_image(token_name, predicted_category, holders, mcap, bundles, gemini_analysis, token_link):
    """Generates a signal image with token details and analysis."""
    try:
        img = Image.open(BACKGROUND_IMAGE_PATH).convert("RGBA") # Ensure background is RGBA
        draw = ImageDraw.Draw(img)

        # --- Fonts ---
        header_font = ImageFont.truetype(FONT_PATH, HEADER_FONT_SIZE)
        default_font = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE)

        # --- Coordinates for text elements (adjust as needed) ---
        header_y = 50
        text_start_y = 150
        line_height = DEFAULT_FONT_SIZE + 10
        x_offset = 50

        # --- Header ---
        header_text = "New Token Signal"
        draw.text((x_offset, header_y), header_text, font=header_font, fill=HEADER_COLOR)

        # --- Token Details ---
        current_y = text_start_y
        draw.text((x_offset, current_y), f"Token: {token_name}", font=default_font, fill=TEXT_COLOR)
        current_y += line_height
        draw.text((x_offset, current_y), f"Signal Category: {predicted_category}", font=default_font, fill=TEXT_COLOR)
        current_y += line_height
        draw.text((x_offset, current_y), f"Holders (Solscan API): {holders}  MCAP: ${mcap:,.0f}  Bundles: {bundles}", font=default_font, fill=TEXT_COLOR)
        current_y += line_height
        draw.text((x_offset, current_y), f"Link: {token_link}", font=default_font, fill=TEXT_COLOR)
        current_y += line_height * 2 # Extra space before analysis

        # --- Gemini Analysis ---
        analysis_header_font = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE + 4)
        draw.text((x_offset, current_y), "Gemini Analysis:", font=analysis_header_font, fill=TEXT_COLOR)
        current_y += line_height
        analysis_lines = gemini_analysis.split('\n') # Split analysis into lines for better formatting
        for line in analysis_lines:
            draw.text((x_offset, current_y), line, font=default_font, fill=TEXT_COLOR)
            current_y += line_height

        # --- Disclaimer ---
        disclaimer_font = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE - 2)
        disclaimer_text = "Disclaimer: NOT financial advice. Meme coins are high risk. Signal category is probabilistic."
        draw.text((x_offset, img.height - 50), disclaimer_text, font=disclaimer_font, fill=TEXT_COLOR) # Position at the bottom

        # --- Save Image ---
        image_path = f"{token_name}_signal_image.png" # Save with token name
        img.save(image_path)
        logging.info(f"Signal image created: {image_path}")
        return image_path

    except Exception as e:
        logging.error(f"Error creating signal image: {e}")
        return None # Or handle error as needed


def record_analysis_data(token_data, gemini_analysis, prediction_timestamp, call_outcome=None, signal_category="Unknown"): # Added signal_category
    """Records token analysis data to CSV, now including signal_category and call outcome."""
    header_exists = os.path.exists(DATA_STORAGE_FILE)
    with open(DATA_STORAGE_FILE, mode='a', newline='') as csvfile:
        fieldnames = ['token_name', 'token_link', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'prediction_timestamp', 'call_outcome', 'signal_category', 'token_address'] # Added 'signal_category', 'token_address'
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
            'signal_category': signal_category, # Record predicted signal category
            'token_address': token_data['address'] # Record token address - NEW
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
            logging.warning(f"No initial price data found for {symbol} on {exchange_id} at prediction time.")
            return None

        initial_df = pd.DataFrame(initial_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        initial_price = initial_df['close'].iloc[-1]

        if initial_price == 0:
            logging.error(f"Error: Initial price for {symbol} is 0. Cannot calculate performance.")
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
                        logging.info(f"  Token {token_name} reached target price ({SUCCESS_METRIC_PERCENT_INCREASE*100}%) - Good Call!")
                        break

                    logging.info(f"  Token {token_name}: Current Price: {current_price:.6f}, Target Price: {target_price:.6f}, Max Reached: {max_price_reached:.6f}, Time left: {end_time - current_time}")
                else:
                    logging.warning(f"Warning: No price data returned from exchange for {symbol} at {current_time}.")

            except ccxt.ExchangeError as e:
                logging.error(f"Exchange error while tracking {token_name}: {e}")
            except Exception as e:
                logging.error(f"Error during price tracking for {token_name}: {e}")

            await asyncio.sleep(PRICE_CHECK_INTERVAL_SECONDS)
            current_time = datetime.now()

        final_percent_change = (max_price_reached - initial_price) / initial_price if initial_price != 0 else 0
        logging.info(f"  Token {token_name} Performance Tracking completed. Max Price Reached: {max_price_reached:.6f}, Percent Change: {final_percent_change:.4f}%, Good Call: {is_good_call}")

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
        {good_calls_df.head(3).to_string(columns=['token_name', 'signal_category', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'call_outcome', 'token_address'], index=False)}
        [End of Good Call Data Example]

        Here are examples of *Bad Calls*:
        (Tokens that did NOT achieve >= {SUCCESS_METRIC_PERCENT_INCREASE*100}% increase, showing *predicted signal category*):
        [Start of Bad Call Data Example]
        {bad_calls_df.head(3).to_string(columns=['token_name', 'signal_category', 'holders', 'mcap', 'bundles', 'gemini_analysis', 'call_outcome', 'token_address'], index=False)}
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
        5.  **Update Trending Metas List:** Based on the analysis, suggest updates to the CURRENT_TRENDING_METAS list in the script. Which metas are losing relevance? Are there new metas that should be added based on recent good/bad calls and category analysis?
        6.  **Prompt Refinement for Categories, Metas & Twitter:** Suggest specific prompt improvements to better incorporate 'trending metas', 'general Twitter trend signals', *'token-specific Twitter buzz'*, AND *signal category prediction* factors in `analyze_token_with_gemini`.

        Focus on actionable recommendations to improve 'Good Call' rate *within each signal category* and improve the *accuracy of signal category predictions*.
        """

        response = MODEL_GEMINI.generate_content(prompt)
        gemini_feedback = response.text
        print("\n--- Gemini Feedback on Prediction Performance (Signal Category Analysis): ---\n")
        print(gemini_feedback)

        # --- **Action Required: Manual Review and Update** ---
        print("\n--- **Action Required: Review Gemini's Feedback Above.**")
        print("--- **1. MANUALLY UPDATE the CURRENT_TRENDING_METAS list in the script.**")
        print("--- **2. MANUALLY REFINE the `analyze_token_with_gemini` prompt (especially for category prediction).** ---")
        print("--- **3. Consider refining Twitter scanning for category-specific signals.** ---")


    except Exception as e:
        logging.error(f"Error evaluating prediction performance (Signal Category Analysis): {e}")


async def debate_token_signal(token):
    """Analyzes token with Gemini and potentially OpenAI, sends signal, and records data."""
    token_name = token['name']
    token_link = token['link']
    holders = token['holders'] # Holders now from Solscan API
    mcap = token['mcap']
    bundles = token['bundles']
    token_address = token['address'] # Token address from scraping

    logging.info(f"\n-- Analyzing New Token: {token_name} --")
    logging.info(f"  Link: {token_link}, Holders (Solscan): {holders}, MCAP: ${mcap:.0f}, Bundles: {bundles}, Address: {token_address}")

    twitter_trends_data = scan_twitter_for_trends()
    token_twitter_mentions_data = scan_twitter_for_token_mentions(token_name)

    prediction_timestamp = datetime.now()
    gemini_analysis_result, predicted_category_gemini = analyze_token_with_gemini(
        token_name, token_link, holders, mcap, bundles, twitter_trends_data, token_twitter_mentions_data
    )
    logging.info(f"-- Gemini Analysis: --\n{gemini_analysis_result}")
    logging.info(f"-- Predicted Signal Category (Gemini): {predicted_category_gemini} --")


    openai_analysis_result, predicted_category_openai = "OpenAI Analysis Disabled", "Disabled" # Default if not used
    if USE_OPENAI:
        openai_analysis_result, predicted_category_openai = analyze_token_with_openai(
            token_name, token_link, holders, mcap, bundles, twitter_trends_data, token_twitter_mentions_data
        )
        logging.info(f"-- OpenAI Analysis: --\n{openai_analysis_result}")
        logging.info(f"-- Predicted Signal Category (OpenAI): {predicted_category_openai} --")


    # --- Select Gemini's category for now (can add logic to choose or combine) ---
    predicted_category = predicted_category_gemini
    gemini_analysis_to_use = gemini_analysis_result # Use Gemini analysis for now

    image_path = create_signal_image(token_name, predicted_category, holders, mcap, bundles, gemini_analysis_to_use, token_link) # Create image

    if image_path: # Send image if created successfully
        send_telegram_message(image_path)
        await send_discord_message(image_path)
        os.remove(image_path) # Clean up image after sending
    else: # Fallback to text message if image creation fails
        telegram_message = f"*New Token Analysis*\n\n*Token:* {token_name}\n*Signal Category:* *{predicted_category}*\n*Holders (Solscan API):* {holders}  *MCAP:* ${mcap:,.0f}  *Bundles:* {bundles}\n*Trending Meta Alignment:*\n[Assess]\n*General Twitter Trend Signal:*\n[Assess]\n*Token-Specific Twitter Buzz:*\n[Assess]\n*Link:* {token_link}\n\n*Gemini Analysis (Category Justification):*\n{gemini_analysis_to_use}\n\n*Disclaimer:* NOT financial advice. Meme coins are high risk. Signal category is probabilistic."
        send_telegram_message(text_message=telegram_message) # Send text message as fallback
        await send_discord_message(text_message=telegram_message) # Send text to Discord too


    record_analysis_data(token, gemini_analysis_to_use, prediction_timestamp, signal_category=predicted_category) # Record Gemini analysis and category

    return predicted_category # Return predicted category from chosen model


async def admin_discussion_mode():
    """Interactive admin discussion mode with Gemini for script improvement."""
    print("\n--- Entering Admin Discussion Mode with Gemini ---")
    print("Type 'exit' to end discussion.")

    conversation_history = "" # Keep track of conversation history for context

    while True:
        admin_query = input("\nAdmin Query (for Gemini - script improvement ideas): ")
        if admin_query.lower() == 'exit':
            print("Exiting Admin Discussion Mode.")
            break

        prompt = f"""
        [Previous Conversation History (for context):]
        {conversation_history}

        [Admin's Current Query/Suggestion:]
        {admin_query}

        Continue our discussion about improving the cryptocurrency token scanning and analysis script.
        Focus on actionable improvements to the script's scanning methods, analysis prompts, signal categories,
        performance evaluation, or any other aspect of the script that could be enhanced.

        Provide specific, concise suggestions for improvement based on the admin's query and the conversation history.
        If the query is open-ended, ask clarifying questions to guide the discussion towards concrete improvements.
        """

        try:
            response = MODEL_GEMINI.generate_content(prompt)
            gemini_response = response.text
            print(f"\nGemini Response:\n{gemini_response}")

            # Update conversation history
            conversation_history += f"\n[Admin Query:] {admin_query}\n[Gemini Response:] {gemini_response}\n"

            # Log the admin discussion
            with open(LOG_ADMIN_DISCUSSION_FILE, 'a') as log_file:
                log_file.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                log_file.write(f"[Admin Query:] {admin_query}\n")
                log_file.write(f"[Gemini Response:] {gemini_response}\n")


        except Exception as e:
            error_message = f"Error during admin discussion with Gemini: {e}"
            print(f"Error: {error_message}")
            logging.error(error_message)


async def main(admin_mode=False): # Added admin_mode parameter
    """... (rest of main function) ..."""
    await DISCORD_BOT.login(DISCORD_BOT_TOKEN) # Login Discord bot

    if admin_mode: # Enter admin discussion mode if flag is set
        await admin_discussion_mode()
        await DISCORD_BOT.close() # Close Discord bot if only in admin mode
        return # Exit main function after admin discussion

    print("Starting pump.fun token scanner, Gemini/OpenAI analyzer, and Telegram/Discord relay (Image Signals)...")
    logging.info("Starting script: pump.fun token scanner, Gemini/OpenAI analyzer, and Telegram/Discord relay (Image Signals)...")

    while True:
        logging.info(f"\n--- Scanning for new tokens at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        new_tokens = scrape_pump_fun_new_tokens(PUMP_FUN_NEW_TOKENS_URL)

        if new_tokens:
            logging.info(f"Found {len(new_tokens)} new token(s) scraped from pump.fun.")

            filtered_tokens = []
            for token in new_tokens:
                if token['holders'] >= MIN_HOLDERS_THRESHOLD and token['mcap'] >= MIN_MCAP_THRESHOLD and token['bundles'] <= MAX_BUNDLES_THRESHOLD: # Holders now from Solscan API
                    filtered_tokens.append(token)
                else:
                    logging.info(f"Token '{token['name']}' filtered out - Holders (Solscan): {token['holders']}, MCAP: ${token['mcap']:.0f}, Bundles: {token['bundles']} - Below thresholds.")

            if filtered_tokens:
                logging.info(f"Analyzing {len(filtered_tokens)} token(s) meeting criteria.")
                for token in filtered_tokens:
                    if token['name'] not in SEEN_TOKENS:
                        predicted_category = await debate_token_signal(token) # Analyze and signal, get category
                        SEEN_TOKENS.add(token['name'])
                        asyncio.create_task(track_token_performance_and_update(token['name'], datetime.now())) # Track performance

                    else:
                        logging.info(f"Token '{token['name']}' already analyzed, skipping.")
            else:
                logging.info("No new tokens met holder, MCAP, and bundle criteria.")


        else:
            logging.info("No new tokens found on pump.fun.")

        if datetime.now().minute % 30 == 0:
            print("\n--- Evaluating Prediction Performance (Signal Category Analysis) ---")
            evaluate_prediction_performance()

        logging.info(f"Waiting {SCAN_INTERVAL_SECONDS} seconds before next scan...")
        time.sleep(SCAN_INTERVAL_SECONDS)

    await DISCORD_BOT.close() # Close Discord bot connection at the end


async def track_token_performance_and_update(token_name, prediction_timestamp):
    """... (unchanged) ..."""
    performance_data = await get_token_price_data(token_name, prediction_timestamp)

    if performance_data:
        is_good_call = performance_data['is_good_call']
        call_outcome = "Good" if is_good_call else "Bad"

        logging.info(f"\n--- Performance Tracking completed for Token: {token_name} ---")
        logging.info(f"  Call Outcome: {call_outcome}")

        update_call_outcome_in_csv(token_name, call_outcome)
    else:
        logging.warning(f"\n--- Performance Tracking failed to get data for Token: {token_name} ---")
        update_call_outcome_in_csv(token_name, "DataError")


def update_call_outcome_in_csv(token_name, call_outcome):
    """... (unchanged) ..."""
    try:
        df = pd.read_csv(DATA_STORAGE_FILE)
        df.loc[df['token_name'] == token_name, 'call_outcome'] = call_outcome
        df.to_csv(DATA_STORAGE_FILE, index=False)
        logging.info(f"  CSV updated for Token: {token_name} - Call Outcome set to: {call_outcome}")

    except Exception as e:
        logging.error(f"Error updating CSV with call outcome for {token_name}: {e}")



if __name__ == "__main__":
    import sys # Import sys for command-line arguments

    admin_mode_flag = False
    if len(sys.argv) > 1 and sys.argv[1] == "admin": # Check for "admin" argument
        admin_mode_flag = True
        print("Admin mode activated for discussion with Gemini.")

    asyncio.run(main(admin_mode=admin_mode_flag))

#========================================================================================================
#TOMA AYTAKIN (EMPRESS)
#REGISTERED CODE IN WEBFLOW #2005910000058681 12/14/2025 11:41:30 UTC
#========================================================================================================