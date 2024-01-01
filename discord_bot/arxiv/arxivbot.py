import datetime
import importlib
import json
import os
import re
import tomllib
from logging import config, getLogger
from urllib import parse
from urllib.request import Request, urlopen

import dateutil.parser
import db
import discord
import feedparser
import requests
from croniter import croniter
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from models import Feed
from sqlalchemy import or_

current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
log_folder_abspath = os.path.join(grandparent_folder_abspath, "logs")
dbs_folder_abspath = os.path.join(grandparent_folder_abspath, "dbs")
configpath = os.path.join(grandparent_folder_abspath, "pyproject.toml")
basename = os.path.basename(__file__).split(".")[0]
with open(configpath, "rb") as f:
    log_conf = tomllib.load(f).get("logging")
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(log_folder_abspath, f"{basename}.log")
logger = getLogger(__name__)
config.dictConfig(log_conf)
dotenvpath = os.path.join(grandparent_folder_abspath, ".env")
load_dotenv(dotenvpath)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN_ARXIV")
GAE_TRANSLATE_URL = os.getenv("GAE_TRANSLATE_URL")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
CHANNEL_ID_ARXIVEXPORT = os.getenv("CHANNEL_ID_ARXIVEXPORT")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

logger.info("Start arXiv notificator.")

# はじめて実行された時はdbを作成する
if not os.path.exists(os.path.join(dbs_folder_abspath, os.path.basename(current_folder_abspath) + ".db")):
    importlib.import_module("models").create_db()
    logger.info("created database.")


def translated_en_to_jp(en_text: str) -> str:
    """
    Translates the given English text to Japanese using an external translation service. See https://script.google.com/home .

    Args:
        en_text (str): The English text to be translated.

    Returns:
        str: The translated Japanese text.

    Raises:
        HTTPError: If there is an error in the HTTP request.
        URLError: If there is an error in the URL.

    Examples:
        >>> translated_en_to_jp("Hello, how are you?")
        'こんにちは、お元気ですか？'
        >>> translated_en_to_jp("Goodbye!")
        'さようなら！'

    Note:
        This function uses an external translation service. Please refer to the service provider's documentation for more details.
    """
    data = parse.urlencode({"text": en_text, "source": "en", "target": "ja"}).encode()
    request = Request(GAE_TRANSLATE_URL, data=data)
    with urlopen(request) as res:
        return res.read().decode()


async def updated(feed: feedparser.FeedParserDict, channel: discord.abc.GuildChannel) -> bool:
    """
    Check if the feed has been updated within the last 24 hours and send a notification to the specified channel.

    Args:
        feed (feedparser.FeedParserDict): A parsed feed object representing the feed to check for updates.
        channel (discord.abc.GuildChannel): The Discord guild channel to send the notification to.

    Returns:
        bool: True if there are new updates, False if there are no updates.

    """
    today = str(datetime.datetime.now().strftime("%Y/%m/%d"))
    one_day_ago = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=+9), "JST")) - datetime.timedelta(
        days=1
    )
    feed_updated_time = dateutil.parser.parse(feed.updated)
    if one_day_ago < feed_updated_time:
        # 更新あり
        channel_name = channel.name
        feed_len = len(feed.entries)
        logger.info(f"{today}: {channel_name} has {feed_len} new feeds.")
        content = "-" * 48 + f"\n{today}\n{channel_name} has {feed_len} new feeds.\n" + "-" * 48
        await channel.send(content=content)
        return True
    else:
        # 更新なし
        content = "-" * 48 + f"\n{today}\nNo update today.\n" + "-" * 48
        logger.info(f"{today}: no updated.")
        await channel.send(content=content)
        return False


async def feed_post(channel_id: str, rss_url: str):
    """
    Fetch the RSS feed from the provided URL, check for updates, and post the new entries to the specified Discord channel.

    Args:
        channel_id (str): The ID of the Discord channel where the feed entries will be posted.
        rss_url (str): The URL of the RSS feed to fetch and check for updates.

    """
    discord_channel = client.get_channel(int(channel_id))
    logger.info(f"start loop: {discord_channel.name}")
    feed = feedparser.parse(rss_url)
    is_updated = await updated(feed, discord_channel)
    if is_updated:
        for entry in feed.entries:
            title = re.search(r"^([^\(]+)", entry.title).group(1).strip()
            doi = re.search(r"\(([^\)]+)\)$", entry.title).group(1).strip()
            href = entry.id
            abstract = re.sub("<.*?>", "", entry.summary.replace("\n", " "))
            authors = ",".join([re.sub("<.*?>", "", author.name) for author in entry.authors])

            translated_text = translated_en_to_jp(abstract)
            content = (
                f"**Title**: [{title}]({href})\n"
                f"**doi**: {doi}\n"
                f"**Authors**: {authors}\n"
                f"**Abstract** (translated by Google)\n"
                f">>> {translated_text}"
            )
            await discord_channel.send(content=content)

    logger.info(f"{discord_channel.name} has done.")


@tasks.loop(seconds=60)
async def loop():
    """
    Check the registered feeds against the current time and post updates to the corresponding Discord channels.

    This function retrieves the registered feeds from the database and compares the scheduled time for each feed with the current time. If the current time matches the scheduled time, the function calls the `feed_post` function to fetch and post any new entries for that feed to the associated Discord channel.

    """
    registered_rows = db.session.query(Feed).all()
    now = datetime.datetime.now()
    for row in registered_rows:
        if croniter.match(row.scheduled_time, now):
            await feed_post(row.discord_channel_id, row.link)


@client.event
async def on_ready():
    """Executes when the bot is ready to start processing events."""
    print(f"We have logged in as {client.user}")
    loop.start()
    await tree.sync()


def post_to_notion_database(title: str, authors: str, link: str):
    """
    Posts the given information to a Notion database.

    Args:
        title (str): The title of the page.
        authors (str): The authors of the content.
        link (str): The URL link of the content.

    Returns:
        None

    Raises:
        None
    """
    notion_page_url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    data = json.dumps(
        {
            "parent": {"database_id": f"{NOTION_DATABASE_ID}"},
            "properties": {
                "link": {"rich_text": [{"text": {"content": f"{link}", "link": {"url": f"{link}"}}}]},
                "名前": {"title": [{"text": {"content": f"{title}"}}]},
                "Author": {"rich_text": [{"text": {"content": f"{authors}"}}]},
            },
        }
    )

    response = requests.post(notion_page_url, headers=headers, data=data)
    if response.status_code == 200:
        logger.info(f"successed to export to notion: {link}")
    else:
        logger.info(f"failed to export to notion:\n {response}")


@client.event
async def on_raw_reaction_add(payload):
    """
    This function is called when a user adds a reaction to a message.

    Args:
        payload: The information about the reaction event.

    Returns:
        None

    Raises:
        None
    """
    # リアクションされたメッセージのチャンネル
    txt_channel = client.get_channel(payload.channel_id)
    # リアクションされたメッセージ
    message = await txt_channel.fetch_message(payload.message_id)
    # リアクションしたユーザ
    user = payload.member
    if message.author == user:
        return
    registered = db.session.query(Feed).all()
    registered_channel_ids = [row.discord_channel_id for row in registered]

    if str(payload.channel_id) in registered_channel_ids:
        title = message.content.split("\n")[0].split(": ")[1]
        authors = message.content.split("\n")[2].split(": ")[1]
        match = re.search(r"\[(.*?)\]\((.*?)\)", title)
        title_text = match.group(1)
        link = match.group(2)

        content = f"{payload.emoji} {message.jump_url} {title}"
        channel = client.get_channel(payload.channel_id)
        if CHANNEL_ID_ARXIVEXPORT is not None:
            channel = client.get_channel(int(CHANNEL_ID_ARXIVEXPORT))

        await channel.send(content)
        logger.info(f"reaction detected! {content}")

        if NOTION_TOKEN is not None:
            post_to_notion_database(title_text, authors, link)


@tree.command(name="arxiv-registration-list", description="arxiv botに登録されているチャンネル一覧")
async def registered_list(interaction: discord.Interaction):
    """
    Get a list of registered RSS feeds and send it as a response.

    Args:
        interaction (discord.Interaction): The user interaction object.

    Returns:
        None: This function doesn't return anything.

    Raises:
        None: This function doesn't raise any exceptions.

    Notes:
        - This function retrieves all the registered RSS feeds from the database.
        - It creates a formatted string representation of the registered feeds.
        - Each feed is listed with its title and scheduled time in the response message.
        - The function sends the list of registered feeds as the response to the user's interaction.
    """
    logger.info("registered list command executed!")
    registered = db.session.query(Feed).all()
    content = "登録RSSフィード一覧: \n"
    for row in registered:
        content += f"- {row.title} ({row.scheduled_time})\n"
    await interaction.response.send_message(content=content)


@tree.command(name="arxiv-unregister", description="arxiv botから登録解除する")
async def unregister(interaction: discord.Interaction, input_string: str):
    """
    Unregister a feed based on user interaction.

    Args:
        interaction (discord.Interaction): The user interaction object.
        input_string (str): The input string representing the feed to unregister (channel_id, title, or link).

    Returns:
        None: This function doesn't return anything.

    Raises:
        None: This function doesn't raise any exceptions.

    Notes:
        - This function removes a registered feed from the database based on the provided input.
        - The input string can be the channel ID, title, or link of the feed.
        - If a feed matching the input string is found and removed, a success message is sent as the response.
        - If no feed matching the input string is found, an error message is sent as the response.
        - The function sends a message back as the response to the user's interaction.
    """
    logger.info("unregister command executed!")
    que = db.session.query(Feed).filter(
        or_(Feed.discord_channel_id == input_string, Feed.title == input_string, Feed.link == input_string)
    )
    if db.session.query(que.exists()).scalar():
        que.delete(synchronize_session=False)
        db.session.commit()
        content = f"「{que.first().title}」の登録を解除しました。"
        logger.info(f"unregistered {que.first().title}.")
    else:
        content = f"「{input_string}」に対応する登録RSSフィードは存在しませんでした。"
        logger.info(f'There were no registered RSS feeds corresponding to "{input_string}".')
    logger.info(content)
    await interaction.response.send_message(content=content)


@tree.command(name="arxiv-change-scheduling", description="arxiv botのcron設定を変更する")
async def update_cron_schedule(interaction: discord.Interaction, cron_scheduled_time: str):
    """
    Change the scheduled time for a cron configuration based on user interaction.

    Args:
        interaction (discord.Interaction): The user interaction object.
        cron_scheduled_time (str): The new cron scheduled time to be set.

    Returns:
        None: This function doesn't return anything.

    Raises:
        None: This function doesn't raise any exceptions.

    Notes:
        - This function updates the scheduled time of a cron configuration in the database.
        - If the new cron scheduled time is not valid, an error message is sent as the response.
        - The function sends a message back as the response to the user's interaction.
    """
    logger.info("change scheduling command executed!")
    if croniter.is_valid(cron_scheduled_time):
        channel_id = interaction.channel_id
        que = db.session.query(Feed).filter(Feed.discord_channel_id == channel_id)
        if db.session.query(que.exists()).scalar():
            row = que.first()
            row.scheduled_time = cron_scheduled_time
            db.session.commit()
            logger.info(f"Cron setting changed to {cron_scheduled_time}.")
            row = db.session.query(Feed).filter(Feed.discord_channel_id == channel_id).first()
            channel = client.get_channel(int(channel_id))
            topic = f"{row.title} {row.link} (update schedule: {row.scheduled_time})"
            await channel.edit(topic=topic)
            logger.info(f'Channel topic changed to "{topic}".')
            content = f"cron設定とチャンネルトピックを変更しました: {cron_scheduled_time}"
        else:
            logger.info("There were no registered RSS feeds corresponding to this channel.")
            content = "このチャンネルに対応する登録RSSフィードは存在しませんでした。"
        await interaction.response.send_message(content=content)
    else:
        response = "Configuration change failed: cron_scheduled_time is not valid."
        logger.info(response)
        await interaction.response.send_message(response)


@tree.command(name="arxiv-register", description="arxiv botに登録する")
async def register(interaction: discord.Interaction, link: str, cron_scheduled_time: str):
    """Registers an RSS feed to a Discord channel and stores the information in a database.

    Args:
        interaction (discord.Interaction): User interaction object.
        link (str): URL of the RSS feed.
        cron_scheduled_time (str): Cron expression for scheduling updates.

    Returns:
        None

    Raises:
        None

    """
    logger.info(f"register command executed! link: {link}, scheduled_time: {cron_scheduled_time}")
    await interaction.response.defer()

    feed = feedparser.parse(link)
    # validation of the link
    if feed["status"] != 200 and feed["status"] != 301:
        response = f"registration failed: The RSS feed is not found. status code: {feed['status']}"
        logger.info(response)
        await interaction.followup.send(response)
        return

    feed_title = feed["feed"]["title"]

    # validation of the cron expression
    if not croniter.is_valid(cron_scheduled_time):
        response = "registration failed: cron_scheduled_time is not valid."
        logger.info(response)
        await interaction.followup.send(response)
        return

    # existence of the database
    que = db.session.query(Feed).filter(Feed.link == link)
    if db.session.query(que.exists()).scalar():
        response = "そのチャンネルはすでに登録されています。"
        logger.info("The channel is already registered.")
        await interaction.followup.send(response)
        return

    # discordのチャンネルを特定のcategoryに作成する処理
    guild = client.guilds[0]
    title = feed_title.split()[0].replace(".", "-")
    category_name = "ARXIV"
    for _category in guild.categories:
        if _category.name == category_name:
            category = _category
            break
    else:
        category = await guild.create_category(category_name)
    channel = await guild.create_text_channel(name=title, category=category)
    await channel.edit(topic=f"{title} {link} (update schedule: {cron_scheduled_time})")
    discord_channel_id = channel.id

    db_feed_row = Feed()
    db_feed_row.discord_channel_id = discord_channel_id
    db_feed_row.title = title
    db_feed_row.link = link
    db_feed_row.scheduled_time = cron_scheduled_time
    db.session.add(db_feed_row)
    db.session.commit()

    logger.info(f"registered accomplished! {title}")

    channel_link = f"https://discord.com/channels/{guild.id}/{channel.id}"
    content = f'Created {title} ch. with scheduled "{cron_scheduled_time}" setting! Please visit -> {channel_link}'
    await interaction.followup.send(content)


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
