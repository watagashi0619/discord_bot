import datetime
import importlib
import os
import tomllib
from logging import config, getLogger

import db
import discord
import feedparser
import pandas as pd
from apiclient.discovery import build
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from models import Channel, IsOnAir, State
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

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN_YOUTUBE")
CHANNEL_ID = os.getenv("CHANNEL_ID_YOUTUBE")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# はじめて実行された時はdbを作成する
if not os.path.exists(os.path.join(dbs_folder_abspath, os.path.basename(current_folder_abspath) + ".db")):
    importlib.import_module("models").create_db()
    logger.info("created database.")


# https://zenn.dev/meihei/articles/1021b1a3f8c226
# https://console.cloud.google.com/apis/credentials?project=iconic-treat-403315
def youtube_video_details(video_ids: list[str]) -> list:
    """
    Get the broadcast time information from the YouTube API based on the video ID.

    Args:
        video_ids (list[str]): A list of YouTube video IDs.

    Returns:
        list: A list of video responses.
    """
    video_response = []
    for i in range(0, len(video_ids), 50):
        video_response += (
            youtube.videos()
            .list(
                id=",".join(video_ids[i : min(i + 50, len(video_ids))]),
                part="liveStreamingDetails",
                maxResults=5,
            )
            .execute()
            .get("items", [])
        )
    return video_response


def livestreamingdetailstime_tojst(strtime: str) -> str:
    """
    Convert a string representing a time without a timezone to a string representing JST (Japan Standard Time) timezone.

    Args:
        strtime (str): A string representing a time without a timezone.

    Returns:
        str: A string representing the time in JST timezone.
    """
    dt = datetime.datetime.strptime(
        strtime,
        "%Y-%m-%dT%H:%M:%SZ",
    )
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    time_jst = dt.astimezone(datetime.timezone(datetime.timedelta(hours=9)))
    return time_jst.strftime("%Y/%m/%d %H:%M")


def desc_from_isonair(ser: pd.core.series.Series) -> str:
    """
    Generate a description based on the given pandas series.

    Args:
        ser (pd.core.series.Series): A pandas series.

    Returns:
        str: A description of the broadcast times.
    """
    if ser["isonair"] == IsOnAir.BEFOREONAIR:
        return "Scheduled start time: {}".format(ser["start_time"])
    elif ser["isonair"] == IsOnAir.NOWONAIR:
        return "Start time: {}".format(ser["start_time"])
    elif ser["isonair"] == IsOnAir.AFTERONAIR:
        return "Start time: {}\nEnd time: {}".format(ser["start_time"], ser["end_time"])
    else:
        return


def check_detail(detail: dict) -> tuple:
    """
    Extract and format the necessary information based on the information obtained from the youtube api.

    Args:
        detail (dict): one piece of information list obtained from youtube api.

    Returns:
        tuple: broadcast information, start time, end time, colour.
    """
    if "liveStreamingDetails" in detail.keys():
        lsds_keys = detail["liveStreamingDetails"].keys()
        if "actualEndTime" in lsds_keys:
            is_onair = IsOnAir.AFTERONAIR
            start_time = livestreamingdetailstime_tojst(detail["liveStreamingDetails"]["actualStartTime"])
            end_time = livestreamingdetailstime_tojst(detail["liveStreamingDetails"]["actualEndTime"])
            colour = 0x0099E1
        elif "actualStartTime" in lsds_keys:
            is_onair = IsOnAir.NOWONAIR
            start_time = livestreamingdetailstime_tojst(detail["liveStreamingDetails"]["actualStartTime"])
            end_time = None
            colour = 0xFF0000
        elif "scheduledStartTime" in lsds_keys:
            is_onair = IsOnAir.BEFOREONAIR
            start_time = livestreamingdetailstime_tojst(detail["liveStreamingDetails"]["scheduledStartTime"])
            end_time = None
            colour = 0x969C9F
    else:
        is_onair = IsOnAir.NOTLIVE
        start_time = None
        end_time = None
        colour = 0x0099E1
    return is_onair, start_time, end_time, colour


@client.event
async def on_ready():
    """on ready"""
    print(f"We have logged in as {client.user}")
    loop.start()
    await tree.sync()


@tree.command(name="youtube-registration-list", description="youtube botに登録されているチャンネル一覧")
async def registered_list(interaction: discord.Interaction):
    """
    Get a list of channels registered with the YouTube bot.

    Args:
        interaction (discord.Interaction): The interaction object.

    Returns:
        None
    """
    logger.info("registered list command executed!")
    registch_rows = db.session.query(Channel).all()
    content = "登録チャンネル一覧：\n"
    for registch_row in registch_rows:
        content += "- {}\n".format(registch_row.name)
    await interaction.response.send_message(content=content)


@tree.command(name="youtube-unregister", description="youtube botから登録解除する")
async def youtube_unregister(interaction: discord.Interaction, input_string: str):
    """
    Unsubscribe from channels that are registered with the YouTube bot.

    Args:
        interaction (discord.Interaction): The interaction object.
        input_string (str): The channel ID or channel name that you wish to unsubscribe from.

    Returns:
        None
    """
    logger.info("unregister executed!")
    que = db.session.query(Channel).filter(or_(Channel.channel_id == input_string, Channel.name == input_string))
    if db.session.query(que.exists()).scalar():
        content = "「{}」の登録を解除しました。".format(que.first().name)
        que.delete(synchronize_session=False)
        db.session.commit()
    else:
        content = "「{}」に対応する登録チャンネルは存在しませんでした。".format(input_string)
    logger.info(content)
    await interaction.response.send_message(content=content)


@tree.command(name="youtube-register", description="youtube botに登録する")
async def youtube_register(interaction: discord.Interaction, channel_id: str):
    """
    Subscribe channels to the database using the YouTube bot.

    Args:
        interaction (discord.Interaction): The interaction object.
        channel_id (str): The channel ID you wish to register.

    Returns:
        None
    """
    logger.info("register command executed! channel id: {}".format(channel_id))
    logger.info("fetch rss feeds.")

    columns = ["video_id", "title", "link", "author"]
    rss_url = "https://www.youtube.com/feeds/videos.xml?channel_id={}".format(channel_id)
    feed = feedparser.parse(rss_url)
    if feed["status"] != 200:
        response = "registration failed: The channel is not found."
        logger.info(response)
        await interaction.response.send_message(response)
        return
    entries = feed["entries"]
    data = [[entry["yt_videoid"], entry["title"], entry["link"], entry["author"]] for entry in entries]
    df = pd.DataFrame(data=data, columns=columns).set_index("video_id")

    que = db.session.query(Channel).filter(Channel.channel_id == channel_id)
    if db.session.query(que.exists()).scalar():
        response = "そのチャンネルはすでに登録されています。"
        logger.info("The channel is already registered.")
        await interaction.response.send_message(response)
        return
    channel = Channel()
    channel.channel_id = channel_id
    channel.name = entries[0]["author"]
    db.session.add(channel)
    db.session.commit()

    response = "registered: {}\nhttps://www.youtube.com/channel/{}".format(entries[0]["author"], channel_id)
    logger.info("registered {}.".format(entries[0]["author"]))
    await interaction.response.send_message(response)

    details = youtube_video_details(df.index.tolist())
    logger.info("called youtube api, {} queries.".format((len(details) + 49) // 50))
    for detail in details:
        (
            df.loc[detail["id"], "isonair"],
            df.loc[detail["id"], "start_time"],
            df.loc[detail["id"], "end_time"],
            df.loc[detail["id"], "colour"],
        ) = check_detail(detail)

    discord_channel = client.get_channel(int(CHANNEL_ID))
    ser = df.iloc[0, :]
    description = desc_from_isonair(ser)
    embed = discord.Embed(
        title=ser["title"],
        colour=discord.Colour(int(ser["colour"])),
        url=ser["link"],
        description=description,
    )
    embed.set_image(url="https://avatar-resolver.vercel.app/youtube-thumbnail/q?url={}".format(ser["link"]))
    embed.set_author(
        name=ser["author"],
        icon_url="https://avatar-resolver.vercel.app/youtube-avatar/q?url={}".format(ser["link"]),
    )
    message = await discord_channel.send(embed=embed)
    state = State()
    state.video_id = df.index[0]
    state.title = ser["title"]
    state.is_onair = ser["isonair"]
    state.message_id = message.id
    db.session.add(state)
    db.session.commit()

    for video_id, row in df[1:].iterrows():
        state = State()
        state.video_id = video_id
        state.title = row["title"]
        state.is_onair = row["isonair"]
        state.message_id = None
        db.session.add(state)
        db.session.commit()


# 15分ごとに確認
@tasks.loop(seconds=60 * 15)
async def loop():
    """Loop to check the status of registered channels."""
    # feedから引っ張ってくる
    logger.info("start loop.")
    logger.info("fetch database rows.")
    registch_rows = db.session.query(Channel).all()
    logger.info("fetch rss feeds.")
    columns = ["video_id", "title", "link", "author"]
    df = pd.DataFrame(columns=columns).set_index("video_id")
    for registch_row in registch_rows:
        rss_url = "https://www.youtube.com/feeds/videos.xml?channel_id={}".format(registch_row.channel_id)
        feed = feedparser.parse(rss_url)
        entries = feed["entries"]
        data = [[entry["yt_videoid"], entry["title"], entry["link"], entry["author"]] for entry in entries]
        df = pd.concat([df, pd.DataFrame(data=data, columns=columns).set_index("video_id")], axis=0)
    # データベースの中で、feedにないものを削除（古いのを消す）
    que = db.session.query(State).filter(~State.video_id.in_(df.index.tolist()))
    deleted_row_count = que.count()
    que.delete(synchronize_session=False)
    db.session.commit()
    logger.info("deleted {} old rows in the database.".format(deleted_row_count))
    # データベースを引っ張ってくる
    que = db.session.query(State).all()
    # feedからデータベースとかぶってる部分を消す
    df = df.drop(list(map(lambda row: row.video_id, que)), axis=0, errors="ignore")
    # 結合リストを作る、ただし、データベースの中の動画と放送終了のものを除く
    states = (
        db.session.query(State).filter(State.is_onair != IsOnAir.NOTLIVE, State.is_onair != IsOnAir.AFTERONAIR).all()
    )
    data = [
        [
            state.video_id,
            state.title,
            "https://www.youtube.com/watch?v={}".format(state.video_id),
            "dummy author",
        ]
        for state in states
    ]
    df = pd.concat([df, pd.DataFrame(data=data, columns=columns).set_index("video_id")], axis=0)
    df = df[::-1]

    logger.info("concatenate database and feed.")
    # apiをたたく
    details = youtube_video_details(df.index.tolist())
    logger.info("called youtube api with {} rows, {} queries.".format(len(details), (len(details) + 49) // 50))

    for detail in details:
        # まとめるとdtypeが異なるのでfuture warningが出る
        (
            df.loc[detail["id"], "isonair"],
            df.loc[detail["id"], "start_time"],
            df.loc[detail["id"], "end_time"],
            df.loc[detail["id"], "colour"],
        ) = check_detail(detail)

    discord_channel = client.get_channel(int(CHANNEL_ID))
    for video_id, row in df.iterrows():
        description = desc_from_isonair(row)
        embed = discord.Embed(
            title=row["title"],
            colour=discord.Colour(int(row["colour"])),
            url=row["link"],
            description=description,
        )
        embed.set_image(url="https://avatar-resolver.vercel.app/youtube-thumbnail/q?url={}".format(row["link"]))
        embed.set_author(
            name=row["author"],
            icon_url="https://avatar-resolver.vercel.app/youtube-avatar/q?url={}".format(row["link"]),
        )
        # Stateテーブルに存在するか
        que = db.session.query(State).filter(State.video_id == video_id)
        if db.session.query(que.exists()).scalar():
            # 存在すれば、生放送状態が変化したか確認
            state = que.first()
            if state.is_onair != row["isonair"]:
                # 生放送ステータスが変わっていたら
                state.is_onair = row["isonair"]
                if state.message_id is None:
                    # メッセージidがない=投稿されてない
                    message = await discord_channel.send(embed=embed)
                    state.message_id = message.id
                else:
                    message = await discord_channel.fetch_message(state.message_id)
                    embed = message.embeds[0]
                    logger_message = "edit {}: {}, {} >> {}".format(
                        state.message_id, embed.title, embed.description, description.replace("\n", "\\n")
                    )
                    embed.description = description
                    embed.colour = discord.Colour(int(row["colour"]))
                    await message.edit(embed=embed)
                    logger.info(logger_message)
                db.session.add(state)
                db.session.commit()
        else:
            # テーブルにないので、書き込み
            message = await discord_channel.send(embed=embed)
            state = State()
            state.video_id = video_id
            state.title = row["title"]
            state.is_onair = row["isonair"]
            state.message_id = message.id
            db.session.add(state)
            db.session.commit()
            logger.info("commit {}".format(row["title"]))
    logger.info("loop accomplished.")


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
