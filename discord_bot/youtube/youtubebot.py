import argparse
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
from croniter import croniter
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from models import Channel, IsOnAir, State
from sqlalchemy import or_

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
isdebug = parser.parse_args().debug
current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
log_folder_abspath = os.path.join(grandparent_folder_abspath, "logs")
dbs_folder_abspath = os.path.join(grandparent_folder_abspath, "dbs")
configpath = os.path.join(grandparent_folder_abspath, "pyproject.toml")
basename = os.path.basename(__file__).split(".")[0]
with open(configpath, "rb") as f:
    log_conf = tomllib.load(f).get("logging")
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(log_folder_abspath, f"{basename}.log")
    if isdebug:
        log_conf["handlers"]["consoleHandler"]["level"] = "DEBUG"
        log_conf["handlers"]["fileHandler"]["level"] = "DEBUG"
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_rows", 500)
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
UPDATE_SCHEDULE = os.getenv("YOUTUBE_UPDATE_SCHEDULE")
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


def desc_from_isonair(ser: pd.core.series.Series) -> str:
    """
    Generate a description based on the given pandas series.

    Args:
        ser (pd.core.series.Series): A pandas series.

    Returns:
        str: A description of the broadcast times.
    """
    one_day_ago = datetime.datetime.now() - datetime.timedelta(days=1)
    if ser["isonair"] == IsOnAir.BEFOREONAIR:
        if ser["start_time"] is not None and ser["start_time"] < one_day_ago:
            return "~~Scheduled start time: {}~~ canceled".format(ser["start_time"].strftime("%Y/%m/%d %H:%M"))
        else:
            return "Scheduled start time: {}".format(ser["start_time"].strftime("%Y/%m/%d %H:%M"))
    elif ser["isonair"] == IsOnAir.NOWONAIR:
        if ser["start_time"] is not None and ser["start_time"] < one_day_ago:
            return "Start time: {}\nEnd time: -- (maybe unarchived)".format(
                ser["start_time"].strftime("%Y/%m/%d %H:%M")
            )
        else:
            return "Start time: {}".format(ser["start_time"].strftime("%Y/%m/%d %H:%M"))
    elif ser["isonair"] == IsOnAir.AFTERONAIR:
        return "Start time: {}\nEnd time: {}".format(
            ser["start_time"].strftime("%Y/%m/%d %H:%M"), ser["end_time"].strftime("%Y/%m/%d %H:%M")
        )
    else:
        return


def strtime_to_jstdt(strtime: str) -> datetime.datetime:
    """
    Convert a string representing a time without a timezone to a string representing JST (Japan Standard Time) timezone.

    Args:
        strtime (str): A string representing a time without a timezone.

    Returns:
        str: A string representing the time in JST timezone.
    """
    # youtubeのapiはUTCで返してくるっぽい？
    dt = datetime.datetime.strptime(
        strtime,
        "%Y-%m-%dT%H:%M:%SZ",
    )
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    # dbにtimezoneが乗らないのでawareではなくnaiveを利用する
    time_jst = dt.astimezone(datetime.timezone(datetime.timedelta(hours=9))).replace(tzinfo=None)
    return time_jst


def check_detail(detail: dict) -> tuple:
    """
    Extract and format the necessary information based on the information obtained from the youtube api.

    Args:
        detail (dict): one piece of information list obtained from youtube api.

    Returns:
        tuple: broadcast information, start time, end time, colour.
    """
    is_onair = IsOnAir.NOTLIVE
    start_time = None
    end_time = None
    colour = 0x0099E1
    if "liveStreamingDetails" in detail.keys():
        lsds_keys = detail["liveStreamingDetails"].keys()
        if "actualEndTime" in lsds_keys:
            is_onair = IsOnAir.AFTERONAIR
            start_time = strtime_to_jstdt(detail["liveStreamingDetails"]["actualStartTime"])
            end_time = strtime_to_jstdt(detail["liveStreamingDetails"]["actualEndTime"])
            colour = 0x0099E1
        elif "actualStartTime" in lsds_keys:
            is_onair = IsOnAir.NOWONAIR
            start_time = strtime_to_jstdt(detail["liveStreamingDetails"]["actualStartTime"])
            end_time = None
            colour = 0xFF0000
        elif "scheduledStartTime" in lsds_keys:
            is_onair = IsOnAir.BEFOREONAIR
            start_time = strtime_to_jstdt(detail["liveStreamingDetails"]["scheduledStartTime"])
            end_time = None
            colour = 0x969C9F
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
    await interaction.response.defer(thinking=True)
    logger.info("register command executed! channel id: {}".format(channel_id))
    logger.info("fetch rss feeds.")

    columns = ["video_id", "title", "link", "author"]
    rss_url = "https://www.youtube.com/feeds/videos.xml?channel_id={}".format(channel_id)
    feed = feedparser.parse(rss_url)
    if feed["status"] != 200:
        response = "registration failed: The channel is not found."
        logger.info(response)
        await interaction.followup.send(response)
        return
    entries = feed["entries"]
    data = [[entry["yt_videoid"], entry["title"], entry["link"], entry["author"]] for entry in entries]
    df = pd.DataFrame(data=data, columns=columns).set_index("video_id")
    que = db.session.query(Channel).filter(Channel.channel_id == channel_id)
    if db.session.query(que.exists()).scalar():
        response = "そのチャンネルはすでに登録されています。"
        logger.info("The channel is already registered.")
        await interaction.followup.send(response)
        return
    channel = Channel()
    channel.channel_id = channel_id
    channel.name = entries[0]["author"]
    db.session.add(channel)
    db.session.commit()

    response = "registered: {}\nhttps://www.youtube.com/channel/{}".format(entries[0]["author"], channel_id)
    logger.info("registered {}.".format(entries[0]["author"]))
    await interaction.followup.send(response)

    details = youtube_video_details(df.index.tolist())
    logger.info("called youtube api, {} queries.".format((len(details) + 49) // 50))
    for detail in details:
        (
            df.loc[detail["id"], "isonair"],
            df.loc[detail["id"], "start_time"],
            df.loc[detail["id"], "end_time"],
            df.loc[detail["id"], "colour"],
        ) = check_detail(detail)
    # pd.NaTはデータベースには格納できません
    df.replace(None, inplace=True)
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
    state.start_time = ser["start_time"]
    state.title = ser["title"]
    state.is_onair = ser["isonair"]
    state.message_id = message.id
    db.session.add(state)
    db.session.commit()

    for video_id, row in df[1:].iterrows():
        state = State()
        state.video_id = video_id
        state.start_time = row["start_time"]
        state.title = row["title"]
        state.is_onair = row["isonair"]
        state.message_id = None
        db.session.add(state)
        db.session.commit()


# 毎時15分に確認
@tasks.loop(seconds=60)
async def loop():
    """Loop to check the status of registered channels."""
    if not croniter.match(UPDATE_SCHEDULE, datetime.datetime.now()):
        return
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
    # データベースの中で、feedにない、かつ、AFTERONAIRかNOTLIVEのものを削除（古いのを消す）
    que = db.session.query(State).filter(
        ~State.video_id.in_(df.index.tolist()),
        or_(State.is_onair == IsOnAir.AFTERONAIR, State.is_onair == IsOnAir.NOTLIVE),
    )
    deleted_row_count = que.count()
    que.delete(synchronize_session=False)
    db.session.commit()
    logger.info("deleted {} old rows in the database.".format(deleted_row_count))
    # feedにない、かつ、NOWONAIRかBEFOREONAIRはunarciveか消されたもの 終わったことにする 空なこともある
    que = db.session.query(State).filter(
        ~State.video_id.in_(df.index.tolist()),
        or_(State.is_onair == IsOnAir.BEFOREONAIR, State.is_onair == IsOnAir.NOWONAIR),
    )
    states = que.all()
    que.delete(synchronize_session=False)
    db.session.commit()
    data = [
        [
            state.video_id,
            state.title,
            f"https://www.youtube.com/watch?v={state.video_id}",
            IsOnAir.AFTERONAIR,
            state.start_time,
            None,
            0x0099E1,
        ]
        for state in states
    ]
    df_deleted = pd.DataFrame(
        data=data, columns=["video_id", "title", "link", "isonair", "start_time", "end_time", "colour"]
    ).set_index("video_id")
    # データベースの中の動画と放送終了のものを除いたdataframeを作成
    states = (
        db.session.query(State).filter(State.is_onair != IsOnAir.NOTLIVE, State.is_onair != IsOnAir.AFTERONAIR).all()
    )
    data = [[state.video_id, state.title, f"https://www.youtube.com/watch?v={state.video_id}"] for state in states]
    df_fromdb = pd.DataFrame(data=data, columns=columns[:-1]).set_index("video_id")
    # feedからデータベースとかぶってるが、タイトルが違うものを置き換える（実際には全部書き換えているが）
    df_fromdb.title.loc[df_fromdb.index.isin(df.index)] = df.title.loc[df.index.isin(df_fromdb.index)]
    # データベースを引っ張ってくる
    que = db.session.query(State).all()
    # feedからデータベースとかぶってる部分を消す
    df = df.drop(list(map(lambda row: row.video_id, que)), axis=0, errors="ignore")
    # feedの新規とdatabaseの放送前か放送中のデータだけのdataframeができる
    logger.info("concatenate database and feed.")
    df = pd.concat([df, df_fromdb], axis=0)
    df = df[::-1]
    # apiをたたく
    details = youtube_video_details(df.index.tolist())
    logger.info("called youtube api with {} rows, {} queries.".format(len(details), (len(details) + 49) // 50))
    # 時間情報の格納
    for detail in details:
        # まとめるとdtypeが異なるのでfuture warningが出る
        (
            df.loc[detail["id"], "isonair"],
            df.loc[detail["id"], "start_time"],
            df.loc[detail["id"], "end_time"],
            df.loc[detail["id"], "colour"],
        ) = check_detail(detail)
    # 削除予定のものと組み合わせる
    if not df_deleted.empty:
        logger.debug("df_deleted is noe empty!")
        df = pd.concat([df, df_deleted], axis=0)

    discord_channel = client.get_channel(int(CHANNEL_ID))
    for video_id, row in df.iterrows():
        # Stateテーブルに存在するか
        que = db.session.query(State).filter(State.video_id == video_id)
        description = desc_from_isonair(row)
        if db.session.query(que.exists()).scalar():
            # 存在すれば、生放送状態が変化したか確認
            state = que.first()
            if state.message_id is None:
                # メッセージidがなければ、投稿されてない
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
                message = await discord_channel.send(embed=embed)
                state.message_id = message.id
            else:
                # 投稿はされてる
                message = await discord_channel.fetch_message(state.message_id)
                # 投稿を取得
                embed = message.embeds[0]

                title = row["title"]
                is_onair = row["isonair"]
                start_time = row["start_time"]
                colour = row["colour"]

                logger_message = f"edit {state.message_id}, {title}:\n"

                one_day_ago = datetime.datetime.now() - datetime.timedelta(days=1)

                if row["start_time"] is not None and row["start_time"] < one_day_ago:
                    # 予定消失もしくは削除された生放送なら
                    embed.colour = discord.Colour(int(colour))
                    embed.description = description
                    state.is_onair = is_onair
                    logger_message += f"- deleted: {embed.description} >> {description}"
                elif state.title != row["title"]:
                    # タイトルが変わっていたら
                    logger_message += f"- title changed: {embed.title} >> {title}"
                    embed.title = title
                    state.title = title
                elif state.is_onair != is_onair:
                    # 放送状態が変わっていたら
                    logger_message += f"- status changed: {embed.description} >> {description}"
                    embed.description = description
                    embed.colour = discord.Colour(int(colour))
                    state.is_onair = is_onair
                    state.start_time = start_time
                elif state.start_time is None or state.start_time != start_time:
                    # 時間が変わっていたら
                    logger_message += f"schedule changed: {embed.description} >> {description}"
                    embed.description = description
                    state.start_time = start_time
                else:
                    continue
                await message.edit(embed=embed)
            logger.info(logger_message)
            db.session.add(state)
            db.session.commit()
        else:
            # テーブルにないので、書き込み
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
            message = await discord_channel.send(embed=embed)
            state = State()
            state.start_time = row["start_time"]
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
