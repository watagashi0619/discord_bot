# Discord bot

それぞれが独立したbotになっています。

## GPT bot (`gpt/gptbot.py`)

特定のチャンネルにChatGPTを常駐させます。

### command

- `gpt-hflush`: ChatGPTのチャット履歴を消去する
- `gpt-switch`: 言語モデルをgpt-3.5-turboとgpt-4との間でスイッチングする
- `gpt-system`: ChatGPTのキャラクター設定をする
- `gpt`: 常駐チャンネルに限らず任意のチャンネルでGPTを呼び出す

## YouTube Notificator (`youtube/youtubebot.py`)

監視登録チャンネルのRSSに更新があった場合にチャンネルに情報を流します。

時間情報の取得のためにYouTube APIを叩く必要があります。叩きすぎるとrate limitに引っかかるので、15分に1回の設定にしています。

- 灰色線: 放送前
- 赤線: 放送中
- 青線: 放送後 or 動画

`youtube/db.py`, `youtube/models.py` が依存ファイルです。

### command

- `youtube-register`: 監視リストに登録する
- `youtube-unregister`: 監視リストから登録解除する
- `youtube-registeration-list`: 監視リストを一覧表示する

## arXiv Notificator (`arxiv/arxivbot.py`)

arXivの対象フィードを定期的に見に行って、チャンネルに情報を流します。

Notionと連携で、リアクションを押した場合にNotionのデータベースに流せます。

フィードはコマンドで登録します。対応するチャンネルがbotにより自動で生成されます。

`arxiv/db.py`, `arxiv/models.py` が依存ファイルです。

### command

- `arxiv-register`: 監視リストに登録する
    - arXivのRSSのリンクと、RSSを見にいく時間をcron式で記述
- `arxiv-unregister`: 監視リストから登録解除する
- `arxiv-registeration-list`: 監視リストを一覧表示する
- `arxiv-change-scheduling`: RSSを見にいく時間の設定を変更する（cron式で記述）

## 設定

- `.env` の設定
    - `OPENAI_API_KEY`: ChatGPTキー
    - `DISCORD_BOT_TOKEN_GPT`: DiscordのChatGPT botトークン
    - `DISCORD_BOT_TOKEN_YOUTUBE`: DiscordのYouTube botトークン
    - `DISCORD_BOT_TOKEN_ARXIV`: DiscordのarXiv botトークン
    - `YOUTUBE_API_KEY`: YouTube api v3を叩く用 GCPで取得する
    - `GAE_TRANSLATE_URL`: GAEで自分で設置するGoogle TranslateしてくれるAPI
    - `CHANNEL_ID_GPT`: GPT botが投稿するチャンネルのID
    - `CHANNEL_ID_YOUTUBE`: YouTube botが投稿するチャンネルのID
    - (optional) `NOTION_TOKEN`: Notionのトークン（arXiv botでリアクションしたらNotionに転送したい時用）
    - (optional) `NOTION_DATABASE_ID`: Notionのデータベースのid（arXiv botでリアクションしたらNotionに転送したい時用）
    - (optional) `CHANNEL_ID_ARXIVEXPORT`: arXiv botでリアクションをした投稿を転送するチャンネルのID（デフォルトは同じチャンネル）
- logの設定
    - `pyproject.toml` にログの設定
        - `logs` フォルダに各ファイルの実行履歴が流れる
- 自動実行設定
    - `/etc/systemd/system`
    - 各種botはsystemctlで起動時に設定をする（`.service`）
    - arXivのRSSは米国東部時間20:30（おそらく平日のみ）なので、~~10時JST（コード内で更新ありかチェックするので毎日実行でよい）にプログラムを実行する設定にする（`.timer`を併用）~~
        - なぜかうまくいかない（RSSが空になっている？）ので、17時JSTごろに実行するように設定するとよさそう？

## Directory structure

```
.
├── dbs
│   ├── arxiv.db
│   └── youtube.db
├── discord_bot
│   ├── arxiv
│   │   ├── arxivbot.py
│   │   ├── db.py
│   │   └── models.py
│   ├── gpt
│   │   └── gptbot.py
│   ├── __init__.py
│   └── youtube
│       ├── db.py
│       ├── models.py
│       └── youtubebot.py
├── .env
├── logs
│   ├── arxivbot.log
│   ├── gptbot.log
│   └── youtubebot.log
├── poetry.lock
├── pyproject.toml
└── README.md
    tests
    └── __init__.py
```