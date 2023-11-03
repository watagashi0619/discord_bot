import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
dbs_folder_abspath = os.path.join(grandparent_folder_abspath, "dbs")
dotenvpath = os.path.join(grandparent_folder_abspath, ".env")
load_dotenv(dotenvpath)

DBNAME = os.path.basename(current_folder_abspath)

RDB_PATH = f"sqlite:///{dbs_folder_abspath}/{DBNAME}.db"
engine = create_engine(RDB_PATH)
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()
