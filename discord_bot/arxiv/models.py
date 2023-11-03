from db import Base, engine
from sqlalchemy import Column, String


class Feed(Base):
    __tablename__ = "subscribes"
    discord_channel_id = Column(String, primary_key=True)
    title = Column(String)
    link = Column(String)
    scheduled_time = Column(String)


def create_db():
    Base.metadata.create_all(bind=engine)
