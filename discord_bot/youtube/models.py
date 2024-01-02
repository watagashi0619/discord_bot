import enum

from db import Base, engine
from sqlalchemy import Column, DateTime, Enum, Integer, String


class IsOnAir(enum.Enum):
    BEFOREONAIR = enum.auto()
    NOWONAIR = enum.auto()
    AFTERONAIR = enum.auto()
    NOTLIVE = enum.auto()
    DELETED = enum.auto()
    UNARCHIVED = enum.auto()


class Channel(Base):
    __tablename__ = "subscribes"
    channel_id = Column(String, primary_key=True)
    name = Column(String)


class State(Base):
    __tablename__ = "state"
    video_id = Column(String, primary_key=True)
    title = Column(String)
    is_onair = Column(Enum(IsOnAir))
    start_time = Column(DateTime(timezone=True))
    message_id = Column(Integer)


def create_db():
    Base.metadata.create_all(bind=engine)
