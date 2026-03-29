import os
import enum
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean,
    Enum as SAEnum, ForeignKey, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class TTSStatus(str, enum.Enum):
    none = "none"
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=True)
    epub_filename = Column(String, nullable=False)
    cover_image_path = Column(String, nullable=True)
    tts_status = Column(SAEnum(TTSStatus), nullable=False, default=TTSStatus.none)
    tts_voice = Column(String, nullable=True, default="af_heart")
    tts_voice_blend = Column(String, nullable=True)
    tts_blend_ratio = Column(Float, nullable=False, default=0.5)
    tts_language = Column(String, nullable=False, default="a")
    tts_speed = Column(Float, nullable=False, default=1.0)
    tts_progress_pct = Column(Float, nullable=False, default=0.0)
    tts_error = Column(String, nullable=True)
    tts_use_cast = Column(Boolean, nullable=False, default=False)
    series = Column(String, nullable=True)
    series_index = Column(Float, nullable=True)
    publisher = Column(String, nullable=True)
    published_date = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    audio_index = relationship("AudioIndex", back_populates="book", uselist=False, cascade="all, delete-orphan")
    character_cast = relationship("CharacterCast", back_populates="book", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "epub_filename": self.epub_filename,
            "cover_image_path": self.cover_image_path,
            "tts_status": self.tts_status.value if self.tts_status else "none",
            "tts_voice": self.tts_voice,
            "tts_progress_pct": self.tts_progress_pct,
            "tts_error": self.tts_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CastStatus(str, enum.Enum):
    none = "none"
    analysing = "analysing"
    done = "done"
    error = "error"


class CharacterCast(Base):
    __tablename__ = "character_casts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False, unique=True)
    status = Column(SAEnum(CastStatus), nullable=False, default=CastStatus.none)
    llm_provider = Column(String, nullable=True)
    progress_pct = Column(Float, nullable=False, default=0.0)
    error_msg = Column(String, nullable=True)
    cast_json = Column(Text, nullable=True)   # JSON blob: {characters:{}, segments:[]}
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    book = relationship("Book", back_populates="character_cast")


class AudioIndex(Base):
    __tablename__ = "audio_indexes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False, unique=True)
    audio_filename = Column(String, nullable=False)
    index_json_filename = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    chapter_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    book = relationship("Book", back_populates="audio_index")


# Module-level engine and Session — initialised by db_init()
engine = None
Session = None


def db_init(config_dir: str = "/config"):
    global engine, Session

    os.makedirs(config_dir, exist_ok=True)
    db_path = os.path.join(config_dir, "inkwright.db")
    db_url = f"sqlite:///{db_path}"

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    _migrate(engine)
    _reset_stuck_jobs()


def _migrate(eng):
    """Apply lightweight column additions for existing databases."""
    migrations = [
        "ALTER TABLE books ADD COLUMN tts_speed FLOAT NOT NULL DEFAULT 1.0",
        "ALTER TABLE books ADD COLUMN tts_voice_blend VARCHAR",
        "ALTER TABLE books ADD COLUMN tts_blend_ratio FLOAT NOT NULL DEFAULT 0.5",
        "ALTER TABLE books ADD COLUMN tts_language VARCHAR NOT NULL DEFAULT 'a'",
        "ALTER TABLE books ADD COLUMN series VARCHAR",
        "ALTER TABLE books ADD COLUMN series_index FLOAT",
        "ALTER TABLE books ADD COLUMN publisher VARCHAR",
        "ALTER TABLE books ADD COLUMN published_date VARCHAR",
        "CREATE TABLE IF NOT EXISTS character_casts (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL UNIQUE REFERENCES books(id), status VARCHAR NOT NULL DEFAULT 'none', llm_provider VARCHAR, progress_pct FLOAT NOT NULL DEFAULT 0.0, error_msg VARCHAR, cast_json TEXT, created_at DATETIME, updated_at DATETIME)",
        "ALTER TABLE books ADD COLUMN tts_use_cast INTEGER NOT NULL DEFAULT 0",
    ]
    with eng.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # column already exists


def _reset_stuck_jobs():
    """On startup, reset any book stuck in 'processing' to 'queued', and any
    CharacterCast stuck in 'analysing' back to 'none'."""
    session = Session()
    try:
        stuck = session.query(Book).filter(Book.tts_status == TTSStatus.processing).all()
        for book in stuck:
            book.tts_status = TTSStatus.queued
            book.tts_progress_pct = 0.0

        stuck_casts = session.query(CharacterCast).filter(
            CharacterCast.status == CastStatus.analysing
        ).all()
        for cast in stuck_casts:
            cast.status = CastStatus.none
            cast.progress_pct = 0.0

        session.commit()
    finally:
        session.close()


def get_session():
    """Return a new SQLAlchemy session. Caller is responsible for closing."""
    if Session is None:
        raise RuntimeError("Database not initialised — call db_init() first.")
    return Session()
