"""
Database configuration for VAR-ify
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/varify.db")

# Handle postgres URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Analysis(Base):
    """Video analysis record"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String, index=True)
    video_size = Column(Integer)
    duration = Column(Float)
    trimmed = Column(Boolean, default=False)
    
    # Results
    handball_count = Column(Integer, default=0)
    offside_count = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    
    # Full result JSON
    result_json = Column(JSON)
    result_video = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(String, nullable=True)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
