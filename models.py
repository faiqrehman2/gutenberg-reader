from database import db
from sqlalchemy import Column, Integer, String, Text
import json

class Book(db.Model):
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)

    def set_metadata(self, metadata_dict):
        """Convert dictionary to JSON string before saving."""
        self.metadata_json = json.dumps(metadata_dict)

    def get_metadata(self):
        """Convert JSON string back to dictionary."""
        return json.loads(self.metadata_json) if self.metadata_json else {}
