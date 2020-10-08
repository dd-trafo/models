import hashlib
import os
from pathlib import Path
from typing import List, Union

import pandas as pd
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

Base = declarative_base()


class Prediction(Base):
    """
    Define structure of cache table.
    """

    __tablename__ = 'cache'

    HASH = Column(String, primary_key=True)
    LABELS = Column(JSON)

    def __repr__(self):
        return f'HASH: {self.HASH}'


# pylint: disable=maybe-no-member
class Cacher:
    def __init__(self, path: str, model_hash: str) -> None:

        # The db will be stored in folder 'path'
        self.path = Path(path)

        # Create folder if it does not exist
        os.makedirs(self.path, exist_ok=True)

        # 'model_hash' determines the db filename inside 'path'
        self.model_hash = model_hash
        self.db_file = self.path / f'{self.model_hash}.db'

        self.engine = create_engine(f'sqlite:///{self.db_file}', echo=False)

        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # Create table if it does not exist
        if not self.engine.dialect.has_table(self.engine, 'cache'):
            Base.metadata.create_all(self.engine)

    def get(self, hashes: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get cached predictions from sqlite db.
        """

        if type(hashes) == str:
            hashes = [hashes]

        sql = self.session.query(Prediction).filter(
            Prediction.HASH.in_(hashes)).statement

        return pd.read_sql(sql, self.session.bind)

    def cache(self, predictions: Union[dict, List[dict]]) -> None:
        """
        Persist predictions to db.
        """
        if type(predictions) == dict:
            predictions = [predictions]

        for pred in predictions:
            self.session.merge(
                Prediction(HASH=pred['HASH'], LABELS=pred['LABELS']))

        self.session.commit()
