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

    # Define columns in table `cache`.
    HASH = Column(String, primary_key=True)
    LABELS = Column(JSON)

    def __repr__(self):
        return f'HASH: {self.HASH}'


# pylint: disable=maybe-no-member
class Cacher:
    """
    Cacher initializes a sqlite3 database to store and retrieve predictions.
    :param: path: Determines directory where `cache.db` is created.
    """
    def __init__(self, path: str) -> None:

        # The db will be stored in folder 'path'
        self.path = Path(path)

        # Create folder if it does not exist
        os.makedirs(self.path, exist_ok=True)

        # Create engine
        self.db_file = self.path / 'cache.db'
        self.engine = create_engine(f'sqlite:///{self.db_file}', echo=False)

        # Create session
        self.session = sessionmaker(bind=self.engine)()

        # Create table if it does not exist
        if not self.engine.dialect.has_table(self.engine, 'cache'):
            Base.metadata.create_all(self.engine)

    def get(self, HASHES: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get cached predictions from db
        :param: HASHES: List of strings to be queried from db column `HASH`
        """

        if type(HASHES) == str:
            HASHES = [HASHES]

        try:
            sql = self.session.query(Prediction).filter(
                Prediction.HASH.in_(HASHES)).statement

            df = pd.read_sql(sql, self.session.bind)
        except Exception as e:
            # This happens e.g. if the database file is removed
            print(f'Error: {str(e)}. Re-initializing database.')
            self.__init__(self.path)
            df = pd.DataFrame(columns=['HASH', 'LABELS'])

        return df

    def set(self, HASH: str, LABELS: dict) -> None:
        """
        Persist a prediction to db
        :param: HASH: A string to be used as key to store the value (i.e. `LABELS`).
                      If `HASH` already exists, supplied `LABELS` will overwrite
                      its previous entry.
        :param: LABELS: A dictionary containing predictions, must not contain
                        np.floats in order to be JSON serializable.
        """

        self.session.merge(Prediction(HASH=HASH, LABELS=LABELS))
        self.session.commit()

    def close(self):
        self.session.close()
