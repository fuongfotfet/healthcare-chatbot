import psycopg2
import config


class DatabaseManager:

    def __init__(self):
        self.host = config.DB_HOST
        self.port = config.DB_PORT
        self.dbname = config.DB_NAME
        self.user = config.DB_USER
        self.password = config.DB_PASS

    def get_connection(self):
        """Mở kết nối mới đến Database"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )