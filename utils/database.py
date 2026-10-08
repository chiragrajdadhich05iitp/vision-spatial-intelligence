import sqlite3
import datetime

class StoreAnalyticsDB:
    def __init__(self, db_name="store_analytics.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # Footfall & Direction tracking events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS footfall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                direction TEXT,
                timestamp DATETIME
            )
        ''')
        # Hourly aggregated analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_summary (
                hour_slot TEXT PRIMARY KEY,
                total_in INTEGER,
                total_out INTEGER,
                peak_occupancy INTEGER
            )
        ''')
        self.conn.commit()

    def log_event(self, track_id, direction):
        cursor = self.conn.cursor()
        now = datetime.datetime.now()
        cursor.execute('''
            INSERT INTO footfall_events (track_id, direction, timestamp)
            VALUES (?, ?, ?)
        ''', (track_id, direction, now.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()
        print(f"[DB LOG] Track ID {track_id} -> {direction} at {now.strftime('%H:%M:%S')}")

    def get_live_metrics(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM footfall_events WHERE direction='IN'")
        total_in = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM footfall_events WHERE direction='OUT'")
        total_out = cursor.fetchone()[0]
        current_occupancy = max(0, total_in - total_out)
        return total_in, total_out, current_occupancy