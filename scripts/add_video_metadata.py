"""
One-off script: populate a video_metadata table in chess_atlas.db.

Fetches title/channel/thumbnail for every distinct video_id in the
positions table via YouTube's oEmbed endpoint (no API key required),
then writes them to a new video_metadata table in the same database.

Usage:
    python scripts/add_video_metadata.py [path/to/chess_atlas.db]

Re-runnable: already-fetched video_ids are skipped.
"""

import json
import sqlite3
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chess_atlas.db"
OEMBED_URL = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json"
MAX_WORKERS = 12
TIMEOUT_SECONDS = 15


def fetch_metadata(video_id: str) -> dict | None:
    """Return oEmbed metadata for a video, or None if unavailable (private/deleted)."""
    req = urllib.request.Request(
        OEMBED_URL.format(vid=video_id),
        headers={"User-Agent": "ChessAtlas/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  FAILED {video_id}: {e}")
        return None
    return {
        "video_id": video_id,
        "title": data.get("title"),
        "author_name": data.get("author_name"),
        "author_url": data.get("author_url"),
        "thumbnail_url": data.get("thumbnail_url"),
    }


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS video_metadata (
            video_id      TEXT PRIMARY KEY,
            title         TEXT,
            author_name   TEXT,
            author_url    TEXT,
            thumbnail_url TEXT
        )
        """
    )
    conn.commit()

    all_ids = {r[0] for r in conn.execute("SELECT DISTINCT video_id FROM positions")}
    done_ids = {r[0] for r in conn.execute("SELECT video_id FROM video_metadata")}
    todo = sorted(all_ids - done_ids)
    print(f"{len(all_ids)} distinct videos, {len(done_ids)} already done, {len(todo)} to fetch")

    fetched = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_metadata, vid): vid for vid in todo}
        for i, future in enumerate(as_completed(futures), 1):
            meta = future.result()
            if meta is None:
                failed.append(futures[future])
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO video_metadata
                    (video_id, title, author_name, author_url, thumbnail_url)
                VALUES (:video_id, :title, :author_name, :author_url, :thumbnail_url)
                """,
                meta,
            )
            fetched += 1
            if i % 100 == 0:
                conn.commit()
                print(f"  {i}/{len(todo)} processed ({fetched} ok, {len(failed)} failed)")
    conn.commit()

    print(f"Done: {fetched} fetched, {len(failed)} failed")
    if failed:
        print("Failed video_ids (private/deleted videos — no metadata row written):")
        for vid in failed:
            print(f"  {vid}")

    count = conn.execute("SELECT COUNT(*) FROM video_metadata").fetchone()[0]
    print(f"video_metadata now has {count} rows")
    conn.close()


if __name__ == "__main__":
    main()
