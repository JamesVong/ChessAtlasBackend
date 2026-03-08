import os
import sqlite3
import struct

import numpy as np

_DB_PATH = os.getenv("DB_PATH", "chess_atlas.db")

# Must match ChessAtlasExplorer/pipeline/zobrist.py exactly.
_FEN_TO_IDX: dict[str, int] = {
    "r": 0, "n": 1, "b": 2, "q": 3, "k": 4, "p": 5,
    "R": 6, "N": 7, "B": 8, "Q": 9, "K": 10, "P": 11,
}
_rng = np.random.default_rng(0xDEADBEEFCAFEBABE)
_TABLE: np.ndarray = _rng.integers(0, 2**63, size=(64, 12), dtype=np.uint64)


def _zobrist(fen_board: str) -> int:
    """Hash the piece-placement part of a FEN, returning a signed int64."""
    h = np.uint64(0)
    sq = 0
    for ch in fen_board:
        if ch == "/":
            continue
        if ch.isdigit():
            sq += int(ch)
        else:
            idx = _FEN_TO_IDX.get(ch)
            if idx is not None:
                h ^= _TABLE[sq, idx]
            sq += 1
    return struct.unpack("q", struct.pack("Q", int(h)))[0]


def lookup_position(fen: str) -> list[dict]:
    """
    Look up a position in the database by Zobrist hash (indexed columns).

    Accepts a full or partial FEN; only the piece-placement part is used.
    Searches both zobrist_white and zobrist_black and returns all matches.
    """
    if not os.path.isfile(_DB_PATH):
        raise FileNotFoundError(f"Database not found at {_DB_PATH}")

    placement = fen.strip().split()[0]
    zh = _zobrist(placement)

    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT video_id, timestamp_seconds, 'white' AS orientation
              FROM positions
             WHERE zobrist_white = ?
            UNION ALL
            SELECT video_id, timestamp_seconds, 'black' AS orientation
              FROM positions
             WHERE zobrist_black = ?
            """,
            (zh, zh),
        )
        rows = cur.fetchall()

    return [
        {
            "video_id": row["video_id"],
            "timestamp_seconds": row["timestamp_seconds"],
            "orientation": row["orientation"],
        }
        for row in rows
    ]
