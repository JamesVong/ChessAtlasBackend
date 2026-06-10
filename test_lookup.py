#!/usr/bin/env python3
"""
Test the position lookup service directly (no server required).

Usage:
    python test_lookup.py
    python test_lookup.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    python test_lookup.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1"
"""

import sys
import os
import json

# Default to the starting position after 1. e4
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1"

fen = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEN

# Point at the local DB if DB_PATH isn't set
os.environ.setdefault("DB_PATH", "chess_atlas.db")

from chess_analyzer.services.position_lookup import lookup_position, _zobrist

placement = fen.strip().split()[0]
zh = _zobrist(placement)
print(f"FEN placement : {placement}")
print(f"Zobrist hash  : {zh}")
print()

if not os.path.isfile(os.environ["DB_PATH"]):
    print(f"ERROR: database not found at '{os.environ['DB_PATH']}'")
    print("Download it first:")
    print("  wget -O chess_atlas.db https://github.com/JamesVong/ChessAtlasBackend/releases/download/v1.1.0/chess_atlas.db")
    sys.exit(1)

results = lookup_position(fen)

if not results:
    print("No matches found.")
else:
    print(f"{len(results)} match(es) found:")
    # print(json.dumps(results, indent=2))
