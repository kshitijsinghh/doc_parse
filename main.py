import os
import re
import json
import logging
import traceback
import httpx
import pdfplumber
from io import BytesIO
import sqlite3
import time
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocParse API")

# ── Session middleware (required by Authlib for OAuth state/nonce) ────────────
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5500",
        "http://127.0.0.1:5500", "http://127.0.0.1:3000",
        "http://localhost:3001",
        os.getenv("FRONTEND_URL", "*"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE  (SQLite — swap for PostgreSQL via asyncpg in production)
# ═══════════════════════════════════════════════════════════════════════════════

DB_PATH = Path(os.getenv("DB_PATH", "docmind_users.db"))

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                google_id     TEXT    UNIQUE NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                name          TEXT,
                picture       TEXT,
                first_login   INTEGER NOT NULL,
                last_login    INTEGER NOT NULL,
                login_count   INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()
    logger.info("DB ready at %s", DB_PATH)

init_db()

def upsert_user(google_id: str, email: str, name: str, picture: str) -> dict:
    now = int(time.time())
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM users WHERE google_id = ?", (google_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE users SET last_login=?, login_count=login_count+1, name=?, picture=? WHERE google_id=?",
                (now, name, picture, google_id),
            )
        else:
            conn.execute(
                "INSERT INTO users (google_id,email,name,picture,first_login,last_login,login_count) VALUES (?,?,?,?,?,?,1)",
                (google_id, email, name, picture, now, now),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE google_id=?", (google_id,)).fetchone()
    logger.info("User upserted: %s login_count=%d", email, row["login_count"])
    return dict(row)

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH 2.0
# ═══════════════════════════════════════════════════════════════════════════════

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:3001")
FRONTEND_URL     = os.getenv("FRONTEND_URL",     "http://localhost:5500")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@app.get("/auth/google")
async def auth_google(request: Request):
    redirect_uri = f"{BACKEND_BASE_URL}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error("OAuth callback error: %s", e)
        return RedirectResponse(f"{FRONTEND_URL}?error=oauth_failed")

    userinfo = token.get("userinfo") or {}
    if not userinfo:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {token['access_token']}"},
            )
            userinfo = resp.json()

    google_id = userinfo.get("sub")
    email     = userinfo.get("email", "")
    name      = userinfo.get("name", "")
    picture   = userinfo.get("picture", "")

    if not google_id or not email:
        return RedirectResponse(f"{FRONTEND_URL}?error=missing_user_info")

    upsert_user(google_id, email, name, picture)

    params = urllib.parse.urlencode({
        "sso_success": "1",
        "name": name,
        "email": email,
        "picture": picture,
    })
    return RedirectResponse(f"{FRONTEND_URL}?{params}")

@app.get("/admin/users")
async def list_users():
    """List all SSO users — protect or remove before going to production!"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,email,name,first_login,last_login,login_count FROM users ORDER BY last_login DESC"
        ).fetchall()
    return [dict(r) for r in rows]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Groq API key pool (round-robin) ──────────────────────────────────────────
# Reads up to 4 keys from env: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3,
# GROQ_API_KEY_4. Also falls back to legacy GROQ_API_KEY if none of the numbered
# keys are set. Add keys to your .env file like:
#   GROQ_API_KEY_1=gsk_xxx
#   GROQ_API_KEY_2=gsk_yyy
#   GROQ_API_KEY_3=gsk_zzz
#   GROQ_API_KEY_4=gsk_www
def _load_groq_keys() -> list[str]:
    keys = []
    for i in range(1, 5):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    if not keys:
        # legacy fallback — single key
        legacy = os.getenv("GROQ_API_KEY", "").strip()
        if legacy:
            keys.append(legacy)
    return keys

GROQ_API_KEYS: list[str] = _load_groq_keys()
_groq_key_index = 0          # global round-robin pointer
_groq_key_lock  = __import__("asyncio").Lock()   # async-safe increment

async def _next_groq_key() -> str:
    """Return the next API key in round-robin order (async-safe)."""
    global _groq_key_index
    if not GROQ_API_KEYS:
        raise HTTPException(
            status_code=500,
            detail="No Groq API keys configured. Set GROQ_API_KEY_1 … GROQ_API_KEY_4 in .env"
        )
    async with _groq_key_lock:
        key = GROQ_API_KEYS[_groq_key_index % len(GROQ_API_KEYS)]
        _groq_key_index += 1
    return key

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id → {
#   "regex":  { "Field Name": "regex_pattern", ... },   <- text field extraction
#   "tables": [ { "headers": [...], "col_map": { "safe_header": col_idx } }, ... ]
#                                                        <- table line item extraction
# }
# Populated by /api/detect-fields, consumed by /api/extract-fields.
# Never written to disk. Dropped automatically when the process ends
# or when the session explicitly clears it.
_EXTRACTOR_STORE: dict[str, dict] = {}


def _is_garbled_text(text: str) -> bool:
    """
    Detect PDFs with font-encoding issues where pdfplumber produces scrambled text.
    Signs: colons appearing INSIDE words (e.g. "Numb:e7r", "Amoun:t"),
    digits embedded mid-word (e.g. "N1o6i"), or very high garble ratio.
    These PDFs must use LLM extraction — regex on garbled text always fails.
    """
    if not text or len(text) < 50:
        return False
    words = text.split()
    if not words:
        return False
    # Colon splits a word: "Numb:e7r", "Numbe:r1", "Amoun:t"
    colon_in_word = sum(1 for w in words if re.search(r'[a-zA-Z]:[a-zA-Z0-9]', w))
    # Digit embedded mid-word: "N1o6i", "ReveCrhsaer"  
    digit_mid_word = sum(1 for w in words if re.search(r'[a-zA-Z][0-9][a-zA-Z]', w))
    garble_ratio = (colon_in_word + digit_mid_word) / len(words)
    return garble_ratio > 0.04  # >4% of words show garbling

def _extract_table_by_xpos(pdf_bytes: bytes, page_index: int = 0) -> dict:
    """
    Generic table column extractor using PDF word x-coordinates.
    Finds any table with column headers + a TOTAL/data row,
    maps column labels to amount values by x-position alignment.
    Works for any invoice layout — no hardcoded column names.
    Returns dict of {normalized_col_label: numeric_string}.
    """
    try:
        import pdfplumber
        from io import BytesIO

        COL_KEYWORDS = {
            'qty', 'quantity', 'gross', 'discount', 'other', 'taxable',
            'cgst', 'sgst', 'ugst', 'igst', 'cess', 'total', 'amount',
            'charges', 'value', 'rate', 'price', 'unit', 'tax', 'item'
        }

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            if page_index >= len(pdf.pages):
                return {}
            page = pdf.pages[page_index]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                return {}

            # Group words by y-position (row)
            rows_by_y: dict = {}
            for w in words:
                y = round(w['top'])
                rows_by_y.setdefault(y, []).append(w)

            # Identify header rows and TOTAL row
            header_ys: set = set()
            total_y = None
            for y, row_words in sorted(rows_by_y.items()):
                texts = [w['text'].lower() for w in row_words]
                has_keyword = any(any(k in t for k in COL_KEYWORDS) for t in texts)
                has_big_number = any(re.search(r'\d{2,}\.', t) for t in texts)
                is_total_row = any(t in ('total', 'totals') for t in texts) and has_big_number
                if has_keyword and not has_big_number:
                    header_ys.add(y)
                if is_total_row and total_y is None:
                    total_y = y

            if not header_ys or total_y is None:
                return {}

            # Build column spans from header words (merge words within 8px)
            header_words = sorted(
                [w for y in header_ys for w in rows_by_y[y]],
                key=lambda w: w['x0']
            )
            columns: list = []  # [(x0, x1, label)]
            for w in header_words:
                if columns and w['x0'] - columns[-1][1] <= 8:
                    x0, x1, label = columns[-1]
                    columns[-1] = (x0, max(x1, w['x1']), label + ' ' + w['text'])
                else:
                    columns.append((w['x0'], w['x1'], w['text']))

            # Get numeric amounts from TOTAL row
            total_words = sorted(rows_by_y.get(total_y, []), key=lambda w: w['x0'])
            amounts = [
                (w['x0'], w['x1'], w['text'])
                for w in total_words
                if re.match(r'[\d,\.]+$', w['text']) and '.' in w['text']
            ]

            if not amounts:
                return {}

            # Assign each amount to the column whose x-range it falls in
            result = {}
            for ax0, ax1, aval in amounts:
                a_center = (ax0 + ax1) / 2
                best_col = None
                best_dist = float('inf')
                for cx0, cx1, clabel in columns:
                    c_center = (cx0 + cx1) / 2
                    in_range = (cx0 - 15) <= a_center <= (cx1 + 15)
                    dist = abs(a_center - c_center)
                    if in_range and dist < best_dist:
                        best_dist = dist
                        best_col = clabel
                    elif not in_range and dist < best_dist:
                        # Fallback to nearest if nothing in range
                        best_dist = dist
                        best_col = clabel
                if best_col:
                    # Normalise: lowercase, strip noise words
                    norm = re.sub(
                        r'\b(qty|quantity|total|totals|no|sno|sr)\b', '',
                        best_col, flags=re.IGNORECASE
                    ).strip().lower()
                    norm = re.sub(r'\s+', ' ', norm).strip()
                    if norm:
                        result[norm] = aval

            return result
    except Exception as e:
        logger.warning("_extract_table_by_xpos error: %s", e)
        return {}


def _match_field_to_col(field_name: str, col_map: dict) -> str:
    """
    Match a user field name to a detected column label using fuzzy keyword overlap.
    e.g. "Gross Amount_occ1" → "gross amount" → "2999.00"
    Returns the matched value or "" if no match.
    """
    # Strip _occ suffix
    base = re.sub(r'_occ\d+$', '', field_name).strip().lower()
    # Remove filler words
    base = re.sub(r'\b(total|amount|rs|inr|value)\b', ' ', base).strip()
    base_tokens = set(base.split())

    best_match = None
    best_score = 0
    for col_label, col_val in col_map.items():
        col_tokens = set(col_label.split())
        overlap = len(base_tokens & col_tokens)
        if overlap > best_score:
            best_score = overlap
            best_match = col_val

    return best_match if best_score > 0 else ""





# ── Known-good patterns for fields the LLM consistently gets wrong ────────────
# These override any LLM-generated pattern when the field name matches (case-insensitive).
# Patterns must use double-escaped backslashes (they are stored as raw strings then compiled).
# _KNOWN_PATTERNS removed — all patterns generated dynamically by LLM
_KNOWN_PATTERNS: dict[str, str] = {}  # kept for backwards compat, always empty


def _apply_known_pattern(field: str, text: str) -> str:
    """No-op — patterns are now generated entirely by LLM schema."""
    return ""


    try:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            # Pick first non-None group (handles patterns with alternatives)
            groups = [g for g in m.groups() if g is not None] if m.lastindex else []
            val = groups[0].strip() if groups else m.group(0).strip()
            if "\n" in val:
                val = val.split("\n")[0].strip()
            return val
    except re.error:
        pass
    return ""



def _get_field_from_table(field: str, table_rows: list[list]) -> str:
    """
    Last-resort: search table_rows for a column whose header matches `field`,
    and return the first non-empty data cell value in that column.
    Used when regex schema returns a header row instead of a real value.
    """
    if not table_rows:
        return ""
    field_lower = field.lower().strip()
    for i, row in enumerate(table_rows):
        row_cells = [c.strip() if c else "" for c in row]
        # Check if this looks like a header row
        row_lower = [c.lower() for c in row_cells]
        # Find matching column
        col_idx = None
        for j, cell in enumerate(row_lower):
            if field_lower in cell or cell in field_lower:
                col_idx = j
                break
        if col_idx is not None:
            # Look for first data row below this header
            for data_row in table_rows[i+1:]:
                data_cells = [c.strip() if c else "" for c in data_row]
                if col_idx < len(data_cells) and data_cells[col_idx]:
                    val = data_cells[col_idx]
                    # Only return if it looks like a number/amount, not another header
                    if re.search(r'[\d\.]', val):
                        return val
    return ""


def _run_schema(schema: dict, flat_text: str, fields: list[str],
                table_rows: list[list] = None,
                table_schema: list[dict] = None,
                left_text: str = "") -> dict:
    """
    Apply a regex schema to extract fields from flattened PDF text,
    AND extract table line items using the stored table_schema.
    Uses left_text (single-column) first to avoid two-column contamination.
    """
    result     = {}
    clean_text = "\n".join(line.strip() for line in flat_text.split("\n"))
    # Single-column text avoids "Noida  Rangareddy," two-column merges
    clean_left = "\n".join(line.strip() for line in left_text.split("\n")) if left_text else ""

    # ── Pre-build table line items from table_rows if table_schema available ──
    table_cells: dict[str, str] = {}
    if table_rows and table_schema:
        table_cells = _apply_table_schema(table_rows, table_schema)

    # ── Build a direct col-position lookup from raw table_rows ───────────────
    # Maps normalised header name → (col_index, data_value) from the TOTAL row
    # This lets us extract CGST, IGST, Total etc. directly from table cells
    # rather than relying on fragile regex over flat text.
    direct_table_lookup: dict[str, str] = {}
    if table_rows:
        header_row  = None
        data_row    = None
        total_row   = None
        for row in table_rows:
            cells = [c.strip() if c else "" for c in row]
            if _is_header_row(cells) and header_row is None:
                header_row = cells
            elif header_row is not None:
                joined = " ".join(c for c in cells if c).lower()
                if re.match(r'total\b', joined):
                    total_row = cells
                elif any(re.search(r'[\d,\.]+', c) for c in cells):
                    if data_row is None:
                        data_row = cells
        use_row = total_row or data_row
        if header_row and use_row:
            for col_idx, hdr in enumerate(header_row):
                if not hdr:
                    continue
                norm = re.sub(r'\s+', ' ', hdr.lower().strip())
                # Also add sub-parts of multi-word headers
                direct_table_lookup[norm] = use_row[col_idx].strip() if col_idx < len(use_row) else ""
                for word in norm.split():
                    if len(word) >= 3 and word not in direct_table_lookup:
                        direct_table_lookup[word] = use_row[col_idx].strip() if col_idx < len(use_row) else ""

    for field in fields:
        # ── Table field: SomeHeader_N ─────────────────────────────────────────
        if re.match(r'^.+_\d+$', field) and field in table_cells:
            result[field] = table_cells[field]
            continue

        # ── Direct table column match ─────────────────────────────────────────
        # If the field name matches a table column header, extract directly
        # from the data row — more specific/longer matches take priority
        field_norm = re.sub(r'\s+', ' ', field.lower().strip())
        # Collect all matching keys and pick the one with longest overlap
        best_val = ""
        best_len = 0
        for tbl_key, tbl_val in direct_table_lookup.items():
            if not tbl_val or not re.search(r'[\d,\.]+', tbl_val):
                continue
            if tbl_key == field_norm:
                # Exact match always wins
                best_val = tbl_val
                best_len = 9999
                break
            if field_norm in tbl_key or tbl_key in field_norm:
                overlap = len(set(field_norm.split()) & set(tbl_key.split()))
                if overlap > best_len:
                    best_val = tbl_val
                    best_len = overlap
        if best_val:
            result[field] = best_val
            continue

        # ── Known-good pattern override (runs before LLM-generated schema) ──
        known_val = _apply_known_pattern(field, clean_text)
        if known_val:
            result[field] = known_val
            continue

        # ── Text field: regex pattern ─────────────────────────────────────────
        pattern = schema.get(field)
        if not pattern:
            result[field] = "N/A"
            continue
        try:
            # Try left_text first (avoids two-column contamination)
            m = None
            if clean_left:
                m = re.search(pattern, clean_left, re.IGNORECASE | re.MULTILINE)
            # Fall back to full merged text if not found in left
            if not m:
                m = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE)

            if m:
                if m.lastindex and m.lastindex >= 2:
                    groups      = [g for g in m.groups() if g is not None]
                    first_group = groups[0].strip() if groups else ""
                    field_lower = field.lower().strip()
                    if first_group.lower() == field_lower or len(groups) == 1:
                        val = first_group
                    else:
                        val = "".join(groups[1:]).strip()
                else:
                    val = m.group(1).strip()
                # Reject if value contains newlines (multi-line bleed) — take only first line
                if "\n" in val:
                    val = val.split("\n")[0].strip()
                # Reject if value looks like a table header row (3+ column keywords)
                _TBL_KWS = {"cgst","sgst","igst","total","amount","rate","utgst","tax","value","invoice","description","period"}
                val_kw_hits = sum(1 for kw in _TBL_KWS if kw in val.lower().split())
                if val_kw_hits >= 3:
                    val = "N/A"
                result[field] = val
            else:
                result[field] = "N/A"
        except re.error as e:
            logger.warning("Bad regex for field %r: %s — pattern: %r", field, e, pattern)
            result[field] = "N/A"
            continue

        if result.get(field) == "N/A" and pattern:
            logger.info("Pattern no-match: field=%r pattern=%r", field, pattern)
            result[field] = "N/A"

    return result



# ═══════════════════════════════════════════════════════════════════════════════
# TABLE SCHEMA  —  built once from the preview PDF, reused on all subsequent PDFs
# ═══════════════════════════════════════════════════════════════════════════════

TAX_SUMMARY_KEYWORDS = frozenset([
    "cgst", "sgst", "igst", "taxable value", "total tax",
    "total invoice", "value of service", "taxable amount",
])

def _safe_col_name(header: str) -> str:
    """
    Convert a table header to a safe, readable key name.
    'Unit Price' -> 'Unit_Price', 'HSN/SAC' -> 'HSN_SAC'
    Truncated to 30 chars max to avoid monster keys from data cells
    mistaken for headers.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', header.strip()).strip('_')
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Truncate to 30 chars, cut at last underscore to avoid partial words
    if len(cleaned) > 30:
        cleaned = cleaned[:30].rstrip('_')
    return cleaned


# Known short header keywords — a real header row almost always contains at least one
_HEADER_KEYWORDS = frozenset([
    "sno", "s.no", "sr", "no", "description", "desc", "particulars",
    "qty", "quantity", "rate", "amount", "price", "total", "value",
    "supply", "frequency", "period", "hsn", "sac", "cgst", "sgst",
    "igst", "tax", "discount", "unit", "item", "product", "service",
    "date", "invoice", "narration", "details",
])

def _is_header_row(row: list[str]) -> bool:
    """
    True if this row looks like column headers.
    Checks two things:
    1. At least one cell matches a known header keyword (case-insensitive)
    2. No single cell is longer than 60 chars (data cells are often long paragraphs)
    """
    non_empty = [c.strip() for c in row if c.strip()]
    if not non_empty:
        return False

    # If any cell is very long (>60 chars) it's almost certainly a data cell
    if any(len(c) > 60 for c in non_empty):
        return False

    # Must contain at least one known header keyword
    flat = " ".join(c.lower() for c in non_empty)
    has_keyword = any(kw in flat.split() or
                      any(kw in cell.lower() for cell in non_empty)
                      for kw in _HEADER_KEYWORDS)
    if not has_keyword:
        return False

    # Must be mostly text (not numbers)
    num_count  = sum(1 for c in non_empty if re.match(r'^[\d,\.]+$', c))
    text_count = len(non_empty) - num_count
    return text_count >= len(non_empty) * 0.6


def _is_tax_summary_header(row: list[str]) -> bool:
    """True if the header belongs to a GST tax summary table."""
    flat = " ".join(c.lower() for c in row if c)
    return sum(1 for k in TAX_SUMMARY_KEYWORDS if k in flat) >= 2


def build_table_schema(table_rows: list[list]) -> list[dict]:
    """
    Analyse the table rows from the PREVIEW PDF and build a table_schema list.
    Each entry describes one logical table:
        {
            "headers":        ["Description", "Qty", "Rate", "Amount"],
            "col_map":        {"Description": 0, "Qty": 1, "Rate": 2, "Amount": 3},
            "safe_col_map":   {"Description": 0, "Qty": 1, "Rate": 2, "Amount": 3},
            "is_tax_summary": False,
            "row_count":      2       <- how many data rows found in preview PDF
        }
    Stored in _EXTRACTOR_STORE[session_id]["tables"] for reuse.
    """
    if not table_rows:
        return []

    schema    = []
    cur_hdr   = None
    cur_rows  = []

    def _flush():
        if cur_hdr and cur_rows:
            col_map      = {h.strip(): i for i, h in enumerate(cur_hdr) if h.strip()}
            safe_col_map = {_safe_col_name(h): i for i, h in enumerate(cur_hdr) if h.strip()}
            schema.append({
                "headers":        [h.strip() for h in cur_hdr],
                "col_map":        col_map,
                "safe_col_map":   safe_col_map,
                "is_tax_summary": _is_tax_summary_header(cur_hdr),
                "row_count":      len(cur_rows),
            })

    for row in table_rows:
        row = [c.strip() if c else "" for c in row]
        if _is_header_row(row):
            _flush()
            cur_hdr  = row
            cur_rows = []
        elif cur_hdr:
            non_empty = [c for c in row if c]
            if len(non_empty) >= 2:
                cur_rows.append(row)

    _flush()
    logger.info("Table schema built: %d tables detected", len(schema))
    return schema


def _apply_table_schema(table_rows: list[list], table_schema: list[dict]) -> dict[str, str]:
    """
    Apply a stored table_schema to fresh table_rows from a new PDF.
    Returns { "Description_1": "Item A", "Qty_1": "2", ... } for all line item tables.
    Tax summary tables are skipped (handled by _extract_from_table_rows).
    """
    if not table_rows or not table_schema:
        return {}

    result    = {}
    cur_hdr   = None
    cur_rows  = []
    schema_idx = 0   # which schema entry we are currently matching

    def _flush_and_extract():
        nonlocal schema_idx
        if not cur_hdr or not cur_rows or schema_idx >= len(table_schema):
            return
        tbl = table_schema[schema_idx]
        schema_idx += 1

        if tbl["is_tax_summary"]:
            return  # handled elsewhere

        safe_col_map = tbl["safe_col_map"]
        for row_idx, row in enumerate(cur_rows, start=1):
            row_flat = " ".join(c.lower() for c in row if c)
            # Skip total/subtotal footer rows
            if re.match(r'^\s*(total|subtotal|grand total|sub total)\s*$', row_flat.strip()):
                continue
            for safe_name, col_idx in safe_col_map.items():
                if not safe_name:
                    continue
                cell = row[col_idx].strip() if col_idx < len(row) else ""
                result[f"{safe_name}_{row_idx}"] = cell if cell else "N/A"

    for row in table_rows:
        row = [c.strip() if c else "" for c in row]
        if _is_header_row(row):
            _flush_and_extract()
            cur_hdr  = row
            cur_rows = []
        elif cur_hdr:
            non_empty = [c for c in row if c]
            if len(non_empty) >= 2:
                cur_rows.append(row)

    _flush_and_extract()
    return result


def extract_line_items_from_tables(table_rows: list[list]) -> dict[str, str]:
    """
    One-shot extraction of ALL line items without a pre-built schema.
    Used for PATH B (LLM per PDF / different templates) and as a fallback.
    Returns { "Description_1": "...", "Qty_2": "...", ... }
    """
    schema = build_table_schema(table_rows)
    return _apply_table_schema(table_rows, schema)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION  (always Python, always accurate)
# ═══════════════════════════════════════════════════════════════════════════════

def get_pdf_page_count(file_bytes: bytes) -> int:
    """Return total number of pages in a PDF."""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return len(pdf.pages)


def extract_text_from_pdf(file_bytes: bytes, pages: list | None = None) -> "_TextBundle":
    bundle, _ = extract_text_and_tables_from_pdf(file_bytes, pages)
    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON-ONLY EXTRACTION  (used when same_template = true)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Semantic field groups ──────────────────────────────────────────────────────
# Each group is a frozenset of keywords. If a field name contains ANY keyword
# from a group, it is treated as belonging to that semantic category.
# This makes extraction robust regardless of whether the LLM says
# "Vendor", "Seller", "Supplier" or "Recipient", "Customer", "Buyer".

_SELLER_KEYWORDS   = frozenset(["vendor", "seller", "supplier", "our"])
_BUYER_KEYWORDS    = frozenset(["recipient", "customer", "buyer", "client"])

def _is_seller_field(fl: str) -> bool:
    return any(k in fl for k in _SELLER_KEYWORDS)

def _is_buyer_field(fl: str) -> bool:
    return any(k in fl for k in _BUYER_KEYWORDS)

def _field_contains(fl: str, *keywords) -> bool:
    return any(k in fl for k in keywords)


# Maps canonical semantic categories → all possible PDF label variants.
# Keys are lowercase category names; values are label strings to search in PDF text.
FIELD_ALIASES: dict[str, list[str]] = {
    # ── Document header ────────────────────────────────────────────────────────
    "document type":      ["document type", "type of document"],
    "invoice number":     ["invoice no", "invoice no.", "invoice number", "inv no", "inv#", "invoice#", "#"],
    "invoice date":       ["invoice date", "date", "bill date", "tax invoice date"],
    "order number":       ["order number", "order no", "order no."],
    "order_date":         ["order date"],
    "reference number":   ["reference no", "reference no.", "ref no", "ref no.", "ref#", "reference number"],
    "terms":              ["terms", "payment terms", "terms of payment"],
    "nature_transaction": ["nature of transaction"],
    "nature_supply":      ["nature of supply"],
    # ── Seller / Vendor / Supplier ─────────────────────────────────────────────
    "seller_name":        ["bill from", "seller name", "vendor name", "supplier name", "sold by"],
    "seller_address":     ["vendor address", "supplier address", "our address", "registered address"],
    "seller_pan":         ["pan", "pan no", "pan number"],
    "seller_gstin":       ["gstin", "gstin number", "gst no", "gst number"],
    "seller_state_code":  [],                 # positional: 2nd State Code occurrence
    "seller_place_supply":["place of supply", "place of supply or services"],
    "seller_place_deliv": ["place of delivery",
                           "place of delivery (if different from place of supply or service)"],
    "seller_rev_charge":  ["reverse charge", "whether tax payable under reverse charge",
                           "tax payable under reverse charge"],
    "seller_msme":        ["msme", "msme registered", "is msme"],
    # ── Buyer / Recipient / Customer ───────────────────────────────────────────
    "buyer_name":         ["bill to", "bill to / ship to", "buyer name", "customer name", "recipient name"],
    "buyer_address":      ["recipient address", "billing address", "buyer address",
                           "customer address"],
    "buyer_gstin":        ["gstin no"],       # "GSTIN No:" line = buyer
    "buyer_state_code":   [],                 # positional: 1st State Code occurrence
    # ── Service ────────────────────────────────────────────────────────────────
    "description_service":["description of service", "description of services",
                           "service description", "nature of service"],
    "hsn_sac":            ["hsn/sac", "hsn", "sac", "hsn code", "sac code"],
    "period":             ["period", "service period", "billing period"],
    # ── Tax table values ───────────────────────────────────────────────────────
    "taxable_amount":     ["total value of services", "taxable value", "basic amount",
                           "value of services", "net amount", "taxable amount"],
    "cgst_rate":          ["cgst rate", "cgst(rate)", "cgst %"],
    "sgst_rate":          ["sgst rate", "sgst(rate)", "sgst %", "sgst/ugst rate"],
    "igst_rate":          ["igst rate", "igst(rate)", "igst %"],
    "cgst_amount":        ["cgst amount"],
    "sgst_amount":        ["sgst amount"],
    "igst_amount":        ["igst amount"],
    "total_tax":          ["total tax amount", "total tax", "tax amount"],
    "total_invoice":      ["total invoice amount", "total invoice value",
                           "grand total", "invoice total", "total amount"],
    "amount_words":       ["amount in words", "rupees", "in words", "total in words"],
    # ── TDS ────────────────────────────────────────────────────────────────────
    "tds_rate":           ["tds rate", "tds %"],
    "tds_note":           ["tds to be deducted", "tds note", "tds remark"],
    # ── Other ──────────────────────────────────────────────────────────────────
    "authorised_signatory": ["authorised signatory", "authorized signatory",
                             "for ", "signatory"],
    "e_invoice_note":     ["e-invoice", "e invoice", "exempted from issuance",
                           "notification no", "not required to prepare"],
}


def _resolve_category(fl: str) -> str:
    """Map any LLM field name → canonical category key."""
    # ── Document header ──────────────────────────────────────────────────────
    if _field_contains(fl, "document type", "type of doc"):         return "document type"
    if fl == "#" or _field_contains(fl, "invoice no", "invoice number", "inv no", "bill no"):
                                                                     return "invoice number"
    if _field_contains(fl, "invoice date", "bill date") or fl == "date":
                                                                     return "invoice date"
    if _field_contains(fl, "due date"):                              return "due_date"
    if _field_contains(fl, "order date"):                            return "order_date"
    if _field_contains(fl, "order number", "order no"):             return "order_number"
    if _field_contains(fl, "reference no", "ref no", "ref number"): return "reference number"
    if fl in ("terms", "payment terms", "terms of payment"):         return "terms"
    if _field_contains(fl, "nature of transaction"):                 return "nature_transaction"
    if _field_contains(fl, "nature of supply"):                      return "nature_supply"
    if _field_contains(fl, "packet", "packetid"):                   return "packet_id"
    if _field_contains(fl, "balance due", "amount due", "amount payable"):
                                                                     return "total_invoice"
    if _field_contains(fl, "sub total", "subtotal"):                 return "taxable_amount"
    if fl in ("total", "grand total", "invoice total") or \
       _field_contains(fl, "total amount"):                          return "total_invoice"
    if fl in ("qty", "quantity"):                                    return "qty"
    if _field_contains(fl, "gross amount", "gross"):                 return "gross_amount"
    if fl == "discount" or _field_contains(fl, "discount amount"):   return "discount"

    # ── Seller fields — "Bill From" is seller ────────────────────────────────
    if _field_contains(fl, "bill from", "ship from"):               return "seller_name"
    if _is_seller_field(fl):
        if _field_contains(fl, "company", "name", "firm"):          return "seller_name"
        if _field_contains(fl, "address", "addr"):                  return "seller_address"
        if _field_contains(fl, "pan"):                              return "seller_pan"
        if _field_contains(fl, "gstin", "gst no", "gst number"):   return "seller_gstin"
        if _field_contains(fl, "state code"):                       return "seller_state_code"
        if _field_contains(fl, "place of supply", "supply"):        return "seller_place_supply"
        if _field_contains(fl, "place of deliv", "delivery"):       return "seller_place_deliv"
        if _field_contains(fl, "reverse charge", "rev charge"):     return "seller_rev_charge"
        if _field_contains(fl, "msme"):                             return "seller_msme"

    # ── Buyer fields — "Bill to / Ship to" is buyer ───────────────────────────
    if _field_contains(fl, "bill to", "ship to"):                   return "buyer_name"
    if _is_buyer_field(fl):
        if _field_contains(fl, "company", "name", "firm"):          return "buyer_name"
        if _field_contains(fl, "address", "addr"):                  return "buyer_address"
        if _field_contains(fl, "gstin", "gst no", "gst number"):   return "buyer_gstin"
        if _field_contains(fl, "state code"):                       return "buyer_state_code"

    # ── Fields without vendor/buyer qualifier — use content keywords ──────────
    if _field_contains(fl, "description", "nature of service"):     return "description_service"
    if _field_contains(fl, "hsn", "sac"):                           return "hsn_sac"
    if _field_contains(fl, "service period", "billing period") or fl == "period":
                                                                     return "period"
    if _field_contains(fl, "taxable amount", "taxable value",
                       "value of service", "basic amount",
                       "total amount before tax", "net amount"):    return "taxable_amount"
    if _field_contains(fl, "cgst rate", "cgst(rate)", "cgst %") or \
       re.search(r'\bcgst.*rate', fl):                               return "cgst_rate"
    if _field_contains(fl, "sgst rate", "sgst(rate)", "sgst %", "ugst rate") or \
       re.search(r'\b(sgst|ugst).*rate', fl):                        return "sgst_rate"
    if _field_contains(fl, "igst rate", "igst(rate)", "igst %") or \
       re.search(r'\bigst.*rate', fl):                               return "igst_rate"
    if _field_contains(fl, "tax rate", "gst rate", "% igst", "% gst"):  return "igst_rate"
    if _field_contains(fl, "cgst amount") or \
       (fl == "cgst") or re.search(r'^cgst$', fl):                 return "cgst_amount"
    if _field_contains(fl, "sgst amount", "ugst amount") or \
       re.search(r'^(sgst|ugst)$', fl):                            return "sgst_amount"
    if _field_contains(fl, "igst amount") or \
       (fl == "igst") or re.search(r'^igst$', fl):                 return "igst_amount"
    if _field_contains(fl, "total tax"):                            return "total_tax"
    if _field_contains(fl, "total invoice", "grand total",
                       "invoice total", "invoice value"):           return "total_invoice"
    if _field_contains(fl, "amount in words", "in words", "total in words") or \
       fl in ("rupees", "total in words"):                           return "amount_words"
    if _field_contains(fl, "tds rate", "tds %"):                   return "tds_rate"
    if _field_contains(fl, "tds note", "tds remark") or \
       _field_contains(fl, "tds") and _field_contains(fl, "note", "remark", "deduct"):
                                                                     return "tds_note"
    if _field_contains(fl, "tds"):                                  return "tds_rate"
    if _field_contains(fl, "msme"):                                 return "seller_msme"
    if _field_contains(fl, "reverse charge"):                       return "seller_rev_charge"
    if _field_contains(fl, "place of supply"):                      return "seller_place_supply"
    if _field_contains(fl, "place of deliv", "place of delivery"):  return "seller_place_deliv"
    if _field_contains(fl, "signatory", "authoris", "authoriz"):   return "authorised_signatory"
    if _field_contains(fl, "e-invoice", "e invoice", "exemption",
                       "exempted", "notification"):                  return "e_invoice_note"
    if _field_contains(fl, "pan"):                                  return "seller_pan"
    if _field_contains(fl, "gstin", "gst no") and \
       _field_contains(fl, "no", "number"):                         return "buyer_gstin"
    if _field_contains(fl, "gstin", "gst"):                        return "seller_gstin"
    if _field_contains(fl, "state code"):                           return "seller_state_code"
    if _field_contains(fl, "terms"):                                return "terms"
    if _field_contains(fl, "reference", "ref no"):                  return "reference number"
    if _field_contains(fl, "amount in words", "rupees") or \
       ("amount" in fl and "words" in fl):                          return "amount_words"
    if _field_contains(fl, "total invoice", "invoice value"):       return "total_invoice"
    if _field_contains(fl, "total tax"):                            return "total_tax"
    return ""   # unknown — fall through to direct label search


def _split_two_columns(line: str) -> tuple[str, str]:
    """
    Split a layout=True line into left and right columns.
    pdfplumber layout mode separates columns with 5+ consecutive spaces.
    Strips leading indentation before searching for column gap.
    Returns (left_col, right_col). right_col is "" if single column.
    """
    stripped = line.strip()
    # Look for a gap of 5+ spaces within the stripped line
    m = re.search(r'^(.+?)\s{5,}(\S.*)', stripped)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return stripped, ""


def extract_text_and_tables_from_pdf(file_bytes: bytes, pages: list | None = None) -> tuple[str, list[list]]:
    """
    Extract text and tables from PDF.
    Per-page text stored in bundle.pages[] for multi-invoice PDFs ("Field 1"/"Field 2").
    """
    left_lines_p1  = []
    right_lines_p1 = []
    full_lines_p1  = []
    left_lines_all  = []
    right_lines_all = []
    full_lines_all  = []
    all_table_rows = []
    per_page_text  = []   # full text per page (0-indexed)

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        total = len(pdf.pages)
        if pages:
            page_indices = sorted(set(p for p in pages if 0 <= p < total))
        else:
            page_indices = list(range(total))
        if not page_indices:
            page_indices = list(range(total))

        header_page = page_indices[0]

        for page_num in page_indices:
            page      = pdf.pages[page_num]
            page_text = page.extract_text(layout=True)
            page_full_lines = []
            if page_text:
                for line in page_text.split("\n"):
                    left, right = _split_two_columns(line)
                    merged = (left + "  " + right).strip()
                    if page_num == header_page:
                        if left: left_lines_p1.append(left)
                        if right: right_lines_p1.append(right)
                        full_lines_p1.append(merged)
                    if left: left_lines_all.append(left)
                    if right: right_lines_all.append(right)
                    full_lines_all.append(merged)
                    page_full_lines.append(merged)
            per_page_text.append("\n".join(page_full_lines).strip())

            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean = [cell.strip() if cell else "" for cell in row]
                    if any(clean):
                        all_table_rows.append(clean)

    left_text_p1  = "\n".join(left_lines_p1).strip()
    right_text_p1 = "\n".join(right_lines_p1).strip()
    full_text_p1  = "\n".join(full_lines_p1).strip()
    full_text_all = "\n".join(full_lines_all).strip()

    return _TextBundle(left_text_p1, right_text_p1, full_text_p1, full_text_all, per_page_text), all_table_rows


class _TextBundle:
    """Carries page-1 left/right/full text + all-pages full text + per-page list."""
    def __init__(self, left: str, right: str, full: str, full_all: str = "", pages: list = None):
        self.left     = left
        self.right    = right
        self.full     = full
        self.full_all = full_all or full
        self.pages    = pages or []   # index 0 = page 1, index 1 = page 2, etc.

    def __bool__(self):  return bool(self.full)
    def __str__(self):   return self.full
    def __len__(self):   return len(self.full)


def detect_fields_regex(text: str) -> list:
    """Scan text for label: value patterns and return all found pairs."""
    pairs = []
    seen  = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Za-z][A-Za-z0-9\s\/\(\)\-\.]{1,60}?)\s*:\s*(.+)$', line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key and val and key not in seen and len(key) < 60:
                pairs.append({"key": key, "value": val})
                seen.add(key)
            continue
        m = re.match(r'^([A-Za-z][A-Za-z0-9\s\/\(\)\-]{1,60}?)\.\s+(.+)$', line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            if key and val and key not in seen and len(key) < 60 and len(val) < 200:
                pairs.append({"key": key, "value": val})
                seen.add(key)
    return pairs


def _search_label_in_text(text: str, labels: list[str]) -> str:
    """Try all label aliases against the text, return first match."""
    for label in labels:
        escaped = re.escape(label)
        # colon separator (handles "Label:" and "Label :")
        m = re.search(rf'{escaped}\s*:\s*(.+)', text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = re.split(r'\n', m.group(1))[0].strip()
            # Truncate at 2+ spaces — handles both layout=True column joins (2 spaces)
            # and wide-gap columns (5+ spaces)
            val = re.split(r'\s{2,}', val)[0].strip()
            if val:
                return val
        # dot separator
        m = re.search(rf'{escaped}\.\s+(.+)', text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = re.split(r'\n', m.group(1))[0].strip()
            val = re.split(r'\s{2,}', val)[0].strip()
            if val:
                return val
        # space separator — "GSTIN 09AAACC1206D2ZD" (no colon/dot)
        m = re.search(rf'^{escaped}\s+([^\s].+)$', text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip()
            val = re.split(r'\s{2,}', val)[0].strip()
            # sanity check: value shouldn't look like another label
            if val and not re.match(r'^(and|or|the|is|are|for|of|in|to)\b', val, re.IGNORECASE):
                return val
        # @ symbol (e.g. "TDS to be deducted @2%")
        m = re.search(rf'{escaped}.*?@([\d\.]+%)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _extract_from_table_rows(field_lower: str, table_rows: list[list]) -> str:
    """
    For financial fields (amounts, rates, period) that live in a table,
    find the data row and map by column position.
    """
    header_map: dict[str, int] = {}   # field → column index in data rows
    rate_from_header: dict[str, str] = {}  # field → rate string extracted from header text
    data_rows = []

    for row in table_rows:
        # Normalise cells: join multiline content within a cell
        row = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
        flat = " ".join(c.lower() for c in row if c)

        # Identify header rows by known column keywords
        if any(k in flat for k in ["value of service", "cgst", "sgst", "igst", "total tax", "total invoice"]):
            for i, cell in enumerate(row):
                cell_l = cell.lower()
                # Extract embedded rate percentages from header text e.g. "CGST RATE(9%)"
                rate_match = re.search(r'\((\d+\.?\d*%)\)', cell)
                embedded_rate = rate_match.group(1) if rate_match else None

                if "value of" in cell_l or "taxable" in cell_l or \
                   ("taxable amount" in cell_l):
                    header_map["taxable_amount"] = i
                    header_map.setdefault("period", 1)  # Period is always col 1
                elif "cgst" in cell_l:
                    header_map["cgst_amount"] = i
                    if embedded_rate:
                        rate_from_header["cgst_rate"] = embedded_rate
                elif "sgst" in cell_l or "ugst" in cell_l:
                    header_map["sgst_amount"] = i
                    if embedded_rate:
                        rate_from_header["sgst_rate"] = embedded_rate
                elif "igst" in cell_l:
                    header_map["igst_amount"] = i
                    if embedded_rate:
                        rate_from_header["igst_rate"] = embedded_rate
                elif "total tax" in cell_l:
                    header_map["total_tax"] = i
                elif "total invoice" in cell_l or "invoice amount" in cell_l:
                    header_map["total_invoice"] = i
        else:
            # Data row: must have at least 2 numeric values
            nums = [c for c in row if re.match(r'^\d[\d\.,]+$', c.replace(" ", ""))]
            if len(nums) >= 2:
                data_rows.append(row)

    # Rates are in header text, not data cells
    if field_lower in rate_from_header:
        return rate_from_header[field_lower]

    if not data_rows:
        return ""

    # Use last data row (Total row)
    total_row = data_rows[-1]

    # Period: always column 1, use FIRST data row (Total row has blank period)
    if field_lower == "period":
        for drow in data_rows:
            val = drow[1].replace("\n", "").strip() if len(drow) > 1 else ""
            if val:
                return val
        return ""

    col = header_map.get(field_lower)
    if col is not None and col < len(total_row):
        val = total_row[col].replace("\n", " ").strip()
        return val if val else ""

    return ""


def _extract_from_inline_text_table(text: str, cat: str) -> str:
    """
    For PDFs where pdfplumber finds no tables (e.g. Myntra invoices).
    Standard GST invoice column order:
      Gross | Discount | Other Charges | Taxable | [CGST] | [SGST] | [IGST] | [Cess] | Total
    We detect how many Rs-amounts the TOTAL row has and map accordingly.
    For IGST-only invoices CGST/SGST cells are blank so they produce no amounts.
    """
    lines = text.split("\n")
    tl = text.lower()

    if sum(1 for k in ["gross", "discount", "taxable", "total amount"] if k in tl) < 3:
        return ""

    # Find TOTAL row
    total_row = None
    for line in lines:
        if re.match(r'\s*TOTAL\b', line, re.IGNORECASE):
            total_row = line
            break
    if not total_row:
        for line in lines:
            if re.search(r'Rs\s+[\d,\.]+.*Rs\s+[\d,\.]+', line):
                total_row = line
                break
    if not total_row:
        return ""

    amounts = re.findall(r'Rs\s+([\d,\.]+)', total_row, re.IGNORECASE)
    if not amounts:
        return ""

    # Detect which optional columns appear in header
    # Note: "CessTotal" may run together — check for " cess " or "cess " with space
    has_discount = "discount" in tl
    has_other    = "other" in tl and "charge" in tl
    has_cgst     = "cgst" in tl
    has_sgst     = "sgst" in tl or "ugst" in tl
    has_igst     = "igst" in tl
    # Cess: only count as present if it's a standalone word, not part of "CessTotal"
    has_cess     = bool(re.search(r'\bcess\b', tl)) and not re.search(r'cesstotal', tl)

    # Fixed mandatory cols
    fixed = 1  # gross
    if has_discount: fixed += 1
    if has_other:    fixed += 1
    fixed += 1  # taxable
    fixed += 1  # total
    n_tax = len(amounts) - fixed

    # If IGST-only invoice: the single tax amount should be igst, not cgst
    # Heuristic: if has_cgst+has_sgst+has_igst but n_tax==1, prefer igst
    if n_tax == 1 and has_igst:
        active_tax = ["igst_amount"]
    elif n_tax == 2 and has_cgst and has_sgst:
        active_tax = ["cgst_amount", "sgst_amount"]
    elif n_tax == 2 and has_cgst and has_igst:
        active_tax = ["cgst_amount", "igst_amount"]
    else:
        # Build from detected columns in standard order, capped at n_tax
        tax_options = []
        if has_cgst: tax_options.append("cgst_amount")
        if has_sgst: tax_options.append("sgst_amount")
        if has_igst: tax_options.append("igst_amount")
        if has_cess: tax_options.append("cess")
        active_tax = tax_options[:max(0, n_tax)]

    reduced_seq = ["gross_amount"]
    if has_discount: reduced_seq.append("discount")
    if has_other:    reduced_seq.append("other_charges")
    reduced_seq.append("taxable_amount")
    reduced_seq.extend(active_tax)
    reduced_seq.append("total_invoice")

    col_cat_map = dict(zip(reduced_seq, amounts))
    return col_cat_map.get(cat, "")


def extract_value_smart(text_bundle, table_rows: list[list], field: str) -> str:
    """
    Extraction strategy (in order):
    1. Try the field name as a LITERAL PDF label  ← PRIMARY path now that
       detect_fields_llm returns exact labels (e.g. "Invoice No.", "GSTIN No", "PAN")
    2. Resolve to semantic category → targeted extraction  ← handles positional
       fields (company name, address), table data, ambiguous duplicates, legacy names
    3. Alias map → last-resort fallback

    text_bundle is a _TextBundle with .left, .right, .full attributes.
    For buyer fields we search .left; for seller fields we search .right;
    for unambiguous fields we search .full.
    """
    # Support plain string text (e.g. from detect_fields path)
    if isinstance(text_bundle, str):
        full_text  = text_bundle
        left_text  = text_bundle
        right_text = text_bundle
        page_texts = []
    else:
        full_text  = text_bundle.full
        left_text  = text_bundle.left
        right_text = text_bundle.right
        page_texts = getattr(text_bundle, 'pages', [])

    # ── Page suffix handling: "Invoice Number 1" → page 1, "Invoice Number 2" → page 2
    # When LLM detects the same label on multiple pages, it appends " 1", " 2" etc.
    # Strip the suffix and search only within that page's text.
    page_scoped_text = None
    field_for_lookup = field   # may have suffix stripped
    m_suffix = re.match(r'^(.+?)\s+(\d+)$', field.strip())
    if m_suffix and page_texts:
        base_label = m_suffix.group(1).strip()
        page_num   = int(m_suffix.group(2)) - 1   # convert to 0-indexed
        if 0 <= page_num < len(page_texts):
            page_scoped_text = page_texts[page_num]
            field_for_lookup = base_label

    # Alias: use full text as default working text
    # If we have a page-scoped text, prefer it for literal label search
    text = page_scoped_text if page_scoped_text else full_text
    field_lower = field_for_lookup.lower().strip()

    # ── STEP 1: Literal label search ─────────────────────────────────────────
    # When the LLM returns exact PDF label text, this handles it directly.
    # Skipped only for fields that definitely have no "Label: Value" pattern.
    POSITIONAL_CATS = {"document type", "seller_name", "buyer_name",
                       "seller_address", "buyer_address", "period",
                       "taxable_amount", "cgst_rate", "sgst_rate", "igst_rate",
                       "cgst_amount", "sgst_amount", "igst_amount",
                       "total_tax", "total_invoice", "tds_note",
                       "authorised_signatory", "e_invoice_note",
                       "hsn_sac", "amount_words", "invoice number", "due_date",
                       "qty", "gross_amount", "discount", "invoice date"}
    cat_pre = _resolve_category(field_lower)
    if cat_pre not in POSITIONAL_CATS:
        # Search using the base label (suffix stripped) in the scoped page text
        direct = _search_label_in_text(text, [field_for_lookup])
        if direct:
            return direct
        # Also try original field name in full text as fallback
        if page_scoped_text and field_for_lookup != field:
            direct2 = _search_label_in_text(full_text, [field_for_lookup])
            if direct2:
                return direct2

    # ── STEP 2: Semantic category resolution ─────────────────────────────────
    cat = cat_pre

    # ─────────────────────────────────────────────────────────────────────────
    # DOCUMENT HEADER
    # ─────────────────────────────────────────────────────────────────────────
    if cat == "document type":
        first_line = text.split("\n")[0].strip()
        return first_line if first_line and len(first_line) < 60 else "N/A"

    if cat == "invoice number":
        # "Invoice Number: I0625..." OR "Invoice No.: INV-001"
        m = re.search(r'Invoice\s*(?:Number|No\.?)\s*:\s*(\S+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Hash label: "# : INV-000002"
        m = re.search(r'^#\s*:\s*(\S+)', text, re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1).strip()
        return "N/A"

    if cat == "invoice date":
        # Bare "Date" field (e.g. page 2 of Myntra) — search all pages for standalone Date:
        # not preceded by "Invoice" or "Order" to avoid cross-page contamination.
        if field_for_lookup.strip().lower() == "date":
            search_src = page_scoped_text if page_scoped_text else (text_bundle.full_all if hasattr(text_bundle, "full_all") else full_text)
            m = re.search(
                r'(?<!Invoice\s)(?<!Invoice )(?<!Order\s)(?<!Order )(?<![a-zA-Z])'
                r'Date\s*:\s*([\d\w\s\-\/,]+?)(?:\s{2,}|\n|$)',
                search_src, re.IGNORECASE | re.MULTILINE
            )
            if m: return m.group(1).strip()
        # "Invoice Date: 03 Feb 2025" — capture full date incl. month name
        m = re.search(r'Invoice\s*Date\s*:\s*([\d\w\s\-\/,]+?)(?:\s{2,}|\n|$)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Generic "Date: ..." fallback
        m = re.search(r'(?<!\w)Date\s*:\s*([\d\w\s\-\/,]+?)(?:\s{2,}|\n|$)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        return "N/A"

    if cat == "due_date":
        # Stop at next label — avoids "15-Feb-2026 Status: Unpaid"
        m = re.search(r'Due\s*Date\s*:\s*([\d\w\-\/]+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "order_date":
        m = re.search(r'Order\s*Date\s*:\s*(.+)', text, re.IGNORECASE)
        if m: return re.split(r'\s{2,}|\t', m.group(1))[0].strip()
        return "N/A"

    if cat == "order_number":
        m = re.search(r'Order\s*(?:Number|No\.?)\s*:\s*(\S+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "nature_transaction":
        m = re.search(r'Nature\s*of\s*Transaction\s*:\s*(.+)', text, re.IGNORECASE)
        if m:
            # stop before next label (two-column layouts have labels on same line)
            val = re.split(r'\s{2,}|(?=\b[A-Z][a-z]+\s+(?:Number|Date|of)\s*:)', m.group(1))[0]
            return val.strip()
        return "N/A"

    if cat == "nature_supply":
        m = re.search(r'Nature\s*of\s*Supply\s*:\s*(.+)', text, re.IGNORECASE)
        if m: return re.split(r'\s{2,}', m.group(1))[0].strip()
        return "N/A"

    if cat == "qty":
        # Find first numeric qty in a data row (not a header row) — look for "1" before Rs amounts
        # Data row pattern: "1  Rs 2999.00..." or just "1" at start of line after HSN
        m = re.search(r'^(\d+)\s+Rs\s+[\d,\.]+', text, re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1).strip()
        # Fallback: TOTAL row prefix digit
        m = re.search(r'(?:TOTAL|^(\d+))\s+Rs\s+[\d,\.]+', text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1): return m.group(1).strip()
        return "N/A"

    if cat == "gross_amount":
        v = _extract_from_inline_text_table(text, "gross_amount")
        return v if v else "N/A"

    if cat == "discount":
        v = _extract_from_inline_text_table(text, "discount")
        return v if v else "N/A"

    if cat == "packet_id":
        m = re.search(r'Packet\s*ID\s*:\s*(\S+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "reference number":
        m = re.search(r'Reference\s*No\.?\s*:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip().lstrip(':').strip() if m else "N/A"

    if cat == "terms":
        m = re.search(r'Terms\s*:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    # ─────────────────────────────────────────────────────────────────────────
    # SELLER / VENDOR / SUPPLIER
    # ─────────────────────────────────────────────────────────────────────────
    if cat == "seller_name":
        # Seller is always at the TOP of the invoice, before Billing Details
        billing_pos = full_text.lower().find("billing details")
        header_text = full_text[:billing_pos] if billing_pos > 0 else full_text
        # Pattern 1: explicit From: label
        m = re.search(r'(?:Bill\s*From|From)\s*:?\s*\n(.+)', header_text, re.IGNORECASE)
        if m:
            return re.split(r'\s{5,}', m.group(1).strip())[0].strip()
        # Pattern 2: first company entity line in header section
        for line in [l.strip() for l in header_text.split("\n") if l.strip()]:
            if re.search(r'(pvt\.?\s*ltd|llp|llc|ltd\.?|private limited|inc\b)', line, re.IGNORECASE):
                return re.split(r'\s{5,}', line)[0].strip()
        hlines = [l.strip() for l in header_text.split("\n") if l.strip()]
        return hlines[1] if len(hlines) > 1 else "N/A"

    if cat == "seller_address":
        # Lines between seller company name and MOB/PAN/GSTIN
        billing_pos = full_text.lower().find("billing details")
        header_text = full_text[:billing_pos] if billing_pos > 0 else full_text
        addr_lines = []
        capturing = False
        for line in [l.strip() for l in header_text.split("\n") if l.strip()]:
            if not capturing:
                if re.search(r'(pvt\.?\s*ltd|llp|llc|private limited)', line, re.IGNORECASE):
                    capturing = True
                continue
            if re.match(r'^(MOB|PAN|GSTIN|TAX INVOICE|ORIGINAL)', line, re.IGNORECASE):
                break
            addr_lines.append(line)
        if addr_lines:
            return ", ".join(addr_lines)
        # Fallback: Survey/Plot line
        m = re.search(r'((?:Survey|Plot)\s*No[^,\n]+(?:,\s*[^\n]+)?)', header_text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "seller_pan":
        # Seller PAN appears BEFORE Billing Details
        billing_pos = full_text.lower().find("billing details")
        search_text = full_text[:billing_pos] if billing_pos > 0 else full_text
        m = re.search(r'PAN\s*:?\s*(\S{10})', search_text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "seller_gstin":
        # Seller GSTIN appears BEFORE Billing Details
        billing_pos = full_text.lower().find("billing details")
        search_text = full_text[:billing_pos] if billing_pos > 0 else full_text
        m = re.search(r'GSTIN\s*:?\s*(\S{15})', search_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Also try right_text for two-column PDFs
        if right_text:
            m = re.search(r'GSTN?\s*:?\s*(\S{15})', right_text, re.IGNORECASE)
            if m: return m.group(1).strip()
        return "N/A"

    if cat == "seller_state_code":
        matches = list(re.finditer(r'State\s*Code\s*:?\s*(\d+)', full_text, re.IGNORECASE))
        if len(matches) >= 2: return matches[1].group(1).strip()
        return matches[0].group(1).strip() if matches else "N/A"

    if cat == "seller_place_supply":
        m = re.search(r'Place\s*[Oo]f\s*Supply\s*(?:or\s*Services)?\s*:\s*(.+)',
                      full_text, re.IGNORECASE)
        if m: return re.split(r'\s{5,}', m.group(1))[0].strip()
        return "N/A"

    if cat == "seller_place_deliv":
        m = re.search(r'Place\s*of\s*[Dd]elivery\s*(?:\([^)]+\))?\s*:\s*(.+)',
                      full_text, re.IGNORECASE)
        if m: return re.split(r'\s{5,}', m.group(1))[0].strip()
        return "N/A"

    if cat == "seller_rev_charge":
        m = re.search(r'(?:Whether\s+)?[Tt]ax\s+payable\s+under\s+[Rr]everse\s+[Cc]harge\s*:\s*(\S+)',
                      full_text, re.IGNORECASE)
        if not m:
            m = re.search(r'[Rr]everse\s+[Cc]harge\s*[:\s]+(\S+)', full_text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "seller_msme":
        m = re.search(r'^MSME\s*:\s*(\S+)', full_text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else "N/A"

    # ─────────────────────────────────────────────────────────────────────────
    # BUYER / RECIPIENT / CUSTOMER
    # ─────────────────────────────────────────────────────────────────────────
    if cat == "buyer_name":
        # Pattern 1: "TO,\nCompany"
        m = re.search(r'TO\s*,\s*\n(.+)', full_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Pattern 2: "Bill To / Ship To:\n..."
        m = re.search(r'Bill\s*[Tt]o\s*[/|]?\s*Ship\s*[Tt]o\s*:?\s*\n(.+)', full_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Pattern 3: "Bill To:\nCompany" or "Bill to:\n"
        m = re.search(r'Bill\s*[Tt]o\s*:?\s*\n(.+)', full_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Pattern 4: company name right after "Billing Details" section header
        m = re.search(r'Billing\s*Details\s*\n(.+)', full_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        return "N/A"

    if cat == "buyer_address":
        # Lines after buyer company name (and optional Kind Attention line), before GSTIN/PAN
        m = re.search(r'Billing\s*Details\s*\n.+\n(?:Kind\s*Attention[^\n]*\n)?(.+(?:\n.+){0,5})',
                      full_text, re.IGNORECASE)
        if m:
            out = []
            for line in [l.strip() for l in m.group(1).split("\n") if l.strip()]:
                if re.match(r'^(GSTIN|PAN|Place|Kind)', line, re.IGNORECASE): break
                out.append(line)
            if out: return " ".join(out)
        m = re.search(r'TO\s*,\s*\n.+\n(.+(?:\n.+){0,4})', full_text, re.IGNORECASE)
        if m:
            out = []
            for line in [l.strip() for l in m.group(1).split("\n") if l.strip()]:
                if re.match(r'^(india|gstin|state code|pan)\b', line, re.IGNORECASE): break
                out.append(line)
            if out: return " ".join(out)
        return "N/A"

    if cat == "buyer_gstin":
        # Buyer GSTIN appears AFTER Billing Details section
        billing_pos = full_text.lower().find("billing details")
        search_text = full_text[billing_pos:] if billing_pos > 0 else full_text
        m = re.search(r'GSTIN\s*:?\s*(\S{15})', search_text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Fallback: left_text for two-column PDFs
        if left_text:
            m = re.search(r'GSTN?\s*:\s*(\S+)', left_text, re.IGNORECASE)
            if m: return m.group(1).strip()
        return "N/A"

    if cat == "buyer_state_code":
        m = re.search(r'State\s*Code\s*:?\s*(\d+)', full_text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"


    # ─────────────────────────────────────────────────────────────────────────
    # SERVICE
    # ─────────────────────────────────────────────────────────────────────────
    if cat == "description_service":
        m = re.search(r'Description\s*of\s*Service\s*:\s*(.+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "hsn_sac":
        # Pattern 1: "HSN/SAC: 846755" label:value in text
        m = re.search(r'HSN\s*/\s*SAC\s*:\s*(\S+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.search(r'(?:HSN|SAC)\s*(?:Code)?\s*:\s*(\S+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # Pattern 2: value in table data row (skip header rows, look for 4-8 digit code)
        in_header = True
        for row in table_rows:
            flat = " ".join(c for c in row if c).lower()
            if any(k in flat for k in ["hsn", "sac", "description", "item"]):
                in_header = False
                continue
            if not in_header:
                for cell in row:
                    if cell and re.match(r'^\d{4,8}$', cell.strip()):
                        return cell.strip()
        # Pattern 3: 4-8 digit number appearing on the same line as item description
        m = re.search(r'\b(\d{4,8})\b', text)
        if m: return m.group(1)
        return "N/A"

    if cat == "amount_words":
        # "Indian Rupee Twenty..." standalone line (most reliable)
        m = re.search(r'(Indian Rupee[^\n]+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        # "Total In Words\n<blank lines>\nActual text"
        m = re.search(r'Total\s*In\s*Words\s*\n(?:\n*)(.+)', text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # skip if it looks like a tax line
            if not re.match(r'^(cgst|sgst|igst)', val, re.IGNORECASE):
                return val
        # "Rupees: Three Hundred..."
        m = re.search(r'(?:Amount|Total)\s*in\s*words?\s*:?\s*(.+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        m = re.search(r'Rupees?\s*:\s*(.+)', text, re.IGNORECASE)
        if m: return m.group(1).strip()
        return "N/A"

    if cat == "period":
        result = _extract_from_table_rows("period", table_rows)
        return result if result else "N/A"

    # ─────────────────────────────────────────────────────────────────────────
    # TAX TABLE VALUES
    # ─────────────────────────────────────────────────────────────────────────
    TABLE_CATS = {"taxable_amount", "cgst_rate", "sgst_rate", "igst_rate",
                  "cgst_amount", "sgst_amount", "igst_amount", "total_tax", "total_invoice"}
    if cat in TABLE_CATS:
        result = _extract_from_table_rows(cat, table_rows)
        if result: return result
        # Fallback: parse inline text table (e.g. Myntra invoices have no pdfplumber tables)
        # Header row: "Qty  Gross Amount  Discount  Other Charges  Taxable Amount  CGST  SGST/UGST  IGST  Cess  Total Amount"
        # Data row:   "1  Rs 2999.00  Rs 1920.00  Rs 0.00  Rs 963.39  Rs 115.61  Rs 1079.00"
        # We find the TOTAL row and map values to column positions from the header
        inline = _extract_from_inline_text_table(text, cat)
        if inline: return inline
        # Fallback: summary block extraction for invoices where totals appear as
        # plain "Label : Amount" lines (no table structure) e.g. "Sub Total : 3,000.00"
        summary_patterns = {
            "taxable_amount": r'Total\s*Invoice\s*Value\s*\(In\s*Figure\)\s*:\s*([\d,\.]+)|Sub\s*Total\s*[:\s]+([\d,\.]+)',
            "igst_amount":    r'Total\s*GST\s*Value\s*\(In\s*Figure\)\s*:\s*([\d,\.]+)|IGST\s*\([\d\.]+%\)\s*[:\s]+([\d,\.]+)',
            "total_invoice":  r'Total\s*Invoice\s*Value\s*\(In\s*Figure\)\s*:\s*([\d,\.]+)|Total\s*\(INR\)\s*[:\s]+([\d,\.]+)|Total\s*Due\s*Amount\s*[:\s]+([\d,\.]+)|Balance\s*Due\s+[₹]?\s*([\d,\.]+)',
            "total_tax":      r'Total\s*GST\s*Value\s*\(In\s*Figure\)\s*:\s*([\d,\.]+)|Total\s*Tax\s+(?:Rs\s*)?([\d,\.]+)',
            "cgst_amount":    r'CGST\s*\([\d\.]+%\)\s*[:\s]+([\d,\.]+)',
            "sgst_amount":    r'SGST\s*\([\d\.]+%\)\s*[:\s]+([\d,\.]+)',
            "cgst_rate":      r'CGST\s*\(([\d\.]+)%\)',
            "sgst_rate":      r'SGST\s*\(([\d\.]+)%\)',
            "igst_rate":      r'IGST\s*\(([\d\.]+)%\)|HSN:\s*[\d]+,\s*([\d\.]+)%\s*IGST',
        }
        pattern = summary_patterns.get(cat)
        if pattern:
            # Search full_all (all pages) since totals often appear on the last page
            # Use scoped page text when available (handles "Tax Rate 2" → page 2 only)
            search_src = text if text != full_text else (full_text if isinstance(text_bundle, str) else text_bundle.full_all)
            m = re.search(pattern, search_src, re.IGNORECASE | re.MULTILINE)
            if m:
                # some patterns have multiple groups (alternatives)
                val = next((g for g in m.groups() if g), None)
                if val:
                    return val.replace('₹', '').strip()
        return "N/A"

    # ─────────────────────────────────────────────────────────────────────────
    # TDS & MISC
    # ─────────────────────────────────────────────────────────────────────────
    if cat == "tds_rate":
        m = re.search(r'TDS.*?@\s*([\d\.]+%)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "tds_note":
        m = re.search(r'\(?(TDS to be deducted[^)\n]+)\)?', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    if cat == "authorised_signatory":
        # Appears after "For <company name>" line
        m = re.search(r'For\s+.+\n(.+)', text, re.IGNORECASE)
        if m and "signatory" in m.group(1).lower():
            return m.group(1).strip()
        m2 = re.search(r'Authorised?\s*Signatory', text, re.IGNORECASE)
        return "Authorised Signatory" if m2 else "N/A"

    if cat == "e_invoice_note":
        m = re.search(r'(We hereby declare.+?(?:March \d+, \d{4}|exempted[^\n]+))',
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            return " ".join(m.group(1).split())  # collapse whitespace
        m = re.search(r'(exempted from issuance[^\n]+)', text, re.IGNORECASE)
        return m.group(1).strip() if m else "N/A"

    # ── STEP 3: Alias map + last-resort fallbacks ─────────────────────────────
    aliases = FIELD_ALIASES.get(field_lower, [])
    if aliases:
        result = _search_label_in_text(text, aliases)
        if result:
            return result

    # Try direct label search here too (catches positional-category fields
    # that weren't tried in Step 1)
    direct = _search_label_in_text(text, [field])
    if direct:
        return direct

    # Last resort: last word of multi-word field name
    words = field.split()
    if len(words) > 1:
        result = _search_label_in_text(text, [words[-1]])
        if result:
            return result

    return "N/A"


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION  (used for first PDF always, and all PDFs if different templates)
# ═══════════════════════════════════════════════════════════════════════════════

async def groq_chat(prompt: str) -> str:
    api_key = await _next_groq_key()
    key_hint = f"...{api_key[-6:]}"   # last 6 chars for log tracing only

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": GROQ_MODEL,
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a document data extraction expert for legal and accounting firms in India. "
                            "Always respond with valid JSON only — no markdown, no explanation, no code fences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )

    logger.info("Groq call via key %s → HTTP %s", key_hint, response.status_code)
    data = response.json()
    if response.status_code != 200:
        error_msg = data.get("error", {}).get("message", f"Groq API returned HTTP {response.status_code}")
        raise HTTPException(status_code=500, detail=f"Groq error (key {key_hint}): {error_msg}")
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"].get("message", "Groq API error"))
    if not data.get("choices"):
        raise HTTPException(status_code=500, detail=f"Groq returned no choices. Response: {data}")

    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning") or ""
    if not content.strip():
        raise HTTPException(status_code=500, detail=f"Groq returned empty content. Full message: {message}")
    return content


async def detect_fields_llm(text: str, table_rows: list[list] = None) -> list:
    """Use Groq LLM to detect all key-value pairs from document text,
    then append table line-item fields detected purely via Python."""
    raw = await groq_chat(f"""You are a senior chartered accountant and GST compliance expert preparing data for a statutory audit and GST return filing.

You are reviewing a tax invoice document. Your task is to identify and extract EVERY data field present.
Accuracy is critical — this data feeds directly into GST filings and audit trails.

STEP 1 — IDENTIFY PARTIES BEFORE EXTRACTING ANYTHING:
Every invoice has exactly two parties. Identify them from their section labels:

  SELLER (supplier) = issues the invoice, gets paid
    Section labels: "From:", "Bill From:", "Supplier:", or company name printed at the TOP of the invoice
    Their GSTIN is in the "From" section or labeled "Seller GSTIN" / "From GSTIN"

  BUYER (recipient) = receives the invoice, makes the payment
    Section labels: "Bill to:", "To:", "Ship to:", "Buyer:"
    Their GSTIN is in the "Bill to" section or labeled "Buyer GSTIN"

  ⚠️  NEVER swap these two parties — it causes GST return mismatches and tax notices.
  Example: if the document shows "Bill to: ANALOG LEGALHUB" then ANALOG LEGALHUB = BUYER (not seller).
  Example: if the document shows "From: CTRL S CONNECTIVITY" then CTRL S CONNECTIVITY = SELLER (not buyer).

STEP 2 — EXTRACTION RULES:
1. The "key" MUST be the exact label text from the document. Do NOT rename or rephrase.
   - "Invoice No." → key is "Invoice No." (not "Invoice Number")
   - "GSTIN No:" → key is "GSTIN No" (not "Vendor GSTIN")
   Exception: for unlabeled positional fields, assign these standard keys — correctly mapped to party:
   "Seller Company Name", "Seller Address", "Buyer Company Name", "Buyer Address"
2. Value must be exact as printed. Do not reformat amounts, dates, or codes.
3. For table columns (CGST, SGST, IGST, taxable value, total) use the column header as key.
4. Extract everything — both parties' details, all tax amounts, all footer fields.
5. When the same label appears for both parties (e.g. two GSTINs, two PANs):
   Label them "Seller GSTIN" and "Buyer GSTIN" (or "Seller PAN" / "Buyer PAN") based on which section they appear in.
6. Do NOT include section headers ("Tax Invoice", "Billing Details", "Transaction Details") — no value to extract.
7. Do NOT include line-item product rows — handled separately.
8. Values must be concise — not paragraphs or multi-line blobs.

Return ONLY a valid JSON array — no markdown, no explanation:
[{{"key": "label", "value": "value"}}, ...]

INVOICE DOCUMENT:
{text[:6000]}""")

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result  = json.loads(cleaned)
    fields  = result if isinstance(result, list) else []

    # ── Append table line-item fields (pure Python, zero LLM calls) ──────────
    # These look like "Description_1", "Qty_2" etc. in the UI field selector.
    if table_rows:
        line_items = extract_line_items_from_tables(table_rows)
        seen = set()
        for key, value in line_items.items():
            if key not in seen:
                seen.add(key)
                fields.append({"key": key, "value": value})

    return fields


async def extract_fields_llm(text: str, fields: list) -> dict:
    """Use Groq LLM to extract specific fields from document text."""
    # Normalise: fields may be strings or dicts
    field_strs = [f if isinstance(f, str) else f.get("key", str(f)) for f in fields]
    fields_quoted = chr(10).join(f'  "{f}"' for f in field_strs)

    raw = await groq_chat(f"""You are a senior chartered accountant and GST compliance expert with 20+ years of experience in invoice auditing and tax filing.

You have been handed a tax invoice document and must extract specific fields from it with absolute precision.
This data will be used directly for GST filing and statutory audit — errors have legal and financial consequences.

CRITICAL PARTY IDENTIFICATION — READ THIS FIRST:
An invoice has exactly two parties. You MUST identify them correctly before extracting anything:

  SELLER (supplier) = the company ISSUING the invoice = the one getting paid
    Identified by labels: "From:", "Bill From:", "Supplier:", "We/Our company", or the company name at the TOP of the invoice
    Their GSTIN is labeled "Seller GSTIN", "From GSTIN", "Supplier GSTIN", or GSTIN in the "From" section
    Their address is in the "From" / "Bill From" section

  BUYER (recipient) = the company RECEIVING the invoice = the one paying
    Identified by labels: "Bill to:", "To:", "Buyer:", "Ship to:"
    Their GSTIN is labeled "Buyer GSTIN", "GSTIN" in the "Bill to" section
    Their address is in the "Bill to" / "To" section

  NEVER swap these. A buyer labeled as seller in a GST return causes a notice from the tax department.
  If you see "Bill to: ANALOG LEGALHUB" — ANALOG LEGALHUB is the BUYER, not the seller.
  If you see "From: CTRL S CONNECTIVITY" — CTRL S CONNECTIVITY is the SELLER, not the buyer.

YOUR RESPONSIBILITIES:
- First identify which party is the seller and which is the buyer using the labels above
- Then map all requested fields to the correct party
- For fields like GSTIN, PAN, IRN, HSN — verify the format (GSTIN=15 chars, PAN=10 chars)
- For duplicate labels (e.g. two GSTINs) — always map to the correct party, not just the first occurrence
- Return amounts exactly as shown — do not round, convert, or reformat
- For multi-line addresses — capture the complete address, not just the first line

EXTRACTION RULES:
1. JSON keys MUST match the requested field names EXACTLY (same spelling, same capitalisation)
2. "Seller *" fields belong to the FROM/supplier party. "Buyer *" fields belong to the BILL TO/recipient party
3. Fields ending in "_occ1" or " 1" = first occurrence in document order
   Fields ending in "_occ2" or " 2" = second occurrence in document order
4. Two-column lines like "Invoice Date: X   Due Date: Y" — return ONLY the value for the exact label. Stop at 2+ spaces
5. Return values exactly as they appear in the document
6. If genuinely not present, return "N/A". Never fabricate values.

Requested fields (copy these EXACT strings as JSON keys):
{fields_quoted}

Return ONLY a valid JSON object — no markdown, no explanation, no preamble.

INVOICE DOCUMENT:
{text[:8000]}""")

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("extract_fields_llm: JSON parse failed — returning N/A for all fields")
        return {f: "N/A" for f in field_strs}

    if not isinstance(result, dict):
        return {f: "N/A" for f in field_strs}

    # Remap keys case-insensitively so minor capitalisation differences don't break things
    result_lower = {k.lower().strip(): v for k, v in result.items()}
    remapped = {}
    for f in field_strs:
        if f in result:
            remapped[f] = result[f]
        elif f.lower().strip() in result_lower:
            remapped[f] = result_lower[f.lower().strip()]
        else:
            remapped[f] = "N/A"
    return remapped


async def _fix_schema_llm(
    bad_schema: dict,
    mismatch_info: str,
    sample_text: str,
    left_text: str = "",
    right_text: str = ""
) -> dict:
    """
    Ask the LLM to fix ONLY the broken regex patterns.
    Working patterns are kept in Python — never sent to LLM — saving tokens
    and preventing the LLM from accidentally modifying correct patterns.
    """
    clean = "\n".join(l.strip() for l in sample_text.split("\n") if l.strip())[:5000]

    # Parse broken field names from mismatch_info lines like:
    #   "FieldName": want="X" got="Y" | PDF line: '...'
    broken_fields = set()
    for line in mismatch_info.split("\n"):
        line = line.strip()
        # Strip any leading emoji/spaces
        line = re.sub(r'^[\s❌✅⚠️]+', '', line).strip()
        if line.startswith('"'):
            name = line.lstrip('"').split('"')[0]
            if name:
                broken_fields.add(name)

    # Only the broken patterns go to the LLM
    broken_only = {k: v for k, v in bad_schema.items() if k in broken_fields}
    # Working patterns stay in Python — never sent to LLM
    working_schema = {k: v for k, v in bad_schema.items() if k not in broken_fields}

    logger.info(
        "_fix_schema_llm: sending %d broken patterns to LLM, keeping %d working patterns in Python",
        len(broken_only), len(working_schema)
    )

    if not broken_only:
        logger.warning("_fix_schema_llm: no broken fields parsed from mismatch_info — returning unchanged schema")
        return bad_schema

    broken_patterns_str = "\n".join(
        f'  "{k}": {v!r}' for k, v in broken_only.items()
    )
    broken_fields_str = "\n".join(
        f'  "{f}"' for f in sorted(broken_fields)
    )

    raw = await groq_chat(rf"""Fix the broken regex patterns below.
Only these {len(broken_only)} patterns are broken. Do NOT touch any others.

## PDF TEXT:
---
{clean}
---

## WHAT WENT WRONG — each line: field name, expected value, what the pattern returned, actual PDF line:
{mismatch_info}

## THE BROKEN PATTERNS TO FIX:
{broken_patterns_str}

## YOUR TASK:
Rewrite each broken pattern so it correctly extracts the expected value.
Apply the same rules as before:
- Double all backslashes: \\s \\d \\w \\n (not single \s \d \w \n)
- Stop at two-column boundary: use (?=\\s{{2,}}|$)
- Generic patterns only — not hardcoded to this PDF's specific values
- Look at the actual PDF line shown for each field and write the pattern for what you see

## RETURN:
A JSON object with ONLY the fixed fields — nothing else:
{{"field name": "fixed_pattern", ...}}

Fields that need fixing:
{broken_fields_str}

Return ONLY the JSON. No markdown. No explanation.""")

    cleaned = raw.strip()
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    cleaned = cleaned.rstrip("```").strip()

    def _fix_escapes(s: str) -> str:
        valid = frozenset(chr(34) + chr(92) + "/bfnrtu")
        out, i = [], 0
        while i < len(s):
            if s[i] == chr(92) and i + 1 < len(s):
                out.append(s[i] if s[i+1] in valid else chr(92) + chr(92))
                i += 1
            else:
                out.append(s[i])
            i += 1
        return "".join(out)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            result = json.loads(_fix_escapes(cleaned))
        except json.JSONDecodeError as e:
            logger.warning("_fix_schema_llm JSON parse failed: %s — keeping original schema", e)
            return bad_schema

    if not isinstance(result, dict):
        logger.warning("_fix_schema_llm: LLM returned non-dict — keeping original schema")
        return bad_schema

    # Merge: start with ALL original patterns (working ones preserved)
    # then override only the fields the LLM was asked to fix
    merged = {**bad_schema}
    fixed_count = 0
    for k, v in result.items():
        if isinstance(v, str) and "(" in v:
            merged[k] = v
            fixed_count += 1

    logger.info(
        "_fix_schema_llm: LLM fixed %d/%d broken patterns — total schema: %d patterns",
        fixed_count, len(broken_only), len(merged)
    )
    return merged


async def generate_schema_llm(sample_text: str, fields: list[str],
                              left_text: str = "",
                              right_text: str = "") -> dict:
    """
    Generate regex patterns for each field.
    Uses full text as primary source. Left/right columns shown as supplementary
    context only for fields that appear in both columns (duplicates).
    """
    full_lines_clean = [l.strip() for l in sample_text.split("\n") if l.strip()]
    clean_sample     = "\n".join(full_lines_clean)[:7000]

    # Build right-column lines for supplementary context on duplicate fields
    right_lines_clean = [l.strip() for l in right_text.split("\n") if l.strip()] if right_text else []
    right_col_text    = "\n".join(right_lines_clean)[:2000]

    # ── Per-field context: show the exact PDF line where each value appears ──
    # Also build a value lookup from the fields list (detect-fields LLM output)
    # so we can find positional fields that have no label in the text.
    field_value_lookup = {}
    if isinstance(fields, list):
        for f in fields:
            if isinstance(f, dict) and "key" in f and "value" in f:
                v = str(f.get("value", "")).strip()
                if v and v != "N/A":
                    field_value_lookup[f["key"]] = v

    field_contexts = []
    full_lines     = full_lines_clean

    for field in fields:
        # fields can be a list of dicts or plain strings
        field_key = field["key"] if isinstance(field, dict) else field

        occ_match   = re.match(r'^(.+)_occ(\d+)$', field_key)
        base_label  = occ_match.group(1) if occ_match else field_key
        occ_idx     = int(occ_match.group(2)) if occ_match else 1
        label_lower = base_label.lower()

        # Step 1: Find lines containing the label text
        matching_lines = []
        for idx, line in enumerate(full_lines):
            if label_lower in line.lower():
                ctx = line
                if idx + 1 < len(full_lines):
                    ctx += " | NEXT: " + full_lines[idx + 1]
                matching_lines.append((idx, ctx))

        # Step 2: If label not found, search by VALUE — find which line contains it
        # and show the PREVIOUS line as structural context (the anchor)
        if not matching_lines:
            known_val = field_value_lookup.get(field_key, "")
            if known_val and len(known_val) > 3:
                for idx, line in enumerate(full_lines):
                    if known_val.lower()[:30] in line.lower():
                        # Show preceding line (anchor) → this line (value)
                        prev = full_lines[idx - 1] if idx > 0 else ""
                        nxt  = full_lines[idx + 1] if idx + 1 < len(full_lines) else ""
                        ctx  = f"[PREV LINE: {prev!r}] → VALUE ON THIS LINE: {line!r}"
                        if nxt:
                            ctx += f" | NEXT: {nxt!r}"
                        matching_lines.append((idx, ctx))
                        break

        # For _occ2+ fields, also check right-column text
        right_note = ""
        if occ_idx >= 2 and right_lines_clean:
            right_hits = [l for l in right_lines_clean if label_lower in l.lower()]
            if right_hits:
                right_note = f" [RIGHT COLUMN LINE: {right_hits[0]!r}]"

        if matching_lines:
            # Pick the Nth occurrence
            pair = matching_lines[occ_idx - 1] if occ_idx - 1 < len(matching_lines) else matching_lines[-1]
            pick = pair[1]
        else:
            pick = "(not found in text)"

        occ_note = f" [occurrence {occ_idx} of {len(matching_lines)}]" if occ_match else ""
        field_contexts.append(
            f'  "{field_key}" (label="{base_label}{occ_note}"): {pick}{right_note}'
        )

    field_context_block = "\n".join(field_contexts)
    fields_block = "\n".join(
        f'  "{f["key"] if isinstance(f, dict) else f}"' for f in fields
    )

    raw = await groq_chat(rf"""You are an expert regex pattern generator for structured document data extraction.

## THE DOCUMENT TEXT:
---
{clean_sample}
---

## EACH FIELD AND THE EXACT LINE WHERE ITS VALUE APPEARS:
{field_context_block}

## YOUR TASK:
Write ONE Python regex pattern per field that extracts its value from any document with the same layout.
Return ONLY a valid JSON object: {{"Field Name": "pattern", ...}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 ─ BACKSLASH ESCAPING  (the #1 cause of broken patterns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In JSON strings, every regex backslash must be written as \\\\ (four chars in source = two backslashes).
✅ CORRECT:  "Invoice:\\\\s*(\\\\S+)"
❌ WRONG:    "Invoice:\s*(\S+)"    ← invalid JSON, pattern silently returns nothing

Every regex token must be doubled:
  \\\\s  \\\\d  \\\\w  \\\\n  \\\\S  \\\\D  \\\\W  \\\\.  \\\\(  \\\\)  \\\\b  \\\\+  \\\\*  \\\\?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 ─ LOOK AT THE ACTUAL LINE. WRITE WHAT YOU SEE.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each field above shows you the EXACT line it appears on. Use that line to write the pattern.
Do not guess. Do not use the field name as if it were a label if it has no label in the text.

STEP-BY-STEP for each field:
  1. Look at the line shown: "Invoice Number: I0625NG000064469  PacketID: 8303285326"
  2. Identify what separates the label from the value: colon, space, etc.
  3. Identify what ends the value: newline, or 2+ spaces if another field follows on the same line
  4. Write the pattern accordingly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 ─ STOP AT TWO-COLUMN BOUNDARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a line has TWO fields separated by 2+ spaces, you MUST stop at the boundary:
  Line: "Invoice Number: I0625NG  PacketID: 8303285326"
  ✅  "Invoice Number:\\\\s*([^\\\\n]*?)(?=\\\\s{{2,}}|$)"   ← stops before PacketID
  ❌  "Invoice Number:\\\\s*([^\\\\n]+)"                  ← grabs the whole line

Use (?=\\\\s{{2,}}|$) whenever a value might have another field to its right.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 ─ GENERIC PATTERNS ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patterns run on OTHER documents with the SAME LAYOUT but DIFFERENT values.
  ✅  "Invoice Number:\\\\s*(\\\\S+)"   ← matches any invoice number
  ❌  "I0625NG000064469"              ← hardcoded, breaks on every other document

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 ─ PATTERN TEMPLATES BY SITUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose the template that matches what you see in the actual line:

  [A] Label: Value   (value goes to end of line)
      "Label Name:\\\\s*([^\\\\n]+)"

  [B] Label: Value   (another field follows on the same line after 2+ spaces)
      "Label Name:\\\\s*([^\\\\n]*?)(?=\\\\s{{2,}}|\\\\n|$)"

  [C] Date or multi-word value with right-column neighbour
      "Invoice Date:\\\\s*([\\\\d\\\\w\\\\s\\\\-\\/,]+?)(?=\\\\s{{2,}}|\\\\n|$)"

  [D] Value follows label on the NEXT line
      "Label Name:\\\\s*\\\\n([^\\\\n]+)"

  [E] Single-word or code value (no spaces in value)
      "Label:\\\\s*(\\\\S+)"

  [F] Numeric amount with currency prefix  e.g. "Sub Total : 3,000.00"
      "Sub Total\\\\s*:\\\\s*([\\\\d,\\.]+)"

  [G] Percentage  e.g. "IGST(18%) : 540.00"
      "IGST\\\\(?([\\\\d\\.]+)%\\\\)?\\\\s*:\\\\s*([\\\\d,\\.]+)"

  [H] Fixed-format code (GSTIN=15 chars, PAN=10, IRN=64 hex chars)
      "GSTIN\\\\s*:\\\\s*(\\\\S{{15}})"
      "IRN\\\\s*:\\\\s*([a-f0-9]{{64}})"

  [I] Value is Nth number on a data row  e.g. amounts in a table row
      Use surrounding context: "TOTAL\\\\s+([\\\\d,\\.]+)\\\\s+[\\\\d,\\\\.]+"
      Or anchor to line start:  "^(\\\\d+)\\\\s+Rs" for quantity


  [J] POSITIONAL field — value has NO label, appears at a fixed structural position.
      The context shows: [PREV LINE: 'anchor'] -> VALUE ON THIS LINE: 'value'
      Use the ANCHOR LINE (prev line) to write the pattern, NOT the field name:
        Seller name after "TAX INVOICE":         "TAX INVOICE\\\\s*\\\\n([^\\\\n]+)"
        Seller address after "Authorised Signatory": "Authorised Signatory\\\\s*\\\\n([^\\\\n]+)"
        Buyer name after "TO,":                  "TO,\\\\s*\\\\n([^\\\\n]+)"
      NEVER try to match the field name itself if it is not in the document.

  [K] Nth OCCURRENCE of a repeated label (context shows [occurrence 2 of 2])
      For occ1: normal pattern matches the first occurrence automatically.
      For occ2: anchor to something unique that appears ONLY near the second value:
        e.g. State Code appears twice; occ2 is near "GSTIN: 09...":
          "GSTIN:\\\\s*09[^\\\\n]*\\\\n(?:[^\\\\n]*\\\\n)?State Code[:\\\\s]*(\\\\S+)"
        Or use re.findall approach: capture ALL occurrences and pick Nth in _run_schema.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 ─ EVERY FIELD GETS A PATTERN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a pattern for every field. Never omit one. If unsure, write your best guess.

## FIELDS — one pattern each:
{fields_block}

Return ONLY the JSON object starting with {{ and ending with }}.
""")

    cleaned = raw.strip()
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # ── Fix invalid escape sequences from LLM ─────────────────────────────────
    # LLMs sometimes write \s, \d, \n, \w as single backslash in JSON strings
    # which makes json.loads() crash with "Invalid \escape".
    # Strategy: try raw parse first, then fix escapes if it fails.
    def _fix_escapes(s: str) -> str:
        # Replace bare \x (invalid JSON escapes) with \\x so json.loads works.
        # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        # Invalid (must be doubled): \s \d \w \S \D \W \( \) \. \+ \* \? \^ \[ \] \{ \}
        valid_esc = set('"' + "\\" + "/" + "bfnrtu")
        result = []
        i = 0
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch in valid_esc:
                    result.append(s[i])
                else:
                    result.append("\\\\")
                i += 1
            else:
                result.append(s[i])
            i += 1
        return "".join(result)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Schema JSON parse failed (%s) — fixing escape sequences", e)
        try:
            result = json.loads(_fix_escapes(cleaned))
            logger.info("Schema JSON parsed successfully after escape fix")
        except json.JSONDecodeError as e2:
            raise ValueError(f"Schema LLM returned invalid JSON even after escape fix: {e2}")

    if not isinstance(result, dict):
        raise ValueError("Schema LLM returned non-dict")

    # Filter out empty or placeholder-only patterns
    valid = {k: v for k, v in result.items()
             if isinstance(v, str) and v.strip() and "(" in v}
    logger.info("Schema generated: %d/%d fields have valid patterns", len(valid), len(fields))
    # Log every generated pattern so we can see what the LLM produced
    for k, v in valid.items():
        logger.info("  Pattern [%s]: %r", k, v)
    return valid


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── POST /api/page-count ─────────────────────────────────────────────────────
# Returns total pages so the frontend can render the page selector

@app.post("/api/page-count")
async def page_count(file: UploadFile = File(...)):
    try:
        count = get_pdf_page_count(await file.read())
        return {"page_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/detect-fields ───────────────────────────────────────────────────
# Always uses LLM (this is always called on the first/preview PDF)

@app.post("/api/detect-fields")
async def detect_fields(
    file:          UploadFile = File(...),
    pages:         str        = Form("[]"),
    same_template: str        = Form("true"),   # if true, generate extractor for reuse
    session_id:    str        = Form(""),        # UUID from frontend to key the extractor
):
    try:
        file_bytes  = await file.read()
        pages_list  = json.loads(pages) if pages and pages != "[]" else None
        page_count  = get_pdf_page_count(file_bytes)
        text, table_rows = extract_text_and_tables_from_pdf(file_bytes, pages_list)

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text. PDF may be a scanned image."
            )

        text_str = text.full_all if hasattr(text, 'full_all') else str(text)

        # ── Step 1: Detect all fields via LLM + table headers (1 LLM call) ────
        fields = await detect_fields_llm(text_str, table_rows)

        # ── Step 2: If same_template, generate reusable regex + table schema ──
        extractor_generated = False
        logger.info(
            "detect-fields: same_template=%s session_id=%s fields_count=%d",
            same_template, session_id, len(fields)
        )

        # Detect garbled PDF text (font encoding issues) — skip schema, force LLM
        text_is_garbled = _is_garbled_text(text_str)
        if text_is_garbled:
            logger.warning(
                "⚠️  Garbled text detected for session %s (font encoding issue) — "
                "schema generation skipped. All PDFs will use LLM extraction.",
                session_id
            )
            if session_id:
                _EXTRACTOR_STORE[session_id] = {
                    "regex": {}, "tables": [], "schema_validated": False,
                    "llm_fallback": True, "key_meta": {}, "failed_fields": set(),
                    "garbled": True,
                }

        if same_template.lower() == "true" and session_id and fields and not text_is_garbled:
            # Build set of table column headers (from pdfplumber tables)
            _table_col_headers: set[str] = set()
            if table_rows:
                for _row in table_rows:
                    if _is_header_row([c.strip() if c else "" for c in _row]):
                        for _cell in _row:
                            if _cell and _cell.strip():
                                _table_col_headers.add(_cell.strip().lower())

            # ── Detect table column fields using x-position extraction ────────
            # For each page, extract column→value map from PDF coordinates.
            # Fields whose names match a detected column are flagged as table fields
            # and will be extracted positionally, not by regex.
            _page_col_maps: list[dict] = []  # one dict per page
            _table_field_map: dict[str, tuple[int, str]] = {}  # field → (page_idx, col_label)

            try:
                # Use file_bytes already read at the top of this endpoint
                _n_pages = get_pdf_page_count(file_bytes)

                for _pi in range(_n_pages):
                    _col_map = _extract_table_by_xpos(file_bytes, _pi)
                    _page_col_maps.append(_col_map)
                    if _col_map:
                        logger.info(
                            "Page %d table columns detected: %s", _pi + 1,
                            list(_col_map.keys())
                        )

                # Match each detected field to a table column
                for _f in fields:
                    if not isinstance(_f, dict) or "key" not in _f:
                        continue
                    _fk = _f["key"]
                    for _pi, _col_map in enumerate(_page_col_maps):
                        _val = _match_field_to_col(_fk, _col_map)
                        if _val:
                            _table_field_map[_fk] = (_pi, _val)
                            logger.info(
                                "Field [%s] → table column on page %d → value %r",
                                _fk, _pi + 1, _val
                            )
                            break
            except Exception as _e:
                logger.warning("Table field detection failed: %s", _e)
                _page_col_maps = []
                _table_field_map = {}

            _inline_table_fields: set[str] = set(_table_field_map.keys())

            # ── Detect duplicate keys and suffix them _occ1, _occ2 etc. ────────
            # When the same label appears multiple times (e.g. "State Code" for buyer
            # AND seller), we create distinct schema keys so each gets its own pattern.
            # e.g. ["State Code", "State Code"] → ["State Code_occ1", "State Code_occ2"]
            raw_field_keys = [
                f["key"] for f in fields
                if isinstance(f, dict) and "key" in f
                and not re.match(r'^.+_\d+$', f["key"])
                and f["key"].lower().strip() not in _table_col_headers
                and f["key"] not in _inline_table_fields  # column-position extracted

            ]
            key_count: dict[str, int] = {}
            for k in raw_field_keys:
                key_count[k] = key_count.get(k, 0) + 1

            key_seen: dict[str, int] = {}
            text_field_keys = []
            # Maps schema key → (original key, occurrence index) for later remapping
            schema_key_meta: dict[str, tuple[str, int]] = {}
            for k in raw_field_keys:
                key_seen[k] = key_seen.get(k, 0) + 1
                if key_count[k] > 1:
                    schema_key = f"{k}_occ{key_seen[k]}"
                    schema_key_meta[schema_key] = (k, key_seen[k])
                    if key_seen[k] == 1:
                        logger.info(
                            "Duplicate key detected: \"%s\" appears %d times — "
                            "will generate separate patterns (_occ1, _occ2, ...)",
                            k, key_count[k]
                        )
                else:
                    schema_key = k
                    schema_key_meta[schema_key] = (k, 1)
                text_field_keys.append(schema_key)

            try:
                table_schema = build_table_schema(table_rows)

                # LLM output for PDF 1 — remap to suffixed keys so validation works
                # For duplicate keys, LLM returns [{key: "State Code", value: "09"}, ...]
                # We need to split these into _occ1 → first value, _occ2 → second value
                _raw_llm_output: dict[str, list] = {}
                for f in fields:
                    if not isinstance(f, dict) or "key" not in f:
                        continue
                    k = f["key"]
                    _raw_llm_output.setdefault(k, []).append(f.get("value", "N/A"))

                llm_output: dict[str, str] = {}
                _occ_seen: dict[str, int] = {}
                for schema_key, (orig_key, occ_idx) in schema_key_meta.items():
                    vals = _raw_llm_output.get(orig_key, ["N/A"])
                    llm_output[schema_key] = vals[occ_idx - 1] if occ_idx - 1 < len(vals) else "N/A"

                # ── Self-validating schema: generate → test → fix (up to 2 retries) ──
                regex_schema     = {}
                schema_validated = False
                last_mismatch    = ""

                for attempt in range(3):
                    # Generate or fix schema
                    if attempt == 0:
                        regex_schema = await generate_schema_llm(
                            text_str, text_field_keys,
                            left_text=text.left if hasattr(text, "left") else "",
                            right_text=text.right if hasattr(text, "right") else ""
                        ) if text_field_keys else {}
                    else:
                        # Build retry prompt with exact PDF lines for each mismatch
                        regex_schema = await _fix_schema_llm(
                            regex_schema, last_mismatch, text_str,
                            left_text=text.left if hasattr(text, "left") else "",
                            right_text=text.right if hasattr(text, "right") else ""
                        )

                    # Test schema against PDF 1 (compare vs LLM ground truth)
                    schema_result = _run_schema(
                        regex_schema, text_str, text_field_keys,
                        table_rows=table_rows, table_schema=table_schema,
                        left_text=text.left if hasattr(text, "left") else ""
                    )

                    # ── Per-field pass/fail evaluation ───────────────────────
                    text_lines = [l.strip() for l in text_str.split("\n") if l.strip()]
                    passed     = []   # fields where schema matched LLM output
                    mismatches = []   # fields where schema was wrong

                    for fk in text_field_keys:
                        llm_val    = str(llm_output.get(fk, "N/A")).strip()
                        schema_val = str(schema_result.get(fk, "N/A")).strip()

                        if llm_val in ("N/A", ""):
                            # LLM itself couldn't find it — skip from scoring
                            continue

                        if schema_val != "N/A" and llm_val.lower() in schema_val.lower():
                            passed.append(f'  ✅ "{fk}": schema="{schema_val}"')
                        else:
                            ctx = next(
                                (ln for ln in text_lines
                                 if llm_val.lower() in ln.lower() or fk.lower() in ln.lower()),
                                "(label/value not on a single line)"
                            )
                            mismatches.append(
                                f'  ❌ "{fk}": want="{llm_val}" got="{schema_val}" | PDF line: {ctx!r}'
                            )

                    total_scored = len(passed) + len(mismatches)
                    match_pct    = len(passed) / max(total_scored, 1)

                    # ── Log full per-field breakdown ──────────────────────────
                    logger.info(
                        "Schema validation attempt %d: %d/%d fields correct (%.0f%%)",
                        attempt + 1, len(passed), total_scored, match_pct * 100
                    )
                    if passed:
                        logger.info(
                            "Schema attempt %d — PASSED fields (regex matched LLM output):\n%s",
                            attempt + 1, "\n".join(passed)
                        )
                    if mismatches:
                        logger.warning(
                            "Schema attempt %d — FAILED fields (will trigger retry/fallback):\n%s",
                            attempt + 1, "\n".join(mismatches)
                        )

                    if match_pct >= 0.85:
                        schema_validated = True
                        logger.info(
                            "✅ Schema ACCEPTED on attempt %d (%d/%d fields correct) "
                            "— will use regex for all remaining PDFs",
                            attempt + 1, len(passed), total_scored
                        )
                        break

                    # Keep clean format: "FieldName": want="X" got="Y" | PDF line: '...'
                    last_mismatch = "\n".join(
                        re.sub(r'^\s*[❌✅]\s*', '', m).strip()
                        for m in mismatches
                    )
                    logger.info(
                        "last_mismatch passed to retry:\n%s", last_mismatch
                    )
                    logger.warning(
                        "Schema attempt %d REJECTED (%.0f%% < 85%% threshold) — %s",
                        attempt + 1, match_pct * 100,
                        "retrying with fix prompt" if attempt < 2 else "switching to LLM fallback"
                    )

                # Collect failed fields (original key names, not _occ suffixed)
                # These are schema keys that failed on the last attempt
                failed_schema_keys = set(
                    re.sub(r'^(.+)_occ\d+$', r'\1', m.strip().lstrip('"').split('"')[0])
                    for m in mismatches
                    if m.strip().startswith('"') or m.strip().startswith('❌')
                )
                # Also include original key names via key_meta
                failed_orig_keys = set()
                for schema_key in failed_schema_keys:
                    if schema_key in schema_key_meta:
                        failed_orig_keys.add(schema_key_meta[schema_key][0])
                    else:
                        failed_orig_keys.add(schema_key)

                if not schema_validated:
                    logger.warning(
                        "⚠️  Schema FAILED after 3 attempts for session %s\n"
                        "    Failed fields: %s\n"
                        "    If user selects any of these → LLM. Otherwise → regex.",
                        session_id, sorted(failed_orig_keys)
                    )
                else:
                    if failed_orig_keys:
                        logger.info(
                            "Schema ACCEPTED but %d fields had issues: %s\n"
                            "    If user selects any of these → LLM. Otherwise → regex.",
                            len(failed_orig_keys), sorted(failed_orig_keys)
                        )

                _EXTRACTOR_STORE[session_id] = {
                    "regex":               regex_schema,
                    "tables":              table_schema,
                    "schema_validated":    schema_validated,
                    "llm_fallback":        not schema_validated,
                    "key_meta":            schema_key_meta,
                    "failed_fields":       failed_orig_keys,
                    "table_field_map":     _table_field_map,

                }
                extractor_generated = True
                logger.info(
                    "Session %s: schema_validated=%s regex=%d table=%d",
                    session_id, schema_validated, len(regex_schema), len(table_schema)
                )

            except Exception as e:
                logger.warning("Schema generation failed (will fall back to LLM per PDF): %s", e)

        return {
            "fields":               fields,
            "raw_text":             text_str,
            "total_fields":         len(fields),
            "page_count":           page_count,
            "extractor_ready":      extractor_generated,
        }

    except json.JSONDecodeError as e:
        logger.error("detect-fields JSON decode error: %s", e)
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON. Try again.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("detect-fields unhandled error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /api/extract-fields ──────────────────────────────────────────────────
# Uses LLM or Regex depending on same_template flag

@app.post("/api/extract-fields")
async def extract_fields(
    file:          UploadFile = File(...),
    fields:        str        = Form("[]"),
    same_template: str        = Form("true"),
    pages:         str        = Form("[]"),
    session_id:    str        = Form(""),        # must match the detect-fields call
):
    try:
        file_bytes       = await file.read()
        pages_list       = json.loads(pages) if pages and pages != "[]" else None
        text, table_rows = extract_text_and_tables_from_pdf(file_bytes, pages_list)
        fields_list      = json.loads(fields)
        use_same         = same_template.lower() == "true"

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        if not fields_list:
            raise HTTPException(status_code=400, detail="No fields provided.")

        text_str = text.full_all if hasattr(text, 'full_all') else str(text)

        stored = _EXTRACTOR_STORE.get(session_id) if session_id else None

        logger.info(
            "extract-fields: session=%s use_same=%s stored_type=%s keys_in_store=%s",
            session_id, use_same,
            type(stored).__name__,
            list(stored.keys()) if isinstance(stored, dict) else "N/A"
        )

        if isinstance(stored, dict) and "regex" in stored:
            regex_schema  = stored["regex"]
            table_schema  = stored.get("tables", [])
            llm_fallback         = stored.get("llm_fallback", False)
            key_meta             = stored.get("key_meta", {})
            failed_fields        = stored.get("failed_fields", set())
            text_garbled         = stored.get("garbled", False)
            table_field_map      = stored.get("table_field_map", {})  # field→(page_idx, val_pdf1)
            if text_garbled:
                use_same = False
                logger.info(
                    "⚠️  Garbled PDF session — extracting [%s] via LLM (font encoding issue)",
                    file.filename
                )
        elif isinstance(stored, dict):
            regex_schema  = stored
            table_schema  = []
            llm_fallback  = False
            key_meta      = {}
            failed_fields = set()
        else:
            regex_schema  = {}
            table_schema  = []
            llm_fallback  = False
            key_meta      = {}
            failed_fields = set()

        logger.info(
            "extract-fields: regex_schema=%d patterns, table_schema=%d tables, "
            "llm_fallback=%s, failed_fields=%s",
            len(regex_schema), len(table_schema), llm_fallback, sorted(failed_fields)
        )

        # ── Smart fallback: field-level check always takes priority ────────────
        # Runs regardless of global llm_fallback value.
        # If user selected ONLY fields that passed schema → use regex.
        # If user selected ANY field that failed schema  → use LLM.
        if failed_fields:
            requested_set = set(fields_list)
            overlap = requested_set & failed_fields
            if overlap:
                use_same = False
                logger.warning(
                    "⚡ LLM fallback for [%s] — %d selected field(s) failed schema: %s",
                    file.filename, len(overlap), sorted(overlap)
                )
            else:
                use_same = True
                logger.info(
                    "✅ Regex for [%s] — all selected field(s) passed schema "
                    "(failed fields not selected: %s)",
                    file.filename, sorted(failed_fields)
                )
        elif llm_fallback:
            # No field-level info available, global flag says LLM
            use_same = False
            logger.info(
                "⚡ LLM fallback for [%s] — schema failed globally",
                file.filename
            )

        if use_same and (regex_schema or table_schema):
            # ── PATH A: Regex + Table schema (zero LLM calls) ─────────────────
            # Run schema using the internal schema keys (may include _occ1/_occ2 variants)
            # fields_list from frontend uses original keys — build internal key list
            internal_keys = list(regex_schema.keys()) + [
                k for k in fields_list if k not in regex_schema
            ]
            extracted_internal = _run_schema(
                regex_schema, text_str, internal_keys,
                table_rows=table_rows,
                table_schema=table_schema,
                left_text=text.left if hasattr(text, 'left') else "",
            )

            # ── Extract table column fields by x-position for this PDF ────────
            if table_field_map:
                for _fk, (_pi, _pdf1_val) in table_field_map.items():
                    _col_map = _extract_table_by_xpos(file_bytes, _pi)
                    _val = _match_field_to_col(_fk, _col_map)
                    if _val:
                        extracted_internal[_fk] = _val
                        logger.info(
                            "Table column extracted: [%s] page %d → %r", _fk, _pi + 1, _val
                        )
                    else:
                        logger.info(
                            "Table column [%s] not found on page %d of this PDF", _fk, _pi + 1
                        )

            # Remap _occ schema keys back to original field names expected by frontend
            # key_meta: {"State Code_occ1": ("State Code", 1), "State Code_occ2": ("State Code", 2)}
            # Frontend sends originals: ["State Code"] → backend must return {"State Code": val_of_occ1}
            # The _occ2 value is returned as "State Code__2" so frontend Nth-occurrence logic can use it
            occ_vals: dict[str, list] = {}  # orig_key → [val_occ1, val_occ2, ...]
            for schema_key, (orig_key, occ_idx) in key_meta.items():
                val = extracted_internal.get(schema_key, "N/A")
                bucket = occ_vals.setdefault(orig_key, [])
                # Insert at correct position (occ_idx is 1-based)
                while len(bucket) < occ_idx:
                    bucket.append("N/A")
                bucket[occ_idx - 1] = val

            extracted = dict(extracted_internal)  # start with all values
            # Replace _occ keys with originals
            for schema_key in list(extracted.keys()):
                if schema_key in key_meta:
                    del extracted[schema_key]
            for orig_key, vals in occ_vals.items():
                extracted[orig_key] = vals[0] if vals else "N/A"
                # Extra occurrences stored with double-underscore suffix for frontend
                for i, v in enumerate(vals[1:], start=2):
                    extracted[f"{orig_key}__{i}"] = v

            if key_meta:
                logger.info(
                    "Remapped %d _occ schema keys back to original field names: %s",
                    len(key_meta),
                    {sk: ok for sk, (ok, _) in key_meta.items()}
                )

            # Post-process: if any value looks like a table header row
            # (contains 3+ column keywords), it means the regex matched the
            # header instead of a data cell — replace with N/A and re-extract
            # from table_rows directly
            TABLE_COL_KEYWORDS = {"cgst","sgst","igst","total","amount","rate","utgst","tax","value","invoice"}
            for field, val in list(extracted.items()):
                if val and val != "N/A":
                    val_lower = val.lower()
                    hits = sum(1 for kw in TABLE_COL_KEYWORDS if kw in val_lower)
                    if hits >= 3:
                        # Looks like a header row — try to get value from table directly
                        extracted[field] = _get_field_from_table(field, table_rows) or "N/A"
            mode = "generated_extractor"

            # Safety net: if >80% N/A on TEXT fields only (exclude table _N fields
            # which legitimately may be N/A if table structure differs), fall back to LLM
            text_fields  = [f for f in fields_list if not re.match(r'^.+_\d+$', f) and not re.match(r'^.+__\d+$', f)]
            na_text      = sum(1 for f in text_fields if extracted.get(f) == "N/A")
            na_threshold = len(text_fields) * 0.4 if text_fields else 0  # 40% threshold — trigger LLM sooner

            logger.info(
                "extract-fields PATH A: %d/%d text fields are N/A (threshold %.0f)",
                na_text, len(text_fields), na_threshold
            )

            if text_fields and na_text > na_threshold:
                logger.warning(
                    "Schema returned %d/%d text-field N/As for session %s — LLM fallback",
                    na_text, len(text_fields), session_id
                )
                extracted = await extract_fields_llm(text_str, fields_list)
                # Still merge table line items even in fallback
                line_items = _apply_table_schema(table_rows, table_schema) if table_schema \
                             else extract_line_items_from_tables(table_rows)
                for f in fields_list:
                    if re.match(r'^.+_\d+$', f) and (extracted.get(f, "N/A") == "N/A"):
                        extracted[f] = line_items.get(f, "N/A")
                mode = "llm_fallback"
        else:
            # ── PATH B: LLM per PDF + pure-Python table extraction ────────────
            extracted  = await extract_fields_llm(text_str, fields_list)
            line_items = extract_line_items_from_tables(table_rows)
            for f in fields_list:
                if re.match(r'^.+_\d+$', f) and (extracted.get(f, "N/A") == "N/A"):
                    extracted[f] = line_items.get(f, "N/A")
            mode = "llm"

        # Case-insensitive key remapping
        lower_map = {k.lower().strip(): v for k, v in extracted.items()}
        remapped = {}
        for f in fields_list:
            if f in extracted:
                remapped[f] = extracted[f]
            elif f.lower().strip() in lower_map:
                remapped[f] = lower_map[f.lower().strip()]
            else:
                remapped[f] = "N/A"
        extracted = remapped

        return {"extracted": extracted, "mode": mode, "raw_text": text_str}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON. Try again.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DELETE /api/session/{session_id}  (called by frontend when batch is done) ─

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Drop the generated extractor from memory. Frontend calls this after all
    PDFs in a batch are done -- keeps the store from growing indefinitely."""
    dropped = _EXTRACTOR_STORE.pop(session_id, None)
    return {"cleared": dropped is not None, "session_id": session_id}


# ── POST /api/raw-text  (debug utility) ──────────────────────────────────────

@app.post("/api/raw-text")
async def raw_text(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(await file.read())
        return {"text": text, "length": len(text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "groq_keys_configured": len(GROQ_API_KEYS),
        "groq_key_pool": [f"...{k[-6:]}" for k in GROQ_API_KEYS],
        "pdf_library": "pdfplumber",
        "modes": ["regex+table_schema (same template)", "llm (different templates)"],
    }


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
