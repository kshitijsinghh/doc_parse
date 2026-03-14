import os
import re
import json
import logging
import traceback
import httpx
import pdfplumber
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocParse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id → generated extractor Python code string.
# Populated by /api/detect-fields, consumed by /api/extract-fields.
# Never written to disk. Dropped automatically when the process ends
# or when the session explicitly clears it.
_EXTRACTOR_STORE: dict[str, str] = {}


def _run_schema(schema: dict, flat_text: str, fields: list[str]) -> dict:
    """
    Apply a regex schema to extract fields from flattened PDF text.

    schema = { "Field Name": "regex_pattern", ... }
    Patterns run with re.IGNORECASE | re.MULTILINE — NOT re.DOTALL.
    This keeps [^\n]+ / .+ matching on one line only.

    Patterns with TWO capture groups (e.g. Period split "Jan'202\\n6") have
    their groups auto-concatenated: result = group1 + group2.

    Pre-processes text: strips leading/trailing whitespace from every line.
    """
    result = {}
    clean_text = "\n".join(line.strip() for line in flat_text.split("\n"))

    for field in fields:
        pattern = schema.get(field)
        if not pattern:
            result[field] = "N/A"
            continue
        try:
            m = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE)
            if m:
                if m.lastindex and m.lastindex >= 2:
                    # Multi-group pattern — concatenate all non-None groups.
                    # Convention for row-data patterns that include a row identifier
                    # as group 1 (e.g. "Description" = "GSSJAN26PO"):
                    # if the field name matches the first captured group exactly
                    # (case-insensitive), return only group 1.
                    # Otherwise concatenate from group 2 onwards (e.g. Period = Jan'2026).
                    groups = [g for g in m.groups() if g is not None]
                    first_group = groups[0].strip() if groups else ""
                    field_lower = field.lower().strip()
                    # If first group IS the field value (e.g. Description captures "GSSJAN26PO")
                    if first_group.lower() == field_lower or len(groups) == 1:
                        result[field] = first_group
                    else:
                        # Concatenate remaining groups (skip identifier in group1)
                        result[field] = "".join(groups[1:]).strip()
                else:
                    result[field] = m.group(1).strip()
            else:
                result[field] = "N/A"
        except re.error as e:
            logger.warning("Bad regex for field %r: %s — pattern: %r", field, e, pattern)
            result[field] = "N/A"
    return result


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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")

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

    data = response.json()
    if response.status_code != 200:
        error_msg = data.get("error", {}).get("message", f"Groq API returned HTTP {response.status_code}")
        raise HTTPException(status_code=500, detail=f"Groq error: {error_msg}")
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"].get("message", "Groq API error"))
    if not data.get("choices"):
        raise HTTPException(status_code=500, detail=f"Groq returned no choices. Response: {data}")

    message = data["choices"][0]["message"]
    # Some reasoning models return content as None/empty — fall back to reasoning field
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning") or ""
    if not content.strip():
        raise HTTPException(status_code=500, detail=f"Groq returned empty content. Full message: {message}")
    return content


async def detect_fields_llm(text: str) -> list:
    """Use Groq LLM to detect all key-value pairs from document text."""
    raw = await groq_chat(f"""Extract every labeled field from the document below.

CRITICAL RULES — follow exactly:
1. The "key" MUST be the EXACT label text as it appears in the document — do NOT rename, rephrase, or translate it.
   - If the document says "Invoice No." → key must be "Invoice No." (not "Invoice Number")
   - If the document says "GSTIN No:" → key must be "GSTIN No" (not "Vendor GSTIN" or "GST Number")
   - If the document says "Place of Supply or Services:" → key must be "Place of Supply or Services"
2. The "value" must be the exact value from the document.
3. For fields with no label (e.g. company name on line 2, address at the bottom), use a clear positional key like "Seller Company Name" or "Seller Address".
4. For table data (tax amounts, rates), use column header text as the key.
5. Be exhaustive — extract every field you can find including:
   - All labeled lines (Label: Value format)
   - Company names, addresses
   - All table columns (amounts, rates, periods)
   - Footer fields (PAN, GSTIN, HSN/SAC, place of supply, TDS note)

Return ONLY a valid JSON array — no markdown, no explanation:
[{{"key": "exact label from PDF", "value": "exact value from PDF"}}, ...]

DOCUMENT:
{text[:6000]}""")

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result  = json.loads(cleaned)
    return result if isinstance(result, list) else []


async def extract_fields_llm(text: str, fields: list) -> dict:
    """Use Groq LLM to extract specific fields from document text."""
    raw = await groq_chat(f"""Extract the following fields from the document below.

CRITICAL RULES:
1. Fields ending in " 1" (e.g. "Invoice Number 1") refer to the FIRST invoice/page. Extract only from that section.
2. Fields ending in " 2" (e.g. "Invoice Number 2") refer to the SECOND invoice/page. Extract only from that section.
3. A bare field like "Date" with no number means find it anywhere in the document.
4. Return the EXACT value as it appears in the document — do not add extra text.
5. If a field is genuinely not found, return "N/A".
6. For two-column lines (e.g. "Invoice Number: X   Date: Y"), extract only the value for the requested field label, not the whole line.

Fields to extract:
{chr(10).join(f"- {f}" for f in fields)}

Return ONLY a valid JSON object with no markdown or explanation:
{{"Field Name": "value", ...}}

DOCUMENT:
{text[:8000]}""")

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    result  = json.loads(cleaned)
    return result if isinstance(result, dict) else {}


async def generate_schema_llm(sample_text: str, fields: list[str]) -> dict:
    """
    Ask the LLM to generate a JSON regex schema for extracting fields.

    Instead of writing Python code (which LLMs often get wrong by hardcoding
    values), we ask the LLM to supply one regex pattern per field as JSON.
    WE run re.search() ourselves in _run_schema() — the LLM never executes code.

    CRITICAL RULES enforced by the prompt:
    - Patterns must be GENERIC (label-anchored), never hardcoded values
    - Use [^\\n]+ for single-line values (not .+ which with re.DOTALL spans lines)
    - For values split across two lines (e.g. "Jan'202\\n6"), use two capture groups
    - For fields after "TO," block, chain multiple \\n anchors to find the right line

    Returns: { "Field Name": "regex_pattern", ... }
    """
    fields_block = "\n".join(f'  "{f}"' for f in fields)
    clean_sample = "\n".join(line.strip() for line in sample_text.split("\n"))[:4000]

    raw = await groq_chat(f"""You are a regex pattern generator for PDF data extraction.

I have a flattened PDF text (one value per line, leading/trailing whitespace stripped)
and a list of fields to extract. Write ONE Python regex pattern per field.

SAMPLE PDF TEXT (exact content, line by line):
---
{clean_sample}
---

FIELDS TO EXTRACT:
{fields_block}

STRICT RULES — follow exactly or the patterns will break on other PDFs:

1. Return ONLY a valid JSON object: {{"field name": "regex_pattern", ...}}

2. Patterns run with re.IGNORECASE | re.MULTILINE. NOT re.DOTALL.
   This means "." does NOT match newlines. Use [^\\n]+ for single-line values.

3. NEVER hardcode specific values. Patterns must match the LABEL, not the value.
   CORRECT: "Invoice No\\.:\\s*(\\S+)"     ← matches any invoice number
   WRONG:   "FY26/Jan26/03486"             ← breaks on every other PDF

4. For simple "Label: Value" lines, use:  "Label Name:\\s*([^\\n]+)"
   or for word-only values:               "Label Name:\\s*(\\S+)"

5. For values that come after "TO," (buyer block), chain \\n anchors:
   - Line after TO,:      "TO,\\s*\\n([^\\n]+)"       → Buyer Company Name
   - Two lines after TO,: "TO,\\s*\\n[^\\n]+\\n([^\\n]+)"  → Buyer Address
   - City-Pincode line:   "TO,\\s*\\n[^\\n]+\\n[^\\n]+\\n([\\w][\\w\\s]+?)-\\d{{5,6}}"
   - Pincode only:        "TO,\\s*\\n[^\\n]+\\n[^\\n]+\\n[\\w][\\w\\s]+-(\\d{{5,6}})"
   - State (before INDIA):"TO,\\s*\\n[^\\n]+\\n[^\\n]+\\n[^\\n]+\\n([^\\n]+)\\nINDIA"

6. For values split across two lines (e.g. a period code "Jan'202" on one line 
   and "6" on the next), use TWO capture groups: "(Jan'202)[^\\n]*\\n(\\d)"
   The system will concatenate them automatically.

7. For the Seller Company Name (no label): "TAX INVOICE\\s*\\n([^\\n]+)"
8. For Seller Address (after "Authorised Signatory"): "Authorised Signatory\\s*\\n([^\\n]+)"
9. For fields like "GSTIN No:VALUE" (no space around colon): "GSTIN No:(\\S+)"
10. For amounts in a data row (positional): use the row identifier then count 
    space-separated number tokens:
    "^GSSJAN26PO\\s+Jan'202\\d?\\s+([\\d,]+\\.[\\d]+)"  → first amount = Value of Services
    "^GSSJAN26PO\\s+Jan'202\\d?\\s+[\\d,\\.]+\\s+([\\d,]+\\.[\\d]+)"  → second amount = CGST
    ... and so on positionally

Return ONLY the JSON — no markdown, no explanation:
{{"field name": "regex_pattern", ...}}
""")

    cleaned = raw.strip()
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("Schema LLM returned non-dict")

    # Filter out empty or placeholder-only patterns
    valid = {k: v for k, v in result.items()
             if isinstance(v, str) and v.strip() and "(" in v}
    logger.info("Schema generated: %d/%d fields have valid patterns", len(valid), len(fields))
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
        text        = extract_text_from_pdf(file_bytes, pages_list)

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text. PDF may be a scanned image."
            )

        text_str = text.full_all if hasattr(text, 'full_all') else str(text)

        # ── Step 1: Detect all fields via LLM (1 call — always) ────────────────
        fields = await detect_fields_llm(text_str)

        # ── Step 2: If same_template, generate a reusable extractor (1 call) ──
        # This extractor runs on every subsequent PDF with zero LLM calls.
        extractor_generated = False
        if same_template.lower() == "true" and session_id and fields:
            field_keys = [f["key"] for f in fields if isinstance(f, dict) and "key" in f]
            if field_keys:
                try:
                    schema = await generate_schema_llm(text_str, field_keys)
                    _EXTRACTOR_STORE[session_id] = schema   # store dict of patterns
                    extractor_generated = True
                    logger.info("Regex schema generated for session %s (%d/%d fields with patterns)",
                                session_id, len(schema), len(field_keys))
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

        schema = _EXTRACTOR_STORE.get(session_id) if session_id else None

        if use_same and isinstance(schema, dict) and schema:
            # ── PATH A: Regex schema (zero LLM calls) ─────────────────────────
            extracted = _run_schema(schema, text_str, fields_list)
            mode = "schema"

            # Safety net: if >80% N/A, fall back to LLM
            na_count = sum(1 for v in extracted.values() if v == "N/A")
            if na_count > len(fields_list) * 0.8:
                logger.warning("Schema returned %d/%d N/A for session %s — LLM fallback",
                               na_count, len(fields_list), session_id)
                extracted = await extract_fields_llm(text_str, fields_list)
                mode = "llm_fallback"
        else:
            # ── PATH B: LLM per PDF ───────────────────────────────────────────
            extracted = await extract_fields_llm(text_str, fields_list)
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
    PDFs in a batch are done — keeps the store from growing indefinitely."""
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
        "groq_key_configured": bool(os.getenv("GROQ_API_KEY")),
        "pdf_library": "pdfplumber",
        "modes": ["regex (same template)", "llm (different templates)"],
    }


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
