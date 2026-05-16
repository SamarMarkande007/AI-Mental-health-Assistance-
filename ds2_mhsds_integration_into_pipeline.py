import os
import io
import sys
import zipfile
import logging
import argparse
from datetime import datetime, date, timezone
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────
# OUTPUT PATHS — matches repo structure
# ─────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join("outputs", "w4_ds2_MHSDS_integration_into_pipeline")
MASTER_FILE = os.path.join(OUTPUT_DIR, "mhsds_master.csv")
LOG_FILE    = os.path.join(OUTPUT_DIR, "mhsds_pipeline.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

BASE_URL = (
    "https://digital.nhs.uk/data-and-information/publications/statistical/"
    "mental-health-services-monthly-statistics/performance-{month}-{year}"
)

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

# Add new monthly URLs here as NHS England publishes them.
# How to get the URL:
#   1. Visit the NHS publication page for that month
#   2. Right-click "MHSDS Data File" download button
#   3. Copy link address and paste below
KNOWN_URLS = {
    "2026-02": "https://files.digital.nhs.uk/CD/549410/MHSDS%20Data_FebPerf_2026.zip",
    # "2026-03": "https://files.digital.nhs.uk/...",  # add next month here
}


# ─────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    """
    Configure logger to write to both console and log file.
    Log file saved to OUTPUT_DIR.
    """
    logger = logging.getLogger("mhsds")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — writes to outputs/w4_ds2_.../mhsds_pipeline.log
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream handler — prints to console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


log = setup_logger()


# ─────────────────────────────────────────────────────────
# STEP 1 — BUILD PAGE URL
# ─────────────────────────────────────────────────────────

def build_page_url(year: int, month: int) -> str:
    """
    Build the NHS publication page URL for a given year and month.

    Args:
        year:  e.g. 2026
        month: e.g. 2 (February)

    Returns:
        Full URL string for the NHS publication page.
    """
    month_name = MONTH_NAMES[month - 1]
    url = BASE_URL.format(month=month_name, year=year)
    log.info(f"Publication page URL: {url}")
    return url


# ─────────────────────────────────────────────────────────
# STEP 2 — GET DOWNLOAD URL
# ─────────────────────────────────────────────────────────

def get_data_file_url(year: int, month: int) -> str | None:
    """
    Look up the direct ZIP download URL for a given month from
    the KNOWN_URLS dictionary.

    NHS website returns 403 Forbidden for automated requests so
    URLs are stored manually after copying from the NHS page.

    Args:
        year:  e.g. 2026
        month: e.g. 2 (February)

    Returns:
        ZIP download URL string if found, None otherwise.
    """
    key = f"{year}-{month:02d}"

    if key in KNOWN_URLS:
        url = KNOWN_URLS[key]
        log.info(f"Found URL for {key}: {url}")
        return url
    else:
        log.warning(f"No URL stored for {key}.")
        log.warning(f"Steps to fix:")
        log.warning(f"  1. Visit: {build_page_url(year, month)}")
        log.warning(f"  2. Right-click 'MHSDS Data File' download button")
        log.warning(f"  3. Copy link address")
        log.warning(f"  4. Add to KNOWN_URLS: '{key}': '<url>'")
        return None


# ─────────────────────────────────────────────────────────
# STEP 3 — DOWNLOAD & EXTRACT CSV FROM ZIP
# ─────────────────────────────────────────────────────────

def download_and_extract_csv(zip_url: str) -> pd.DataFrame | None:
    """
    Download the ZIP file from the given URL, find the largest CSV
    inside it and return as a pandas DataFrame.

    The ZIP is handled entirely in memory using io.BytesIO — no
    intermediate files are written to disk.

    Args:
        zip_url: Direct download URL for the ZIP file.

    Returns:
        DataFrame of raw CSV data, or None if download fails.
    """
    log.info(f"Downloading ZIP from: {zip_url}")
    headers = {"User-Agent": "Mozilla/5.0 (MHSDS Pipeline)"}

    try:
        resp = requests.get(zip_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Download failed: {e}")
        return None

    size_mb = len(resp.content) / 1_000_000
    log.info(f"Downloaded {size_mb:.1f} MB")

    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:
        log.error("Downloaded content is not a valid ZIP file.")
        return None

    csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
    if not csv_files:
        log.error("No CSV files found inside the ZIP archive.")
        return None

    # Pick largest CSV — most likely the main data file
    main_csv = max(csv_files, key=lambda f: z.getinfo(f).file_size)
    log.info(f"Extracting: {main_csv}")

    with z.open(main_csv) as f:
        df = pd.read_csv(f, dtype=str)

    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────
# STEP 4 — CLEAN & STANDARDISE
# ─────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw MHSDS DataFrame:
      - Strip whitespace from column names and string values
      - Parse date columns from DD/MM/YYYY text to datetime
      - Split MEASURE_VALUE into RAW, NUMERIC and SUPPRESSED columns
      - Add PIPELINE_INGESTED_AT timestamp

    Args:
        df: Raw DataFrame from download_and_extract_csv()

    Returns:
        Cleaned and standardised DataFrame.
    """
    df = df.copy()

    # Strip whitespace from column names and all string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Parse date columns — NHS uses DD/MM/YYYY format
    for date_col in ["REPORTING_PERIOD_START", "REPORTING_PERIOD_END"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(
                df[date_col], dayfirst=True, errors="coerce"
            )

    # Handle suppressed values (*)
    # NHS suppresses small counts to protect patient privacy
    if "MEASURE_VALUE" in df.columns:
        df["MEASURE_VALUE_RAW"]     = df["MEASURE_VALUE"]
        df["MEASURE_VALUE_NUMERIC"] = pd.to_numeric(
            df["MEASURE_VALUE"].replace("*", pd.NA), errors="coerce"
        )
        df["SUPPRESSED"] = df["MEASURE_VALUE"] == "*"
        df.drop(columns=["MEASURE_VALUE"], inplace=True)

    # Add pipeline ingestion timestamp
    df["PIPELINE_INGESTED_AT"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log.info(
        f"Cleaned: {len(df):,} rows | "
        f"Suppressed: {df['SUPPRESSED'].sum():,} rows"
    )
    return df


# ─────────────────────────────────────────────────────────
# STEP 5 — DEDUPLICATION CHECK
# ─────────────────────────────────────────────────────────

def already_loaded(source_key: str) -> bool:
    """
    Check if data from this source file is already in the master CSV.

    Uses SOURCE_FILE column as the unique identifier per monthly file.
    This correctly handles NHS files that contain revised data from
    multiple reporting periods.

    Args:
        source_key: e.g. 'performance-2026-02'

    Returns:
        True if already loaded (skip), False if new (load it).
    """
    if not os.path.exists(MASTER_FILE):
        log.info("Master file does not exist yet — first load.")
        return False

    existing = pd.read_csv(
        MASTER_FILE,
        usecols=["SOURCE_FILE"],
        dtype=str,
    )

    is_loaded = source_key in existing["SOURCE_FILE"].values

    if is_loaded:
        log.info(f"'{source_key}' already exists in master — skipping.")
    else:
        log.info(f"'{source_key}' not found in master — ready to append.")

    return is_loaded


# ─────────────────────────────────────────────────────────
# STEP 6 — APPEND TO MASTER
# ─────────────────────────────────────────────────────────

def append_to_master(df: pd.DataFrame, source_key: str) -> None:
    """
    Append cleaned DataFrame to the master CSV file in OUTPUT_DIR.
    Tags every row with SOURCE_FILE for full audit traceability.

    Args:
        df:         Cleaned DataFrame to append.
        source_key: e.g. 'performance-2026-02'
    """
    df = df.copy()
    df["SOURCE_FILE"] = source_key

    if os.path.exists(MASTER_FILE):
        existing = pd.read_csv(MASTER_FILE, dtype=str)
        before   = len(existing)
        existing = existing[existing["SOURCE_FILE"] != source_key]
        removed  = before - len(existing)

        if removed > 0:
            log.info(f"Removed {removed:,} old rows from '{source_key}' before re-appending.")

        existing.to_csv(MASTER_FILE, index=False)
        df.to_csv(MASTER_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_FILE, index=False)

    size_mb = os.path.getsize(MASTER_FILE) / 1_000_000
    log.info(f"Appended {len(df):,} rows → master now {size_mb:.1f} MB")
    log.info(f"Saved to: {MASTER_FILE}")


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────

def run_pipeline(year: int, month: int) -> bool:
    """
    Run the full end-to-end pipeline for one month.

    Args:
        year:  e.g. 2026
        month: e.g. 2 (February)

    Returns:
        True if data was appended, False if skipped or failed.
    """
    log.info("=" * 55)
    log.info(f"  Running pipeline for {MONTH_NAMES[month-1].title()} {year}")
    log.info("=" * 55)

    build_page_url(year, month)

    zip_url = get_data_file_url(year, month)
    if not zip_url:
        log.error("No download URL found. Add it to KNOWN_URLS.")
        return False

    df_raw = download_and_extract_csv(zip_url)
    if df_raw is None:
        log.error("Download or extraction failed.")
        return False

    df = clean_dataframe(df_raw)

    source_key = f"performance-{year}-{month:02d}"
    log.info(f"Source key: {source_key}")

    if already_loaded(source_key):
        log.info(f"{MONTH_NAMES[month-1].title()} {year} already loaded — skipping.")
        return False

    append_to_master(df, source_key)
    log.info(f"✅ {MONTH_NAMES[month-1].title()} {year} successfully added to master!")
    return True


# ─────────────────────────────────────────────────────────
# AUTO-DETECT LATEST MONTH
# ─────────────────────────────────────────────────────────

def run_latest() -> None:
    """
    Automatically detect the latest published NHS month and run
    the pipeline for it.
    """
    today = date.today()
    log.info(f"Today's date: {today}")
    log.info("Checking for latest available MHSDS data...")

    for offset in [1, 2, 3]:
        target     = today - relativedelta(months=offset)
        month_name = MONTH_NAMES[target.month - 1].title()
        key        = f"{target.year}-{target.month:02d}"

        log.info(f"Trying {month_name} {target.year}...")

        if key not in KNOWN_URLS:
            log.warning(f"No URL in KNOWN_URLS for {key} — skipping.")
            continue

        success = run_pipeline(target.year, target.month)
        if success:
            log.info(f"Latest data loaded: {month_name} {target.year}")
            return

    log.info("Summary: No new data appended.")
    log.info("Either all recent months are already loaded,")
    log.info("or their URLs are missing from KNOWN_URLS.")


# ─────────────────────────────────────────────────────────
# MASTER DATASET SUMMARY
# ─────────────────────────────────────────────────────────

def inspect_master() -> None:
    """Print a summary of the current master dataset."""
    if not os.path.exists(MASTER_FILE):
        log.warning("Master file not found. Run the pipeline first.")
        return

    master = pd.read_csv(MASTER_FILE, dtype=str)
    master["REPORTING_PERIOD_START"] = pd.to_datetime(
        master["REPORTING_PERIOD_START"], errors="coerce"
    )

    log.info("=" * 55)
    log.info("  MHSDS MASTER DATASET SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total rows:      {len(master):,}")
    log.info(f"  Total columns:   {len(master.columns)}")
    log.info(f"  File size:       {os.path.getsize(MASTER_FILE)/1e6:.1f} MB")
    log.info(f"  Months loaded:   {master['SOURCE_FILE'].nunique()}")
    log.info(f"  Unique measures: {master['MEASURE_ID'].nunique()}")
    log.info(f"  Unique orgs:     {master['PRIMARY_LEVEL'].nunique()}")
    log.info(f"  Date range:      {master['REPORTING_PERIOD_START'].min().date()} → {master['REPORTING_PERIOD_START'].max().date()}")
    log.info("")
    log.info("  Loaded source files:")
    for src, count in master["SOURCE_FILE"].value_counts().items():
        log.info(f"    {src}    {count:,} rows")
    log.info("=" * 55)
    log.info(f"  Output directory: {OUTPUT_DIR}")
    log.info(f"  Master file:      {MASTER_FILE}")
    log.info(f"  Log file:         {LOG_FILE}")
    log.info("=" * 55)

    # This final print is picked up by run_all.py summary
    print(f"MHSDS pipeline complete → {MASTER_FILE} ({os.path.getsize(MASTER_FILE)/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MHSDS Monthly Data Pipeline — NHS England"
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Target month in YYYY-MM format e.g. --month 2026-02. "
             "Defaults to auto-detect latest.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print master dataset summary and exit.",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_master()
    elif args.month:
        try:
            dt = datetime.strptime(args.month, "%Y-%m")
            run_pipeline(dt.year, dt.month)
            inspect_master()
        except ValueError:
            log.error("Invalid format. Use YYYY-MM e.g. --month 2026-02")
            sys.exit(1)
    else:
        run_latest()
        inspect_master()