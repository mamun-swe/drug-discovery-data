import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
import gzip
import shutil

# Base FTP directory
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/"

# Local directories
ZIP_DIR = "pubchem_data/zips"
CSV_DIR = "pubchem_data/csvs"

# Create directories
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Step 1: Scrape the directory listing for .zip files
print("Fetching list of .zip files...")
response = requests.get(BASE_URL)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
zip_links = [a["href"] for a in soup.find_all("a") if a["href"].endswith(".zip")]

print(f"Found {len(zip_links)} zip files.")

# Step 2: Download each .zip file
for link in zip_links:
    zip_url = BASE_URL + link
    local_zip_path = os.path.join(ZIP_DIR, link)
    if not os.path.exists(local_zip_path):
        print(f"Downloading {link}...")
        with requests.get(zip_url, stream=True) as r:
            r.raise_for_status()
            with open(local_zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    else:
        print(f"Skipping {link} (already downloaded)")

# Step 3: Extract all zip files
for zip_name in os.listdir(ZIP_DIR):
    if zip_name.endswith(".zip"):
        zip_path = os.path.join(ZIP_DIR, zip_name)
        print(f"Extracting {zip_name}...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CSV_DIR)

# Step 4: Decompress .csv.gz files
for root, _, files in os.walk(CSV_DIR):
    for file in files:
        if file.endswith(".csv.gz"):
            gz_path = os.path.join(root, file)
            csv_path = os.path.join(root, file[:-3])  # remove .gz extension
            if not os.path.exists(csv_path):
                print(f"Decompressing {file}...")
                with gzip.open(gz_path, "rb") as f_in:
                    with open(csv_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

print("✅ All files downloaded, extracted, and decompressed!")
print(f"CSV files are in: {os.path.abspath(CSV_DIR)}")
