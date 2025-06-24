"""
@DRAFT
Network communication. Downloading files. Checking links etc.
"""

import os
import time
import socket
import urllib.error
from tqdm import tqdm
import urllib.request
import urllib.error
from urllib.parse import urlparse
import ssl
import json


import logging
logger = logging.getLogger(__name__)


import zipfile
from pathlib import Path

def get_zip_path_if_any(file_path: str) -> Path | None:
    """
    Given a file path like 'test.zip/folder/file.xtc',
    detect if it starts with a zip archive name (.zip extension).
    
    Returns:
        Path to the zip archive (e.g., Path('test.zip')) if found,
        else None.
    """
    parts = Path(file_path).parts
    for part in parts:
        if part.endswith(".zip"):
            return Path(part)
    return None

def download_resource_from_uri(
    uri: str, dest: str, override_if_exists: bool = False
) -> int:
    """
    :meta private:
    Download file resource [from uri] to given file destination using urllib

    Args:
        uri (str): file URL
        dest (str): file destination path
        override_if_exists (bool, optional): Override dest. file if exists.
                                             Defaults to False.

    Raises:
        Exception: HTTPException: An error occured during download

    Returns:
        code (int): 0 - OK, 1 - skipped, 2 - redownloaded
    """
    # TODO verify file size before skipping already existing download!

    class RetrieveProgressBar(tqdm):
        # uses tqdm.update(), see docs https://github.com/tqdm/tqdm#hooks-and-callbacks
        def update_retrieve(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    dest = Path(dest)

    # If file is inside a zip (e.g. test.zip/some/file.top)
    parts = dest.parts
    zip_index = next((i for i, p in enumerate(parts) if p.endswith(".zip")), None)

    if zip_index is not None:
        zip_path = Path(*parts[: zip_index + 1])
        file_inside_zip = str(Path(*parts[zip_index + 1:]))

        # Make sure zip is downloaded
        zip_dest_path = zip_path
        zip_url = "/".join(uri.split("/")[: -len(file_inside_zip.split("/"))])
        zip_uri = zip_url + "/" + zip_path.name

        # Only download if needed
        if not zip_dest_path.exists() or override_if_exists:
            zip_dest_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading zip archive from: {zip_uri}")
            with RetrieveProgressBar(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=zip_path.name
            ) as u:
                urllib.request.urlretrieve(zip_uri, zip_dest_path, reporthook=u.update_retrieve)
            logger.info(f"Zip file downloaded at: {zip_dest_path}")
        else:
            logger.info(f"{zip_dest_path}: zip file already exists, skipping download")

        # Extract only the needed file
        extract_path = zip_path.parent / file_inside_zip

        # If the parent is a file (e.g. test.zip), raise a clearer error or adjust path
        if extract_path.parent.is_file():
            raise FileExistsError(f"Cannot create directory {extract_path.parent} — a file with the same name exists.")

        extract_path.parent.mkdir(parents=True, exist_ok=True)


        with zipfile.ZipFile(zip_dest_path, 'r') as zf:
            if file_inside_zip not in zf.namelist():
                raise FileNotFoundError(f"{file_inside_zip} not found in {zip_dest_path}")
            with zf.open(file_inside_zip) as source, open(extract_path, 'wb') as target:
                target.write(source.read())

        logger.info(f"Extracted {file_inside_zip} to {extract_path}")
        return 0
    
    # If it's a regular file (not inside a zip)
    fi_name = uri.split("/")[-1]

    # No zip in dest, proceed normally

    # check if dest path already exists
    if not override_if_exists and os.path.isfile(dest):
        socket.setdefaulttimeout(10)  # seconds

        # compare filesize
        fi_size = urllib.request.urlopen(uri).length  # download size
        if fi_size == os.path.getsize(dest):
            logger.info(f"{dest}: file already exists, skipping")
            return 1
        else:
            logger.warning(
                f"{fi_name} filesize mismatch of local "
                f"file '{fi_name}', redownloading ..."
            )
            return 2

    # download
    socket.setdefaulttimeout(10)  # seconds

    url_size = urllib.request.urlopen(uri).length  # download size

    with RetrieveProgressBar(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=fi_name
    ) as u:
        _ = urllib.request.urlretrieve(uri, dest, reporthook=u.update_retrieve)

    # check if the file is fully downloaded
    size = os.path.getsize(dest)

    if url_size != size:
        raise Exception(f"downloaded filsize mismatch ({size}/{url_size} B)")

    return 0

def resolve_doi_url(doi: str, validate_uri: bool = True) -> str:
    """
    :meta private:
    Returns full doi link of given ressource, also checks if URL is valid.

    Args:
        doi (str): [doi] part from config
        validate_uri (bool, optional): Check if URL is valid. Defaults to True.

    Returns:
        str: full doi link
    """
    res = "https://doi.org/" + doi

    if validate_uri:
        socket.setdefaulttimeout(10)  # seconds
        _ = urllib.request.urlopen(res)
    return res


def resolve_download_file_url(
        doi: str, fi_name: str, validate_uri: bool = True,
        sleep429=5) -> str:
    """
    :meta private:
    Returns full doi link of given ressource, also checks if URL is valid.


    Steps:
    1) Resolve DOI via https://doi.org to get final domain.
    2) Check if domain is a Dataverse by querying /api/info/version.
    3) If Dataverse:
       - Try direct file DOI access
       - If fails, query dataset metadata to find file by name.
    4) If Zenodo, construct direct Zenodo file URL.
    5) Validate final URL if requested.

    Args:
        doi (str): DOI string
        fi_name (str): name of the file to resolve from source
        validate_uri (bool, optional): Check if URI exists. Defaults to True.
        sleep429 (int, optional): Sleep in seconds if 429 HTTP code returned

    Raises:
        NotImplementedError: Unsupported DOI repository
        HTTPError: HTTP Error Status Code
        URLError: Failed to reach the server

    Returns:
        str: file URI
    """

    archive_name = fi_name.split('/')[0]


    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        uri = "https://zenodo.org/record/" + zenodo_entry_number + "/files/" + archive_name

        # check if ressource exists, may throw exception
        if validate_uri:
            _validate_url(uri, sleep429, doi, fi_name)
        return uri

    # Step 1: Resolve DOI to get final URL and domain
    try:
        resolved_url = urllib.request.urlopen(f"https://doi.org/{doi}").geturl()
    except Exception as e:
        raise RuntimeError(f"Could not resolve DOI {doi}: {e}")

    domain = urlparse(resolved_url).netloc
    logger.info(f"DOI resolved to domain: {domain}")

    # Step 2: Confirm Dataverse instance via /api/info/version
    api_version_url = f"https://{domain}/api/info/version"
    try:
        # SSL context to avoid certificate issues (use cautiously)
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(api_version_url, context=ssl_context) as response:
            version_info = json.loads(response.read().decode())
            if version_info.get("status") != "OK":
                raise RuntimeError(f"Dataverse API version check failed at {api_version_url}")
    except Exception as e:
        raise NotImplementedError(f"Domain '{domain}' is not a recognized Dataverse instance: {e}")

    # Step 3a: Try direct file DOI access 
    file_uri = f"https://{domain}/api/access/datafile/:persistentId?persistentId=doi:{doi}"
    try:
        if validate_uri:
            _validate_url(file_uri, sleep429, doi, fi_name)
        return file_uri
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise  # only continue if 404

    # Step 3b: Fall back to dataset metadata lookup
    dataset_uri = f"https://{domain}/api/datasets/:persistentId/?persistentId=doi:{doi}"
    try:
        with urllib.request.urlopen(dataset_uri, context=ssl_context) as response:
            metadata = json.loads(response.read().decode())
    except Exception as e:
        raise RuntimeError(f"Could not fetch dataset metadata from {domain}: {e}")

    try:
        files = metadata['data']['latestVersion']['files']
    except KeyError:
        raise RuntimeError(f"Unexpected metadata structure from {domain}")

    file_id = None
    for f in files:
        if f['dataFile']['filename'] == archive_name:
            file_id = f['dataFile']['id']
            break
    if not file_id:
        raise FileNotFoundError(f"File '{archive_name}' not found in dataset DOI {doi}")

    uri = f"https://{domain}/api/access/datafile/{file_id}"
    if validate_uri:
        _validate_url(uri, sleep429, doi, fi_name)

    return uri


def _validate_url(uri, sleep429, doi, fi_name):
    """Helper to validate URL existence and handle 429 rate limits with retry."""
    socket.setdefaulttimeout(10)
    try:
        urllib.request.urlopen(uri, timeout=10)
    except TimeoutError:
        raise RuntimeError(f"Cannot open {uri}. Timeout error.")
    except urllib.error.HTTPError as hte:
        if hte.code == 429:
            if sleep429 / 5 > 10:
                raise TimeoutError("Too many retries for HTTP 429 rate limit.")
            logger.warning(f"HTTP 429 from {uri}. Sleeping {sleep429} seconds and retrying.")
            time.sleep(sleep429)
            # Recursive retry
            return _validate_url(uri, sleep429 + 5, doi, fi_name)
        else:
            raise
