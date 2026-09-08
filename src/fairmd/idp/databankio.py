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
from urllib.parse import urlparse
import ssl
import json
import libarchive # for archive extraction
import tempfile  # for temporary folders for nested archive extraction

import logging
logger = logging.getLogger(__name__)

from pathlib import Path

# Supported archive extensions. Note that I only tested on .zip
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")

def extract_file_from_archive(archive_path: Path, target_path: str, dest_dir: Path):
    """
    Extract a single file from an archive and save it with a flattened filename.

    Parameters:
    archive_path : Path
        The path to the archive file (e.g., .zip, .tar.gz) to extract from.
    target_path : str
        The relative path of the file inside the archive to extract.
    dest_dir : Path
        The destination directory to which the file should be written. The directory
        will be created if it does not exist.

    Returns:
    None

    Raises:
    FileNotFoundError
        If the specified `target_path` is not found in the archive.
    """
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            if entry.pathname == target_path:
                dest_dir.mkdir(parents=True, exist_ok=True)
                output_file = dest_dir / Path(target_path).name
                with open(output_file, 'wb') as f:
                    for block in entry.get_blocks():
                        f.write(block)
                return
    raise FileNotFoundError(f"'{target_path}' not found in archive {archive_path}")


    
def find_archive_in_path(file_path: Path) -> tuple[Path | None, str | None]:
    """
    Given a path potentially containing nested archives,
    find the first archive part and the relative path inside it.

    This is necessary because I first locate the outermost archive, and then process inner archives step-by-step

    E.g., given 'data.zip/data/inner.zip/file.txt', it returns:
        (Path('data.zip'), 'data/inner.zip/file.txt')

    Returns:
        Tuple of (archive_path, file_inside) or (None, None) if no archive found.
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part.endswith(ARCHIVE_EXTENSIONS):
            archive_path = Path(*parts[:i + 1])
            file_inside = str(Path(*parts[i + 1:]))
            return archive_path, file_inside
    return None, None

def extract_nested_file_from_archives(archive_path: Path, nested_path: str, dest_path: Path):
    """
    Recursively extract a file nested inside potentially multiple layers
    of archives.

    For example, if nested_path is 'data/inner.zip/file.txt', it:
    1) Extracts 'inner.zip' from 'archive_path' to a temp location,
    2) Extracts 'file.txt' from 'inner.zip' to dest_dir.

    Parameters:
        archive_path: Path to the outer archive file.
        nested_path: Path inside the archive(s) pointing to the target file.
        dest_dir: Directory where the final extracted file will be saved.

    Raises:
        FileNotFoundError: If any nested archive or target file is not found.
    """
    parts = Path(nested_path).parts
    current_archive = archive_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Iterate through parts, extract nested archives step-by-step
        last_archive_idx = -1
        for i in range(len(parts) - 1):
            nested_archive_name = str(Path(*parts[:i + 1]))

            if not nested_archive_name.endswith(ARCHIVE_EXTENSIONS): # Skip if not an archive file extension
                continue
            
            # Temporary file to hold the nested archive extracted from current_archive
            tmp_nested_archive_path = Path(tmp_dir) / f"nested_{i}{Path(nested_archive_name).suffix}"
            found = False
            with libarchive.file_reader(str(current_archive)) as entries:
                for entry in entries:
                    if entry.pathname == nested_archive_name:
                        found = True
                        with open(tmp_nested_archive_path, 'wb') as f:
                            for block in entry.get_blocks():
                                f.write(block)
                        break
            if not found:
                raise FileNotFoundError(f"Nested archive '{nested_archive_name}' not found in archive {current_archive}")

            current_archive = tmp_nested_archive_path
            last_archive_idx = i # keep track of where the last archive ends, so you extract the full correct remaining path

        # Extract the final file (flattened) look for full remaining inner path
        final_inner_path = str(Path(*parts[last_archive_idx + 1:]))
        extract_file_from_archive(current_archive, final_inner_path, dest_path)


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
    archive_path, file_inside = find_archive_in_path(dest)

    if archive_path is not None:
        archive_uri = uri
        if not archive_path.exists() or override_if_exists:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading archive from: {archive_uri}")
            with RetrieveProgressBar(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=archive_path.name
            ) as u:
                urllib.request.urlretrieve(archive_uri, archive_path, reporthook=u.update_retrieve)
            logger.info(f"Archive downloaded at: {archive_path}")
        else:
            logger.info(f"{archive_path}: archive already exists, skipping download")

        # Handle nested archives extraction recursively
        extract_nested_file_from_archives(archive_path, file_inside, archive_path.parent)

        logger.info(f"Extracted {file_inside} to {dest}")
        return 0
    
    # If it's a regular file (not inside a zip)
    fi_name = uri.split("/")[-1]

    # No compressed files in dest, proceed normally

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
