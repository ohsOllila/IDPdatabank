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

import logging
logger = logging.getLogger(__name__)


import zipfile
from pathlib import Path

def download_resource_from_uri(
    uri: str, dest: str, override_if_exists: bool = False
) -> int:
    """
    Download file resource [from uri] to given file destination using urllib.
    Supports nested paths like "archive.zip/file.itp".

    Returns:
        code (int): 0 - OK, 1 - skipped, 2 - redownloaded
    """
    class RetrieveProgressBar(tqdm):
        def update_retrieve(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    dest_path = Path(dest)
    dest_str = str(dest_path)

    # Determine if we are working with a nested path like zip/file.txt
    if dest_path.suffix == ".zip" or "/" not in dest_str:
        zip_name = dest_str  # downloading a top-level file (possibly a .zip)
        inner_file = None
    else:
        parts = dest_path.parts
        zip_name = str(parts[0])  # e.g. test.zip
        inner_file = str(Path(*parts[1:]))  # e.g. bilayer.top

    # Download ZIP if necessary
    zip_download_needed = not os.path.isfile(zip_name) or override_if_exists
    zip_size_expected = urllib.request.urlopen(uri).length

    if not override_if_exists and os.path.isfile(zip_name):
        local_size = os.path.getsize(zip_name)
        if local_size == zip_size_expected:
            logger.info(f"{zip_name}: file already exists, skipping download.")
            zip_download_needed = False
        else:
            logger.warning(f"{zip_name}: local file size mismatch, redownloading...")

    if zip_download_needed:
        with RetrieveProgressBar(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
            desc=os.path.basename(zip_name)
        ) as u:
            urllib.request.urlretrieve(uri, zip_name, reporthook=u.update_retrieve)

    if inner_file:
        # Extract file from downloaded zip
        if not os.path.isfile(zip_name):
            raise FileNotFoundError(f"Expected ZIP archive '{zip_name}' does not exist.")

        try:
            with zipfile.ZipFile(zip_name, 'r') as zf:
                if inner_file not in zf.namelist():
                    raise FileNotFoundError(f"'{inner_file}' not found in archive '{zip_name}'")

                # Create parent dirs if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract to temp and move to final destination
                extracted_path = zf.extract(inner_file)
                os.rename(extracted_path, dest_str)
                logger.info(f"Extracted '{inner_file}' from '{zip_name}' to '{dest_str}'")

        except zipfile.BadZipFile:
            raise RuntimeError(f"'{zip_name}' is not a valid zip file.")

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
    Resolve file URI from supported DOI with given filename

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

    # Extract only the top-level filename, e.g., "file.zip" from "file.zip/folder/file.txt"
    archive_name = fi_name.split('/')[0]
    
    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        uri = "https://zenodo.org/record/" + zenodo_entry_number + "/files/" + archive_name

        # check if ressource exists, may throw exception
        if validate_uri:
            try:
                socket.setdefaulttimeout(10)  # seconds
                _ = urllib.request.urlopen(uri, timeout=10)
            except TimeoutError:
                raise RuntimeError(f"Cannot open {uri}. Timeout error.")
            except urllib.error.HTTPError as hte:
                if hte.code == 429:
                    if sleep429/5 > 10:
                        raise TimeoutError(
                            "Too many iteration of increasing waiting time!")
                    logger.warning(f"HTTP error returned from URI: {uri}")
                    logger.warning(f"Site returns 429 code."
                                   f" Try to sleep {sleep429} seconds and repeat!")
                    time.sleep(sleep429)
                    return resolve_download_file_url(doi, fi_name, validate_uri,
                                                     sleep429=sleep429+5)
                else:
                    raise hte
        return uri
    elif "10.17617" in doi:
        uri = "https://edmond.mpg.de/api/datasets/:persistentId/?persistentId=doi:" + doi

        # did not see a direct way with the api to grab an individual file, so look through metadata to find fileid from name
        try:
            with urllib.request.urlopen(uri) as response:
                dataset_metadata = json.loads(response.read().decode())
                # Save metadata to file as a check 
                # with open("edmond_metadata.json", "w", encoding="utf-8") as out_file:
                #     json.dump(dataset_metadata, out_file, indent=2)
        
        except Exception as e:
            raise RuntimeError(f"Could not fetch metadata from Edmond: {e}")

        try: # try to look for file by supplied file nsma 
            files = dataset_metadata['data']['latestVersion']['files']
            # print("files:", files)
            file_id = None
            for f in files:
                if f['dataFile']['filename'] == archive_name:
                    file_id = f['dataFile']['id']
                    # print("file_id:\n", file_id)
                    break
        except KeyError as e:
            raise RuntimeError(f"Unexpected metadata structure: {e}")
        
        if not archive_name:
            raise FileNotFoundError(f"File '{archive_name}' not found in dataset {doi}, might be zipped")
        
        # new uri for the file
        uri = f"https://edmond.mpg.de/api/access/datafile/{file_id}"

        # check if ressource exists, may throw exception (copied from above, can likely condense)
        if validate_uri:
            try:
                socket.setdefaulttimeout(10)  # seconds
                _ = urllib.request.urlopen(uri, timeout=10)
            except TimeoutError:
                raise RuntimeError(f"Cannot open {uri}. Timeout error.")
            except urllib.error.HTTPError as hte:
                if hte.code == 429:
                    if sleep429/5 > 10:
                        raise TimeoutError(
                            "Too many iteration of increasing waiting time!")
                    logger.warning(f"HTTP error returned from URI: {uri}")
                    logger.warning(f"Site returns 429 code."
                                   f" Try to sleep {sleep429} seconds and repeat!")
                    time.sleep(sleep429)
                    return resolve_download_file_url(doi, fi_name, validate_uri,
                                                     sleep429=sleep429+5)
                else:
                    raise hte
        return uri

    else:
        raise NotImplementedError(
            "Repository not validated. Please upload the data for example to zenodo.org or edmonds"
        )