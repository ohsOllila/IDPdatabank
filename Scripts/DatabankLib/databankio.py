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

    fi_name = uri.split("/")[-1]

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