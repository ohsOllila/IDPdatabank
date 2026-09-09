"""Repository downloads and extraction of individual archive members."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import ssl
from pathlib import Path, PurePosixPath
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import libarchive
from tqdm import tqdm

logger = logging.getLogger(__name__)
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")


def validate_source_path(source):
    """Accept relative repository paths, never filesystem traversal."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("A nonempty repository filename is required")
    path = PurePosixPath(source)
    if path.is_absolute() or ".." in path.parts or "\\" in source or path.name in ("", "."):
        raise ValueError(f"Invalid repository path: {source!r}")
    return path


def find_archive_in_path(file_path):
    """Split at the first archive with a member following it."""
    parts = Path(file_path).parts
    for i, part in enumerate(parts[:-1]):
        if part.lower().endswith(ARCHIVE_EXTENSIONS):
            return Path(*parts[:i + 1]), str(Path(*parts[i + 1:]))
    return None, None


def _write_atomic(destination, blocks):
    """Keep incomplete downloads/extractions out of the cache."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            for block in blocks:
                output.write(block)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_file_from_archive(archive_path, target_path, dest_dir):
    """Extract one regular member to dest_dir using its basename."""
    target = validate_source_path(target_path)
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            if PurePosixPath(entry.pathname) == target:
                if not entry.isfile or entry.islnk:
                    raise ValueError(f"Archive member is not a regular file: {target}")
                destination = Path(dest_dir) / target.name
                _write_atomic(destination, entry.get_blocks())
                return destination
    raise FileNotFoundError(f"'{target}' not found in archive {archive_path}")


def extract_nested_file_from_archives(archive_path, nested_path, dest_path):
    """Extract arbitrary archive nesting, resetting member paths at each layer."""
    remaining = str(validate_source_path(nested_path))
    current = Path(archive_path)
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_path) as temporary:
        depth = 0
        while True:
            inner, member = find_archive_in_path(remaining)
            if inner is None:
                return extract_file_from_archive(current, remaining, dest_path)
            current = extract_file_from_archive(current, str(inner), Path(temporary) / str(depth))
            remaining = member
            depth += 1


def _download(uri, destination, override=False):
    destination = Path(destination)
    existed = destination.is_file()
    with urllib.request.urlopen(uri, timeout=10) as response:
        length = response.headers.get("Content-Length")
        expected = int(length) if length is not None else None
        already_complete = (
            existed and not override and expected is not None
            and destination.stat().st_size == expected
        )
        if already_complete:
            return 1

        def blocks():
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                yield block
            if expected is not None and received != expected:
                raise IOError(f"Downloaded size mismatch: {received}/{expected} bytes from {uri}")

        with tqdm(total=expected, unit="B", unit_scale=True, desc=destination.name) as progress:
            def tracked_blocks():
                for block in blocks():
                    progress.update(len(block))
                    yield block
            _write_atomic(destination, tracked_blocks())
    return 2 if existed else 0


def download_resource_from_uri(uri, dest, override_if_exists=False, *, source_path=None):
    """Download to dest, optionally extracting the member in source_path.

    Without source_path, legacy destinations like a.zip/folder/file.xtc are
    supported and extracted beside a.zip. Returns 0 (new), 1 (cached), 2 (replaced).
    """
    dest = Path(dest)
    if source_path is None:
        archive, member = find_archive_in_path(dest)
        if archive is None:
            return _download(uri, dest, override_if_exists)
        destination = archive.parent / Path(member).name
    else:
        source = validate_source_path(source_path)
        archive, member = find_archive_in_path(str(source))
        destination = dest
        if archive is None:
            return _download(uri, destination, override_if_exists)
        # URL identity avoids collisions between repositories and archive names.
        cache_key = hashlib.sha256(uri.encode()).hexdigest()
        archive = dest.parent / ".archives" / cache_key / archive.name
    existed = destination.is_file()
    _download(uri, archive, override_if_exists)
    # The same work directory can serve different simulations from one DOI.
    # An existing basename does not prove it came from the requested member.
    # Extract separately, then atomically publish to the requested local name.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        extracted = extract_nested_file_from_archives(archive, member, Path(temporary))
        os.replace(extracted, destination)
    return 2 if existed else 0


def prepare_file_sources(sim, file_keys):
    """Normalize file fields to local basenames; record archive locations in
    SOURCE_FILES so the file can be resolved again after DIR_WRK is gone."""
    seen = {}
    replacements = []
    for key in file_keys:
        for entry in sim.get(key) or []:
            original = entry[0]
            source = str(validate_source_path(original))
            local = validate_source_path(original).name
            if local in seen and seen[local] != source:
                raise ValueError(
                    f"Files {seen[local]!r} and {source!r} share local filename {local!r}"
                )
            seen[local] = source
            replacements.append((entry, local))
    for entry, local in replacements:
        entry[0] = local
    archived = {local: source for local, source in seen.items() if source != local}
    if archived:
        sim["SOURCE_FILES"] = archived
    return seen


def download_system_file(system, local_name, dest, override_if_exists=False):
    """Download one system file, e.g. for re-download at analysis time, resolving
    it via SOURCE_FILES if it originally came from inside an archive. Falls back
    to local_name itself as the repository filename otherwise."""
    source = (system.get("SOURCE_FILES") or {}).get(local_name, local_name)
    uri = resolve_download_file_url(system["DOI"], source)
    return download_resource_from_uri(uri, dest, override_if_exists, source_path=source)


def download_simulation_files(sim, destination, file_keys, override_if_exists=False):
    """AddData's download stage: normalize metadata and materialize local files."""
    sources = prepare_file_sources(sim, file_keys)
    for local in sources:
        logger.info("Downloading %s from %s", local, sources[local])
        download_system_file(sim, local, Path(destination) / local, override_if_exists)
    return list(sources)


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
    0) If doi is an "mddb:<project>" reference (MDDB/MDposit has no DOIs),
       resolve it separately -- see resolve_mddb_file_url.
    1) Resolve DOI via https://doi.org to get final domain.
    2) Check if domain is a Dataverse by querying /api/info/version.
    3) If Dataverse:
       - Try direct file DOI access
       - If fails, query dataset metadata to find file by name.
    4) If Zenodo, construct direct Zenodo file URL.
    5) Validate final URL if requested.

    Args:
        doi (str): DOI string, or "mddb:<project accession>" for MDDB
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

    if doi.lower().startswith("mddb:"):
        return resolve_mddb_file_url(doi.split(":", 1)[1], fi_name, validate_uri, sleep429)

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


MDDB_API_ROOT = "https://mdposit-dev.mddbr.eu/api/rest/v1"


def resolve_mddb_file_url(
        project_ref: str, fi_name: str, validate_uri: bool = True,
        sleep429=5) -> str:
    """
    :meta private:
    Resolve a download URL for one file of an MDDB/MDposit project.

    MDDB has no DOIs; projects are addressed by accession (e.g. "bsc-A0008"),
    optionally with a ".<mdNumber>" suffix selecting one of several replicas
    ("MDs") stored under the same project, e.g. "bsc-A0008.2". Files
    themselves (trajectory.xtc, topology.tpr, structure.pdb, ...) are plain,
    unarchived downloads -- no extraction is needed.

    The hub host (MDDB_API_ROOT) does not correctly proxy the binary file
    download endpoint, so the project's home node is looked up via its (hub
    served) metadata first, and the file is then fetched directly from that
    node's own host.

    Args:
        project_ref (str): MDDB project accession, optionally suffixed with
            ".<mdNumber>" to select a replica.
        fi_name (str): name of the file to resolve, as listed in the
            project's "files".
        validate_uri (bool, optional): Check if URI exists. Defaults to True.
        sleep429 (int, optional): Sleep in seconds if 429 HTTP code returned

    Returns:
        str: file URI
    """
    accession = project_ref.split(".", 1)[0]
    md_suffix = project_ref[len(accession):]  # "" or ".<mdNumber>"

    metadata_uri = f"{MDDB_API_ROOT}/projects/{accession}"
    try:
        with urllib.request.urlopen(metadata_uri, timeout=10) as response:
            metadata = json.loads(response.read().decode())
    except Exception as e:
        raise RuntimeError(f"Could not fetch MDDB project metadata from {metadata_uri}: {e}")

    node, local = metadata.get("node"), metadata.get("local")
    if not node or not local:
        raise RuntimeError(
            f"MDDB project '{accession}' metadata is missing 'node'/'local': {metadata}"
        )

    uri = f"https://{node}.mddbr.eu/api/rest/v1/projects/{local}{md_suffix}/files/{fi_name}"
    if validate_uri:
        _validate_url(uri, sleep429, project_ref, fi_name)
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
