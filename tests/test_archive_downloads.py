"""Offline regression tests for repository downloads, archive extraction, and
the metadata AddData writes to README.yaml so files can be found again later.

No network access is used: `urllib.request.urlopen` is patched everywhere a
download would otherwise happen.

Tests are grouped by the thing they protect:

* TestPathValidation / TestArchiveExtraction / TestAtomicDownload /
  TestArchiveCaching / TestRepositoryUrlResolution exercise the low-level
  building blocks in isolation.
* TestSourceFileMetadata and TestDownloadSystemFile cover the SOURCE_FILES
  bookkeeping that lets an archived file be found again.
* TestEndToEndRoundTrip is the important one: it simulates AddData writing
  README.yaml, throwing away the working directory (exactly what happens in
  practice), and a *separate* later process re-downloading the trajectory
  using nothing but the DOI and the saved README.yaml. If archive/member
  information is ever lost on the way into README.yaml, or a downloaded file
  lands anywhere but its exact expected path, these tests fail.
"""
import io
import json
import tarfile
import zipfile
from unittest.mock import patch
from urllib.error import HTTPError

import libarchive
import pytest
import yaml

from fairmd.idp import databankio as downloads
from fairmd.idp.core import System
from fairmd.idp.databankLibrary import (
    calc_file_sha1_hash, parse_valid_config_settings
)
from fairmd.idp.settings.engines import get_struc_top_traj_fnames, software_dict

GROMACS_FILE_KEYS = [
    key for key, spec in software_dict['GROMACS'].items()
    if 'file' in spec.get('TYPE', '')
]


def zip_bytes(entries):
    """Build an in-memory zip archive from {member_path: content_bytes}."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


class Response(io.BytesIO):
    """Minimal stand-in for the object urllib.request.urlopen() returns."""

    def __init__(self, content=b'', length=True,
                 url='https://edmond.mpg.de/dataset.xhtml'):
        super().__init__(content)
        self.headers = {'Content-Length': str(len(content))} if length else {}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def geturl(self):
        return self.url


def serve(by_url):
    """Build a urlopen(url) -> Response side_effect from {substring: bytes}.

    Any URL not matching a given substring (e.g. the initial DOI redirect
    lookup) gets a default Response(), matching the default edmond.mpg.de
    domain used elsewhere in these tests.
    """
    def open_url(url, *args, **kwargs):
        for substring, content in by_url.items():
            if substring in url:
                return Response(content)
        return Response()
    return open_url


# --------------------------------------------------------------------------
# Low-level building blocks
# --------------------------------------------------------------------------

class TestPathValidation:

    @pytest.mark.parametrize('path', [
        '/absolute.xtc', '../escape.xtc', 'a/../../escape', r'a\b', ''
    ])
    def test_rejects_unsafe_or_empty_paths(self, path):
        with pytest.raises(ValueError):
            downloads.validate_source_path(path)

    def test_accepts_plain_relative_path(self):
        assert str(downloads.validate_source_path('folder/run.xtc')) == 'folder/run.xtc'


class TestArchiveExtraction:

    @pytest.mark.parametrize('extension,format', [
        ('zip', 'zip'), ('tar', 'pax'), ('tar.gz', 'pax'), ('tgz', 'pax'), ('7z', '7zip'),
    ])
    def test_extracts_member_from_each_supported_format(self, tmp_path, extension, format):
        source = tmp_path / 'input.xtc'
        source.write_bytes(b'trajectory')
        archive = tmp_path / ('data.' + extension)
        options = {'filter_name': 'gzip'} if extension in ('tar.gz', 'tgz') else {}
        with libarchive.file_writer(str(archive), format, **options) as writer:
            writer.add_files(str(source), pathname='folder/input.xtc')

        result = downloads.extract_nested_file_from_archives(
            archive, 'folder/input.xtc', tmp_path / 'out')

        assert result.read_bytes() == b'trajectory'

    def test_descends_through_three_nested_archives(self, tmp_path):
        """Each layer's remaining path must reset, not accumulate the previous
        archive's name (the original nested-extraction bug)."""
        innermost = zip_bytes({'./folder/run.xtc': b'trajectory'})
        middle = zip_bytes({'dir/third.zip': innermost})
        outer = tmp_path / 'outer.zip'
        outer.write_bytes(zip_bytes({'second.zip': middle}))

        result = downloads.extract_nested_file_from_archives(
            outer, 'second.zip/dir/third.zip/folder/run.xtc', tmp_path / 'out')

        assert result.read_bytes() == b'trajectory'

    def test_missing_member_raises_file_not_found(self, tmp_path):
        archive = tmp_path / 'archive.tar'
        with tarfile.open(archive, 'w') as writer:
            writer.addfile(tarfile.TarInfo('present.xtc'), io.BytesIO(b'x'))
        with pytest.raises(FileNotFoundError):
            downloads.extract_file_from_archive(archive, 'absent.xtc', tmp_path / 'out')

    def test_symlink_member_is_rejected_not_followed(self, tmp_path):
        """A symlink entry must never be extracted verbatim -- it could point
        anywhere on the extracting machine's filesystem."""
        archive = tmp_path / 'archive.tar'
        with tarfile.open(archive, 'w') as writer:
            member = tarfile.TarInfo('link.xtc')
            member.type = tarfile.SYMTYPE
            member.linkname = '/etc/passwd'
            writer.addfile(member)

        with pytest.raises(ValueError, match='regular file'):
            downloads.extract_file_from_archive(archive, 'link.xtc', tmp_path / 'out')
        assert not (tmp_path / 'out').exists()


class TestAtomicDownload:

    def test_replaces_truncated_cache_and_skips_when_already_complete(self, tmp_path):
        destination = tmp_path / 'run.xtc'
        destination.write_bytes(b'bad')
        always_complete = lambda *a, **k: Response(b'complete')  # noqa: E731
        with patch.object(downloads.urllib.request, 'urlopen', side_effect=always_complete):
            assert downloads.download_resource_from_uri(
                'https://example/file', destination) == 2
            assert destination.read_bytes() == b'complete'
            assert downloads.download_resource_from_uri(
                'https://example/file', destination) == 1
            assert downloads.download_resource_from_uri(
                'https://example/file', destination, True) == 2

    def test_size_mismatch_leaves_previous_file_untouched(self, tmp_path):
        destination = tmp_path / 'sub/run.xtc'
        complete_response = Response(b'complete', length=False)
        with patch.object(downloads.urllib.request, 'urlopen', return_value=complete_response):
            downloads.download_resource_from_uri('https://example/file', destination)

        response = Response(b'short')
        response.headers['Content-Length'] = '100'
        with patch.object(downloads.urllib.request, 'urlopen', return_value=response):
            with pytest.raises(IOError, match='size mismatch'):
                downloads.download_resource_from_uri('https://example/file', destination, True)

        assert destination.read_bytes() == b'complete'
        assert list(destination.parent.iterdir()) == [destination], \
            "a failed download must not leave a temp file behind"


class TestArchiveCaching:

    def test_shared_archive_downloaded_once_for_two_members(self, tmp_path):
        content = zip_bytes({'folder/run.xtc': b'trajectory', 'folder/run.tpr': b'topology'})
        download_count = 0

        def open_url(*args, **kwargs):
            nonlocal download_count
            download_count += 1
            return Response(content)

        with patch.object(downloads.urllib.request, 'urlopen', side_effect=open_url):
            downloads.download_resource_from_uri(
                'https://example/archive.zip', tmp_path / 'run.xtc',
                source_path='archive.zip/folder/run.xtc')
            downloads.download_resource_from_uri(
                'https://example/archive.zip', tmp_path / 'run.tpr',
                source_path='archive.zip/folder/run.tpr')

        assert (tmp_path / 'run.xtc').read_bytes() == b'trajectory'
        assert (tmp_path / 'run.tpr').read_bytes() == b'topology'
        assert download_count == 2, \
            "one size-check request per requested member is expected"
        assert len(list((tmp_path / '.archives').rglob('archive.zip'))) == 1, \
            "both members must come from a single cached copy of the archive, not two"

    def test_deleted_local_copy_is_restored_from_cached_archive(self, tmp_path):
        content = zip_bytes({'folder/run.xtc': b'trajectory'})
        with patch.object(downloads.urllib.request, 'urlopen', return_value=Response(content)):
            downloads.download_resource_from_uri(
                'https://example/archive.zip', tmp_path / 'run.xtc',
                source_path='archive.zip/folder/run.xtc')
            (tmp_path / 'run.xtc').unlink()
            downloads.download_resource_from_uri(
                'https://example/archive.zip', tmp_path / 'run.xtc',
                source_path='archive.zip/folder/run.xtc')
        assert (tmp_path / 'run.xtc').read_bytes() == b'trajectory'

    def test_same_local_basename_from_different_archive_members_is_not_stale(self, tmp_path):
        """Two simulations from the same DOI can both want a file called
        run.xtc. The cache must key on the *member*, not the local name, or
        the second extraction would silently reuse the first one's bytes."""
        archive = tmp_path / 'source.zip'
        archive.write_bytes(zip_bytes({
            'replica1/run.xtc': b'first', 'replica2/run.xtc': b'second',
        }))
        destination = tmp_path / 'work/run.xtc'

        for member, expected in [('replica1', b'first'), ('replica2', b'second')]:
            downloads.download_resource_from_uri(
                archive.as_uri(), destination, source_path=f'source.zip/{member}/run.xtc')
            assert destination.read_bytes() == expected

        destination.write_bytes(b'corrupted local copy')
        downloads.download_resource_from_uri(
            archive.as_uri(), destination, source_path='source.zip/replica2/run.xtc')
        assert destination.read_bytes() == b'second'

    def test_legacy_dest_encoded_path_is_case_insensitive_for_archive_extension(self, tmp_path):
        archive = tmp_path / 'source.zip'
        archive.write_bytes(zip_bytes({'folder/run.xtc': b'trajectory'}))
        downloads.download_resource_from_uri(
            archive.as_uri(), tmp_path / 'work/archive.ZIP/folder/run.xtc')
        assert (tmp_path / 'work/run.xtc').read_bytes() == b'trajectory'

    def test_bare_archive_destination_is_downloaded_without_extraction(self, tmp_path):
        content = zip_bytes({'run.xtc': b'x'})
        with patch.object(downloads.urllib.request, 'urlopen', return_value=Response(content)):
            downloads.download_resource_from_uri('https://example/a.zip', tmp_path / 'a.zip')
        assert (tmp_path / 'a.zip').read_bytes() == content


class TestRepositoryUrlResolution:

    def test_zenodo_uses_the_archive_name_not_the_member_path(self):
        url = downloads.resolve_download_file_url(
            '10.5281/zenodo.123', 'archive.zip/folder/run.xtc', validate_uri=False)
        assert url == 'https://zenodo.org/record/123/files/archive.zip'

    def test_dataverse_falls_back_to_dataset_metadata_lookup_by_filename(self):
        open_url = serve({
            '/api/info/version': b'{"status":"OK"}',
        })

        def dispatch(url, **kwargs):
            if '/api/access/datafile/:persistentId' in url:
                raise HTTPError(url, 404, 'dataset DOI', {}, io.BytesIO())
            if '/api/datasets/' in url:
                return Response(json.dumps({'data': {'latestVersion': {'files': [
                    {'dataFile': {'filename': 'data.zip', 'id': 42}}]}}}).encode())
            return open_url(url, **kwargs)

        with patch.object(downloads.urllib.request, 'urlopen', side_effect=dispatch):
            url = downloads.resolve_download_file_url('10.17617/3.ABC', 'data.zip/run.xtc')
        assert url == 'https://edmond.mpg.de/api/access/datafile/42'

    def test_dataverse_resolves_directly_when_doi_is_a_file_doi(self):
        version_ok = serve({'/api/info/version': b'{"status":"OK"}'})
        with patch.object(downloads.urllib.request, 'urlopen', side_effect=version_ok):
            url = downloads.resolve_download_file_url('10.17617/3.ABC/file', 'data.zip/run.xtc')
        assert url == (
            'https://edmond.mpg.de/api/access/datafile/:persistentId'
            '?persistentId=doi:10.17617/3.ABC/file'
        )

    def test_mddb_resolves_via_node_lookup(self):
        """A node's hostname must come from /nodes, not be guessed from its
        alias -- e.g. alias "cin" is actually hosted at cineca.mddbr.eu."""
        def dispatch(url, **kwargs):
            if '/nodes' in url:
                return Response(json.dumps(
                    [{'alias': 'cin', 'api_url': 'https://cineca.mddbr.eu/api/'}]).encode())
            if '/projects/cin-A00IR' in url and '/files/' not in url:
                return Response(json.dumps({'node': 'cin', 'local': 'A00IR'}).encode())
            return Response()

        with patch.object(downloads.urllib.request, 'urlopen', side_effect=dispatch):
            url = downloads.resolve_download_file_url('mddb:cin-A00IR', 'trajectory.xtc')
        assert url == 'https://cineca.mddbr.eu/api/rest/v1/projects/A00IR/files/trajectory.xtc'

    def test_mddb_replica_suffix_is_preserved_on_the_node_local_id(self):
        def dispatch(url, **kwargs):
            if '/nodes' in url:
                return Response(json.dumps(
                    [{'alias': 'bsc', 'api_url': 'https://bsc.mddbr.eu/api/'}]).encode())
            if '/projects/bsc-A0008' in url and '/files/' not in url:
                return Response(json.dumps({'node': 'bsc', 'local': 'A0008'}).encode())
            return Response()

        with patch.object(downloads.urllib.request, 'urlopen', side_effect=dispatch):
            url = downloads.resolve_download_file_url('mddb:bsc-A0008.2', 'topology.tpr')
        assert url == 'https://bsc.mddbr.eu/api/rest/v1/projects/A0008.2/files/topology.tpr'

    def test_mddb_missing_node_in_metadata_raises(self):
        with patch.object(downloads.urllib.request, 'urlopen',
                           return_value=Response(json.dumps({}).encode())):
            with pytest.raises(RuntimeError, match="node"):
                downloads.resolve_download_file_url('mddb:bsc-A0008', 'trajectory.xtc')

    def test_mddb_unknown_node_alias_raises(self):
        def dispatch(url, **kwargs):
            if '/nodes' in url:
                return Response(json.dumps([]).encode())
            return Response(json.dumps({'node': 'bsc', 'local': 'A0008'}).encode())

        with patch.object(downloads.urllib.request, 'urlopen', side_effect=dispatch):
            with pytest.raises(RuntimeError, match="not found"):
                downloads.resolve_download_file_url('mddb:bsc-A0008', 'trajectory.xtc')


# --------------------------------------------------------------------------
# SOURCE_FILES metadata: the archive/member address that must survive import
# --------------------------------------------------------------------------

class TestSourceFileMetadata:

    def test_archived_files_get_a_source_files_entry(self):
        sim = {
            'DOI': '10.5281/zenodo.123',
            'TRJ': [['outer.zip/dir/run.xtc']],
            'TPR': [['outer.zip/dir/run.tpr']],
        }

        sources = downloads.prepare_file_sources(sim, ['TRJ', 'TPR'])

        assert sources == {
            'run.xtc': 'outer.zip/dir/run.xtc', 'run.tpr': 'outer.zip/dir/run.tpr',
        }
        assert sim['TRJ'] == [['run.xtc']], "the local field must stay a bare filename"
        assert sim['SOURCE_FILES'] == {
            'run.xtc': 'outer.zip/dir/run.xtc',
            'run.tpr': 'outer.zip/dir/run.tpr',
        }, "the archive path must be recorded, or it can never be found again"

    def test_plain_repository_files_get_no_source_files_entry(self):
        """A file uploaded standalone is already fully addressed by its bare
        name -- SOURCE_FILES would be pure noise, so it must not be added."""
        sim = {'DOI': '10.5281/zenodo.123', 'TRJ': [['run.xtc']], 'TPR': [['run.tpr']]}

        downloads.prepare_file_sources(sim, ['TRJ', 'TPR'])

        assert 'SOURCE_FILES' not in sim

    def test_mixed_archived_and_plain_files_only_record_the_archived_one(self):
        sim = {'DOI': '10.5281/zenodo.123', 'TRJ': [['outer.zip/run.xtc']], 'TPR': [['run.tpr']]}

        downloads.prepare_file_sources(sim, ['TRJ', 'TPR'])

        assert sim['SOURCE_FILES'] == {'run.xtc': 'outer.zip/run.xtc'}

    def test_source_files_metadata_survives_yaml_round_trip(self):
        sim = {
            'DOI': '10.5281/zenodo.123',
            'TRJ': [['outer.zip/dir/run.xtc']],
            'OTHER': [['do/not/modify']],
        }

        downloads.prepare_file_sources(sim, ['TRJ'])
        saved = yaml.safe_load(yaml.safe_dump(sim))

        assert saved['SOURCE_FILES'] == {'run.xtc': 'outer.zip/dir/run.xtc'}
        assert saved['OTHER'] == [['do/not/modify']], "unrelated fields must be untouched"

    def test_two_files_colliding_on_local_name_are_rejected_before_any_mutation(self):
        sim = {'TRJ': [['one.zip/run.xtc'], ['two.zip/run.xtc']]}

        with pytest.raises(ValueError, match='share local filename'):
            downloads.prepare_file_sources(sim, ['TRJ'])

        assert sim['TRJ'][0][0] == 'one.zip/run.xtc', \
            "a rejected batch must not partially mutate sim"
        assert 'SOURCE_FILES' not in sim


class TestDownloadSystemFile:
    """download_system_file() is the one function every re-download call site
    (AddData, databankLibrary, analyze.py, analyze_nmrpca.py) should use."""

    def test_uses_source_files_to_locate_an_archived_member(self, tmp_path):
        content = zip_bytes({'replica1/run.xtc': b'trajectory'})
        system = {
            'DOI': '10.5281/zenodo.123',
            'SOURCE_FILES': {'run.xtc': 'simulation.zip/replica1/run.xtc'},
        }
        destination = tmp_path / 'run.xtc'

        with patch.object(downloads, 'resolve_download_file_url') as resolve, \
                patch.object(downloads.urllib.request, 'urlopen', return_value=Response(content)):
            resolve.return_value = 'https://example/simulation.zip'
            downloads.download_system_file(system, 'run.xtc', destination)
            resolve.assert_called_once_with(system['DOI'], 'simulation.zip/replica1/run.xtc')

        assert destination.read_bytes() == b'trajectory'

    def test_falls_back_to_local_name_when_source_files_is_absent(self, tmp_path):
        """README.yaml files written before this feature existed have no
        SOURCE_FILES key at all; they must keep working exactly as before."""
        system = {'DOI': '10.5281/zenodo.123'}
        destination = tmp_path / 'run.xtc'

        with patch.object(downloads, 'resolve_download_file_url') as resolve, \
                patch.object(downloads.urllib.request, 'urlopen', return_value=Response(b'plain')):
            resolve.return_value = 'https://example/run.xtc'
            downloads.download_system_file(system, 'run.xtc', destination)
            resolve.assert_called_once_with(system['DOI'], 'run.xtc')

        assert destination.read_bytes() == b'plain'


# --------------------------------------------------------------------------
# End-to-end: does a file survive from "AddData imports it" all the way to
# "someone else re-downloads it for analysis, on a different machine"?
# --------------------------------------------------------------------------

class TestEndToEndRoundTrip:

    def _config(self, tmp_path, trj, tpr, gro=None):
        config = {
            'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
            'DIR_WRK': str(tmp_path), 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
            'TRJ': trj, 'TPR': tpr,
        }
        if gro is not None:
            config['GRO'] = gro
        return config

    def test_archived_files_land_at_the_exact_expected_local_paths(self, tmp_path):
        """AddData's download stage must write each file to
        <work>/<local_name> -- nothing nested under the archive's own name,
        nothing left in a temp directory."""
        payloads = {'folder/run.xtc': b'trajectory-bytes', 'folder/run.tpr': b'topology-bytes'}
        repository_archive = tmp_path / 'repo.zip'
        repository_archive.write_bytes(zip_bytes(payloads))

        config = self._config(
            tmp_path, trj=[['repo.zip/folder/run.xtc']], tpr=[['repo.zip/folder/run.tpr']])
        sim, _ = parse_valid_config_settings(config)
        sim = System(sim)
        work = tmp_path / 'work'

        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_archive.as_uri()):
            local_files = downloads.download_simulation_files(sim, work, GROMACS_FILE_KEYS)

        assert sorted(local_files) == ['run.tpr', 'run.xtc']
        assert (work / 'run.xtc').read_bytes() == payloads['folder/run.xtc']
        assert (work / 'run.tpr').read_bytes() == payloads['folder/run.tpr']
        assert sim['SOURCE_FILES'] == {
            'run.xtc': 'repo.zip/folder/run.xtc',
            'run.tpr': 'repo.zip/folder/run.tpr',
        }

    def test_archived_trajectory_is_redownloadable_from_readme_alone(self, tmp_path):
        """The core regression test. It deliberately never reuses the `sim`
        object from the import step: it re-parses a freshly saved
        README.yaml, exactly as a later analysis run (possibly on a
        different computer, long after DIR_WRK was deleted) would."""
        payload = b'the actual trajectory bytes'
        repository_archive = tmp_path / 'simulation_data.zip'
        repository_archive.write_bytes(zip_bytes({
            'replica1/traj.xtc': payload,
            'replica1/traj.tpr': b'the topology bytes',
        }))

        config = self._config(
            tmp_path, trj=[['simulation_data.zip/replica1/traj.xtc']],
            tpr=[['simulation_data.zip/replica1/traj.tpr']])
        sim, _ = parse_valid_config_settings(config)
        sim = System(sim)
        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_archive.as_uri()):
            downloads.download_simulation_files(sim, tmp_path / 'import_work', GROMACS_FILE_KEYS)

        # Simulate AddData's actual save step, then simulate everything about
        # the import being gone except this file.
        readme_path = tmp_path / 'README.yaml'
        readme_path.write_text(yaml.safe_dump(sim.readme, sort_keys=False))
        del sim
        system_from_disk = yaml.safe_load(readme_path.read_text())

        reanalysis_dir = tmp_path / 'reanalysis'
        local_name = system_from_disk['TRJ'][0][0]
        assert local_name == 'traj.xtc'
        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_archive.as_uri()):
            downloads.download_system_file(
                system_from_disk, local_name, reanalysis_dir / local_name)

        assert (reanalysis_dir / 'traj.xtc').read_bytes() == payload

    def test_plain_repository_files_are_also_redownloadable_from_readme_alone(self, tmp_path):
        """Same scenario as above, but for a file that was never zipped --
        confirms the fix does not regress the common, non-archived case."""
        payload = b'standalone trajectory bytes'
        repository_file = tmp_path / 'md_run_nojump.xtc'
        repository_file.write_bytes(payload)

        config = self._config(tmp_path, trj=[['md_run_nojump.xtc']], tpr=[['md_run.tpr']])
        sim, _ = parse_valid_config_settings(config)
        sim = System(sim)
        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_file.as_uri()):
            downloads.download_simulation_files(sim, tmp_path / 'import_work', ['TRJ'])

        assert 'SOURCE_FILES' not in sim.readme

        readme_path = tmp_path / 'README.yaml'
        readme_path.write_text(yaml.safe_dump(sim.readme, sort_keys=False))
        system_from_disk = yaml.safe_load(readme_path.read_text())
        assert 'SOURCE_FILES' not in system_from_disk

        reanalysis_dir = tmp_path / 'reanalysis'
        local_name = system_from_disk['TRJ'][0][0]
        assert local_name == 'md_run_nojump.xtc'
        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_file.as_uri()):
            downloads.download_system_file(
                system_from_disk, local_name, reanalysis_dir / local_name)

        assert (reanalysis_dir / local_name).read_bytes() == payload

    def test_full_adddata_pipeline_hash_save_reload_and_mdanalysis_load(self, tmp_path):
        """Exercises AddData's download + hashing stage on a doubly-nested
        archive, then proves the saved README.yaml is sufficient to reload
        the trajectory into a real MDAnalysis Universe."""
        import MDAnalysis as mda

        gro = tmp_path / 'run.gro'
        gro.write_text(
            'test\n    1\n    1ALA     CA    1   0.100   0.200   0.300\n   1.0   1.0   1.0\n'
        )
        universe = mda.Universe(str(gro))
        xtc = tmp_path / 'run.xtc'
        with mda.Writer(str(xtc), n_atoms=1) as writer:
            writer.write(universe.atoms)
        payloads = {
            'run.gro': gro.read_bytes(),
            'run.xtc': xtc.read_bytes(),
            'run.tpr': b'unsupported TPR',
        }
        archive = zip_bytes({
            'nested.zip': zip_bytes({'folder/' + k: v for k, v in payloads.items()})
        })
        repository_archive = tmp_path / 'outer.zip'
        repository_archive.write_bytes(archive)

        config = self._config(
            tmp_path,
            trj='outer.zip/nested.zip/folder/run.xtc',
            tpr=[['outer.zip/nested.zip/folder/run.tpr']],
            gro=['outer.zip/nested.zip/folder/run.gro'])
        parsed, _ = parse_valid_config_settings(config)
        sim = System(parsed)
        work = tmp_path / 'work'

        with patch.object(downloads, 'resolve_download_file_url',
                           return_value=repository_archive.as_uri()):
            local_files = downloads.download_simulation_files(sim, work, GROMACS_FILE_KEYS)

        for filename in local_files:
            assert (work / filename).read_bytes() == payloads[filename]
            assert len(calc_file_sha1_hash(work / filename)) == 40

        readme = tmp_path / 'README.yaml'
        readme.write_text(yaml.safe_dump(sim.readme))
        saved = yaml.safe_load(readme.read_text())
        assert saved['TRJ'] == [['run.xtc']]
        assert saved['TPR'] == [['run.tpr']]
        assert saved['SOURCE_FILES'] == {
            'run.xtc': 'outer.zip/nested.zip/folder/run.xtc',
            'run.tpr': 'outer.zip/nested.zip/folder/run.tpr',
            'run.gro': 'outer.zip/nested.zip/folder/run.gro',
        }

        structure, _, trajectory = get_struc_top_traj_fnames(saved, join_path=str(work))
        restored = mda.Universe(structure, trajectory)
        assert len(restored.atoms) == 1
        assert len(restored.trajectory) == 1
        assert restored.atoms.positions[0] == pytest.approx([1, 2, 3])
        restored.trajectory.close()

    def test_optional_missing_file_is_skipped_and_ordinary_download_still_works(self, tmp_path):
        parsed, _ = parse_valid_config_settings({
            'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
            'DIR_WRK': str(tmp_path), 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
            'TRJ': 'run.xtc', 'TPR': 'run.tpr', 'GRO': None})
        original = tmp_path / 'original'
        original.write_bytes(b'plain resource')

        with patch.object(downloads, 'resolve_download_file_url', return_value=original.as_uri()):
            downloads.download_simulation_files(parsed, tmp_path / 'work', ['TRJ', 'TPR'])

        assert (tmp_path / 'work/run.xtc').read_bytes() == b'plain resource'
        assert 'SOURCE_FILES' not in parsed


@pytest.mark.parametrize('trajectory', [
    'data.zip/run.xtc', ['data.zip/run.xtc'],
    [['data.zip/run.xtc']], [['data.zip/run.xtc', 'sha1']],
])
def test_adddata_accepts_all_yaml_file_field_shapes(trajectory):
    """TRJ/TPR/etc. may be written in YAML as a bare string, a one-item list,
    a list of one-item lists, or a list of [name, hash] pairs -- AddData must
    normalize all of them the same way before archive paths are extracted."""
    sim, files = parse_valid_config_settings({
        'SOFTWARE': 'GROMACS', 'DOI': '10.5281/zenodo.123', 'SYSTEM': 'test',
        'DIR_WRK': '/tmp', 'PREEQTIME': 0, 'TIMELEFTOUT': 0, 'COMPOSITION': {},
        'TRJ': trajectory, 'TPR': [['data.zip/run.tpr']]})
    assert files == ['data.zip/run.xtc', 'data.zip/run.tpr']
    assert sim['TRJ'][0][0] == 'data.zip/run.xtc'
