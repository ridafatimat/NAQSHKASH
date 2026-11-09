"""Synchronous checkpoint persistence; no authentication or background threads."""
from pathlib import Path
import os
import shutil
import tempfile
import warnings
import zipfile


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix=destination.name, suffix='.tmp')
    os.close(fd)
    try:
        shutil.copyfile(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def backup(directory, archive):
    """Call only at a completed last.pt save, when the local pair is consistent."""
    directory, archive = Path(directory), Path(archive)
    if not (directory / 'last.pt').is_file():
        raise FileNotFoundError(directory / 'last.pt')
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=archive.parent, prefix=archive.name, suffix='.tmp')
    os.close(fd)
    try:
        with zipfile.ZipFile(name, 'w', compression=zipfile.ZIP_STORED) as output:
            for filename in ('last.pt', 'best.pt'):
                if (directory / filename).is_file():
                    output.write(directory / filename, filename)
        os.replace(name, archive)
    finally:
        Path(name).unlink(missing_ok=True)
    return archive


def mirror_checkpoint(path, mirror_dir):
    """Local save already succeeded. Failure warns and preserves the local run."""
    if mirror_dir is None:
        return
    path, mirror_dir = Path(path), Path(mirror_dir)
    try:
        if path.parent.resolve() == mirror_dir.resolve():
            raise ValueError('Mirror directory must differ from active output')
        atomic_copy(path, mirror_dir / path.name)
        if path.name == 'last.pt':
            # A single atomic bundle avoids mismatched best/last after a disconnect
            # between the two epoch-end saves. Restore this bundle preferentially.
            backup(path.parent, mirror_dir / 'checkpoints.zip')
        print(f'Checkpoint persisted: {mirror_dir / path.name}', flush=True)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        warnings.warn(f'CHECKPOINT PERSISTENCE FAILED: {error}; local checkpoint retained. Back up now.', RuntimeWarning)


def restore(archive, directory):
    """Restore only into a clean run folder; reject unexpected archive members."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('*.pt')):
        raise FileExistsError('Restore requires an output folder without .pt files')
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if 'last.pt' not in names or len(names) != len(set(names)) or set(names) - {'last.pt', 'best.pt'}:
            raise ValueError('Expected last.pt and optional best.pt only')
        if source.testzip() is not None:
            raise ValueError('Checkpoint archive failed CRC verification')
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            source.extractall(temporary)
            for name in names:
                atomic_copy(Path(temporary) / name, directory / name)


def quick_dataset(path, limit):
    """Bound metadata I/O for preflight only; retain the existing dataset reader."""
    import csv
    import itertools
    from data.tier_dataset import TalimTierDataset
    dataset = TalimTierDataset.__new__(TalimTierDataset)
    dataset.dataset_root = Path(path).resolve()
    dataset.metadata_path = dataset.dataset_root / 'metadata.csv'
    dataset.transform = None
    dataset.samples = []
    with dataset.metadata_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in itertools.islice(csv.DictReader(stream), limit):
            sample = {key: row[key] for key in ('tier_id', 'row_id', 'parent_sample_id', 'tier_name')}
            for key in ('image_path', 'label_path'):
                sample[key] = (dataset.dataset_root / row[key]).resolve()
                if not sample[key].is_file():
                    raise FileNotFoundError(sample[key])
            dataset.samples.append(sample)
    if not dataset.samples:
        raise ValueError('Empty preflight dataset')
    return dataset
