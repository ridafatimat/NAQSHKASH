"""Restricted reuse of the existing scan engine with independently checked evidence."""
from functools import lru_cache
from common_v2 import *
import io


@lru_cache(maxsize=1)
def engine():
    path=Path(r'C:\FYP Dataset Generation\talamat_scan_scripts_v1_1\talamat_scan_augment.py')
    spec=importlib.util.spec_from_file_location('v2_readonly_scan_engine',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def make_variant(image,seed):
    # Lossless pointwise illumination allows independent exact raster checking
    # after augmentation. Lossy/spatial recipes are deliberately NOT certified.
    recipe=dict(padding=0,operations=[dict(type='brightness',factor=0.92 if seed%2 else 0.96)],
                source_engine='existing talamat_scan_augment.apply_recipe',output_format='PNG',
                scope='mild illumination-style only; JPEG/blur/noise/geometry need a separate evidence verifier')
    module=engine();im=module.apply_recipe(image,recipe)
    stream=io.BytesIO();im.save(stream,format='PNG');raw=stream.getvalue()
    decoded=Image.open(io.BytesIO(raw));decoded.load()
    return raw,decoded,recipe
