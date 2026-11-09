"""Execute byte-identical snapshots; relocate paths/imports, never policy logic.

Only snapshotted modules receive a pathlib proxy. Standard/site packages and
the authorized data adapter retain normal pathlib. Working-project imports
are forbidden. Explicit importlib file loads are covered as well as imports.
Use in a fresh Python process; importing any working policy first is rejected.
"""
import builtins,importlib,importlib.abc,importlib.machinery,importlib.util,json,sys,types
from pathlib import Path as RealPath
from verify_freeze import verify,sha,canonical


class FrozenRuntime:
    def __init__(self,root):
        self.root=RealPath(root).resolve();self.manifest=verify(self.root)
        self.project=self.root/'dependencies';self.external=self.root/'external/Talamat'
        self.adapter=self.root.parent/'heldout_evaluation'
        self.origins=(RealPath(r'C:\NAQSHKASH'),RealPath(r'C:\FYP Dataset Generation'))
        self.allowed_files={str((self.root/p).resolve()).lower() for p in self.manifest['files']}
        self._install()
        self.policy=importlib.import_module('policy_v3');self.integration=importlib.import_module('integration')
        self.r2=importlib.import_module('r2_structural');self.regions=importlib.import_module('review_overlay');self.localization=importlib.import_module('localization_metrics')
        self.configuration=self.policy.load_policy();self.policy.verify_dependencies(self.configuration)

    def remap(self,path):
        p=RealPath(path)
        if p.is_absolute() and (p.is_relative_to(self.root) or p.is_relative_to(self.adapter)):return p
        for original,target in zip(self.origins,(self.project,self.external)):
            if p.is_absolute() and p.is_relative_to(original):return target/p.relative_to(original)
        return p

    def working_path(self,path):
        p=RealPath(path).resolve()
        return any(p.is_relative_to(o) for o in self.origins) and not(p.is_relative_to(self.root) or p.is_relative_to(self.adapter))

    def _install(self):
        runtime=self
        for name,module in list(sys.modules.items()):
            f=getattr(module,'__file__',None)
            if f and self.working_path(f):raise RuntimeError('Working module already imported: '+name)
        sys.dont_write_bytecode=True
        sys.path[:]=[p for p in sys.path if not self.working_path(p or RealPath.cwd())]
        sys.path[:0]=[str(self.root/'src'),str(self.project),str(self.project/'stage2/stage2f_rejection_evaluation_v2/src')]
        class SnapshotPath(type(RealPath())):
            def __init__(self,*args,**kwargs):super().__init__(runtime.remap(RealPath(*args)),**kwargs)
        proxy=types.ModuleType('pathlib')
        import pathlib
        proxy.__dict__.update(pathlib.__dict__);proxy.Path=SnapshotPath
        original_import=builtins.__import__
        def frozen_import(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='pathlib' and level==0:return proxy
            return original_import(name,globals,locals,fromlist,level)
        class Loader(importlib.machinery.SourceFileLoader):
            def exec_module(self,module):
                p=RealPath(self.path).resolve()
                if str(p).lower() not in runtime.allowed_files:raise ImportError('Unlisted frozen source: '+str(p))
                b=dict(vars(builtins));b['__import__']=frozen_import;module.__dict__['__builtins__']=b
                exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__)
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                spec=importlib.machinery.PathFinder.find_spec(fullname,path)
                if spec and spec.origin and spec.origin not in ('built-in','frozen'):
                    p=RealPath(spec.origin).resolve()
                    if runtime.working_path(p):raise ImportError('Working-project fallback blocked: '+fullname+' '+str(p))
                    if p.is_relative_to(runtime.adapter) and p.suffix=='.py' and p.relative_to(runtime.adapter).as_posix() not in runtime.manifest['heldout_adapter_sha256']:
                        raise ImportError('Unlisted held-out adapter source: '+str(p))
                    if p.is_relative_to(runtime.root) and p.suffix=='.py':spec.loader=Loader(fullname,str(p));return spec
                return None
        self.finder=Finder();sys.meta_path.insert(0,self.finder)
        original_spec=importlib.util.spec_from_file_location
        def spec_from_file_location(name,location,*args,**kwargs):
            location=runtime.remap(location);p=RealPath(location).resolve()
            if runtime.working_path(p):raise ImportError('Working explicit file load blocked: '+str(p))
            if p.is_relative_to(runtime.root) and p.suffix=='.py':kwargs['loader']=Loader(name,str(p))
            return original_spec(name,location,*args,**kwargs)
        importlib.util.spec_from_file_location=spec_from_file_location

    def loaded_sources(self):
        local={name:str(RealPath(m.__file__).resolve().relative_to(self.root)) for name,m in list(sys.modules.items()) if getattr(m,'__file__',None) and RealPath(m.__file__).resolve().is_relative_to(self.root)}
        forbidden=[name for name,m in list(sys.modules.items()) if getattr(m,'__file__',None) and self.working_path(m.__file__)]
        if forbidden:raise AssertionError('Working local modules used: '+repr(forbidden))
        return local

    def rendering_context(self,metadata):return self.integration.runtime_rendering_context(metadata)

    def decide(self,observation,rendering_context):
        r1=self.integration.apply_candidate(observation,rendering_context,'R1')
        text=r1['interpreted_observation']['decoded_rows'];r2,findings=self.r2.structural_decision(r1['final_decision'],text)
        result=self.policy.decide(r2,text,observation,self.configuration)
        return dict(R1=r1,R2=r2,R2_findings=findings,R3=result,text=text,carpet_generation_ready=False,
                    selected_carpet_reading=None,carpet_generation_block_reason='R3 text consistency does not establish physical alignment; separate authorized resolution is required',
                    ambiguous_text_alignment=result['alignment_evidence']['analysis']['page_candidate_count_in_search']>1)

    def pipeline(self):
        # Manual inference only. Original factory/model/checkpoint, CPU as approved.
        return self.integration.make_pixel_pipeline('cpu')

    def inspect_inference_imports(self):
        # Import definitions only: no model instantiation, checkpoint deserialization,
        # D1 execution or image read. Covers lazy transitive dependencies preflight.
        legacy=self.project/'stage2/stage2f_negative_rejection/src/evaluate_baseline.py'
        spec=importlib.util.spec_from_file_location('frozen_inference_import_preflight',legacy)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        importlib.import_module('training.manifest_training')
        sys.path.insert(0,str(self.project/'stage2/stage2d_experiments'))
        spec=importlib.util.spec_from_file_location('frozen_d1_import_preflight',self.project/'stage2/stage2d_experiments/pipeline_d1.py')
        d1=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d1;spec.loader.exec_module(d1)
        return self.loaded_sources()
