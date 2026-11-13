"""仅接受经过审核、仅改变路径的已完成任务清单迁移。"""
import hashlib
import json
from pathlib import Path
from .paths import ROOT

def migrated_profile_allowed(manifest_path, saved, current):
    receipt=ROOT/"data/provenance/migration_receipt.json"
    if not receipt.is_file():return False
    registry=json.loads(receipt.read_text(encoding="utf-8"))
    relative=Path(manifest_path).resolve().relative_to(ROOT).as_posix()
    expected=registry.get("legacy_profiles",{}).get(relative)
    if expected is None or hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()!=expected:return False
    pairs=sorted(registry["path_mappings"].items(),key=lambda p:len(p[0]),reverse=True)
    def mapped(value):
        if isinstance(value,dict):return {mapped(k):mapped(v) for k,v in value.items()}
        if isinstance(value,list):return [mapped(v) for v in value]
        if not isinstance(value,str):return value
        text=value.replace("\\","/")
        for old,new in pairs:
            old=(ROOT/old).as_posix();new=(ROOT/new).as_posix()
            if text==old or text.startswith(old+"/"):return new+text[len(old):]
        return text
    a,b=mapped(saved),mapped(current)
    oldprints=a.pop("fingerprints",{})
    newprints=b.pop("fingerprints",{})
    if a!=b:return False
    def iscode(path):return str(path).replace("\\","/").startswith(ROOT.as_posix()+"/code/")
    if {k:v for k,v in oldprints.items() if not iscode(k)}!={k:v for k,v in newprints.items() if not iscode(k)}:return False
    approved=registry.get("approved_code_sha256",{})
    for name,hash_value in newprints.items():
        if iscode(name):
            relative=Path(name).resolve().relative_to(ROOT).as_posix()
            if approved.get(relative)!=hash_value:return False
    return True
