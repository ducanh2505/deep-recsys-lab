#!/usr/bin/env python3
"""Build the MovieLens-20M LightGCN dataset from a local ratings.csv."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data/raw/ml-20m/ratings.csv'
OUT = ROOT / 'data/processed/ml20m_lightgcn'
COLS = ['userId', 'movieId', 'rating', 'timestamp']
SCHEMA = {
    'userId': pl.Int64,
    'movieId': pl.Int64,
    'rating': pl.Float64,
    'timestamp': pl.Int64,
}

def read_source():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Place the MovieLens source file at {SOURCE}')
    df = pl.scan_csv(SOURCE, schema_overrides=SCHEMA).select(COLS).collect(engine='streaming')
    if df.height != 20_000_263:
        raise ValueError(f'Expected 20,000,263 source rows, got {df.height}')
    if df.null_count().row(0) != (0, 0, 0, 0):
        raise ValueError('Source contains null or malformed values')
    if df.select(pl.struct(['userId', 'movieId']).n_unique()).item() != df.height:
        raise ValueError('Source contains duplicate (userId, movieId) pairs')
    return df

def atomic_parquet(df,path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.select(COLS).sort(['userId', 'movieId']).write_parquet(tmp, compression='zstd')
    tmp.replace(path)

def split_df(df):
    rng=np.random.default_rng(42)
    trains=[]; valids=[]; tests=[]
    for g in df.sort(['userId','movieId']).partition_by('userId',maintain_order=True,as_dict=False):
        n=g.height; perm=rng.permutation(n)
        nt=int(np.floor(.20*n)); nv=int(np.floor(.10*(n-nt)))
        tests.append(g[perm[:nt]])
        valids.append(g[perm[nt:nt+nv]])
        trains.append(g[perm[nt+nv:]])
    def cat(groups): return pl.concat(groups) if groups else df.head(0)
    return cat(trains).sort(['userId','movieId']),cat(valids).sort(['userId','movieId']),cat(tests).sort(['userId','movieId'])

def canonical_hash(df):
    h=hashlib.sha256()
    for row in df.sort(['userId','movieId']).iter_rows():
        h.update((f'{row[0]},{row[1]},{row[2]:.1f},{row[3]}\n').encode())
    return h.hexdigest()

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    source = read_source()
    source_n = source.height
    positive = source.filter(pl.col('rating') >= 4.0)
    positive_n = positive.height
    core=positive; passes=[core.height]
    while True:
        uc=core.group_by('userId').len().rename({'len':'uc'})
        ic=core.group_by('movieId').len().rename({'len':'ic'})
        nxt=(core.join(uc,on='userId').join(ic,on='movieId').filter((pl.col('uc')>=10)&(pl.col('ic')>=10)).select(COLS))
        passes.append(nxt.height)
        if nxt.height==core.height: core=nxt; break
        core=nxt
    if core.select(pl.col('userId').n_unique()).item()==0: raise RuntimeError('Empty 10-core')
    atomic_parquet(core,OUT/'interactions.parquet')
    tr,va,te=split_df(core)
    train_items=tr.select('movieId').unique()
    va_keep=va.join(train_items,on='movieId',how='semi'); te_keep=te.join(train_items,on='movieId',how='semi')
    def exclusion(before,after):
        removed=before.join(after.select('userId','movieId'),on=['userId','movieId'],how='anti')
        return {'rows':removed.height,'distinct_items':removed.select(pl.col('movieId').n_unique()).item()}
    exv=exclusion(va,va_keep); ext=exclusion(te,te_keep)
    for name,d in [('train',tr),('valid',va_keep),('test',te_keep)]: atomic_parquet(d,OUT/f'{name}.parquet')
    # Distinguish floor zeros from exclusion-caused zeros by initial/final per-user counts.
    initial_v=va.group_by('userId').len().rename({'len':'v'}); initial_t=te.group_by('userId').len().rename({'len':'t'})
    final_v=va_keep.group_by('userId').len().rename({'len':'v'}); final_t=te_keep.group_by('userId').len().rename({'len':'t'})
    users=core.select('userId').unique()
    zdiag={}
    for key,ini,fin in [('valid',initial_v,final_v),('test',initial_t,final_t)]:
        col=key[0]
        base=users.join(ini,on='userId',how='left').rename({col:f'initial_{key}'})
        fincol='v' if key=='valid' else 't'
        base=base.join(fin,on='userId',how='left').rename({fincol:f'final_{key}'}).with_columns(pl.all().exclude('userId').fill_null(0))
        zdiag[key]={'zero_final_users':base.filter(pl.col(f'final_{key}')==0).height,
                    'zero_due_to_floor_rounding':base.filter((pl.col(f'initial_{key}')==0)&(pl.col(f'final_{key}')==0)).height,
                    'zero_due_to_cold_start_exclusion':base.filter((pl.col(f'initial_{key}')>0)&(pl.col(f'final_{key}')==0)).height}
    counts={'train':tr.height,'valid':va_keep.height,'test':te_keep.height}
    manifest={'source_rating_count':source_n,'positive_rating_count':positive_n,'ten_core_iteration_counts':passes,
      'ten_core':{'users':core.select(pl.col('userId').n_unique()).item(),'items':core.select(pl.col('movieId').n_unique()).item(),'interactions':core.height},
      'initial_split_counts':{'train':tr.height,'valid':va.height,'test':te.height},'excluded_cold_start':{'valid':exv,'test':ext},
      'final_split_counts':counts,'final_split_fractions_of_ten_core':{k:v/core.height for k,v in counts.items()},
      'seed':42,'formulas':{'test':'floor(0.20*n)','valid':'floor(0.10*(n-n_test))','train':'n-n_test-n_valid'},
      'ids':'Original userId and movieId values; no continuous ID mappings.','schema':COLS,
      'zero_evaluation_user_diagnostics':zdiag,'canonical_sha256':{k:canonical_hash(d) for k,d in [('train',tr),('valid',va_keep),('test',te_keep)]},
      'versions':{'python':sys.version.split()[0],'uv':os.environ.get('UV_VERSION','uv 0.11.14'),'polars':pl.__version__,'numpy':np.__version__},
      'paper_scope_note':'This MovieLens split and 10-core rule are requester-specified adaptations, not rules defined by LightGCN §4.1.'}
    tmp=OUT/'manifest.json.tmp'; tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); os.replace(tmp,OUT/'manifest.json')
    print(json.dumps({'source_rows':source_n,'positive_rows':positive_n,'ten_core_iterations':passes,'ten_core':manifest['ten_core'],'initial_split_counts':manifest['initial_split_counts'],'final_split_counts':counts,'excluded_cold_start':manifest['excluded_cold_start'],'zero_evaluation_user_diagnostics':zdiag},indent=2))

def main():
    build()
if __name__=='__main__': main()
