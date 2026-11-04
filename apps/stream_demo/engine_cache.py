"""Verify cached TensorRT engine against graph, SDK image, GPU and config."""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts'/os.environ.get('STRAW_STREAM_ARTIFACT','b1_stream')
ENGINE=OUT/'b1_rtx3050_fp16.engine'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def context():
    manifest=json.loads((OUT/'manifest.json').read_text())
    actual=sha(OUT/f"b1_rawrgb{manifest['size']}.onnx")
    if actual!=manifest['onnx_sha256']:raise RuntimeError('ONNX differs from frozen stream manifest')
    config=configparser.ConfigParser(interpolation=None)
    config.read(OUT/'nvinfer.txt')
    canonical=json.dumps({s:dict(config.items(s)) for s in config.sections()},sort_keys=True).encode()
    return dict(onnx_sha256=actual,checkpoint_sha256=manifest['checkpoint_sha256'],
                sdk_image_id=json.loads((OUT/'base_image.json').read_text())[0]['Id'],
                gpu_identity=(OUT/'gpu_identity.txt').read_text().strip(),
                inference_config_sha256=hashlib.sha256(canonical).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','record']);args=p.parse_args()
    key=context();meta=OUT/'engine_cache.json'
    if args.phase=='check':
        previous=json.loads(meta.read_text()) if meta.exists() else None
        valid=ENGINE.exists() and previous is not None and previous['context']==key and previous['engine_sha256']==sha(ENGINE)
        if ENGINE.exists() and not valid:
            archive=OUT/'engine_archive';archive.mkdir(exist_ok=True)
            ENGINE.rename(archive/f'{uuid.uuid4().hex}.engine')
            print('Archived unverified/stale engine; explicit TensorRT build required')
        print('TensorRT engine verified' if valid else 'TensorRT engine absent: conversion stage required')
    else:
        status=json.loads((OUT/'file.jsonl.summary.json').read_text())
        if status['failed'] or not status['eos'] or status['predicted']!=300 or status['invalid_tensors']:
            raise RuntimeError('Refuse to certify engine without successful complete file replay')
        value=dict(context=key,engine_sha256=sha(ENGINE))
        temp=meta.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(meta)
        print('Recorded engine provenance after 300-frame replay')


if __name__=='__main__':main()
