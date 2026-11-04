"""Resume the public NVIDIA layer with range connections, then verify its SHA.

Only for the isolated WSL distro's first pull. Keeps completed container layers.
Registry tokens are anonymous pull tokens and are never printed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DIGEST='sha256:65df5cb1d201c1994aae1eb24bc5000ee96c4eafd0393129c6999455a6d352d1'
TOTAL=9685195698
IMAGE='ghcr.io/nvidia/deepstream:9.2-samples-multiarch'


def main():
    if '24.04' not in Path('/etc/os-release').read_text():
        raise RuntimeError('Run only in dedicated Ubuntu 24.04')
    cached=subprocess.run(['docker','image','inspect','--format','{{.Id}}',IMAGE],capture_output=True,text=True)
    if cached.returncode and 'no such' not in cached.stderr.lower():
        raise RuntimeError('Docker is unavailable; refusing to start another download')
    if cached.returncode==0:
        if cached.stdout.strip()!='sha256:5ee7458d5e795592e0cb182457a7f8281f2c1da7bb38ce4b8506ac693f266746':
            raise RuntimeError('Image tag changed; review pinned download metadata')
        # These exact temporary files belong to this bootstrap job. The final
        # image already exists; no model, dataset or Docker image is removed.
        owned=Path('/var/tmp/straw-official-layer')
        for name in ('nvidia-layer.gz','nvidia-layer.gz.aria2',DIGEST,DIGEST+'.aria2'):
            item=owned/name
            if item.is_file():item.unlink()
        subprocess.run(['ctr','-n','moby','content','abort','layer-'+DIGEST],capture_output=True)
        print('Image already verified; removed only bootstrap download leftovers',flush=True)
        return
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: args=(proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        except (FileNotFoundError,PermissionError): continue
        if args==[b'docker',b'pull',IMAGE.encode()]:
            os.kill(int(proc.name),signal.SIGTERM)
    time.sleep(2)
    cache=Path('/var/tmp/straw-official-layer')
    cache.mkdir(exist_ok=True)
    target=cache/'nvidia-layer.gz'
    disposition_target=cache/DIGEST
    if disposition_target.exists():
        target=disposition_target
    if not target.exists():
        for ref in Path('/var/lib/containerd/io.containerd.content.v1.content/ingest').glob('*/ref'):
            if DIGEST.split(':')[1] in ref.read_text():
                source=ref.with_name('data')
                if source.exists():
                    from shutil import copyfile
                    copyfile(source,target)
                    print('Preserved partial download bytes:',target.stat().st_size,flush=True)
                    break
    subprocess.run(['apt-get','install','-y','--no-install-recommends','aria2'],check=True,stdout=subprocess.DEVNULL)
    if not target.exists() or target.stat().st_size!=TOTAL or target.with_name(target.name+'.aria2').exists():
        query=urlencode(dict(service='ghcr.io',scope='repository:nvidia/deepstream:pull'))
        with urlopen('https://ghcr.io/token?'+query,timeout=30) as response: token=json.load(response)['token']
        blob='https://ghcr.io/v2/nvidia/deepstream/blobs/'+DIGEST+'?straw_nonce='+str(time.time_ns())
        with urlopen(Request(blob,headers={'Authorization':'Bearer '+token,'Range':'bytes=0-0','Cache-Control':'no-cache'}),timeout=30) as response:
            resolved=response.geturl()
        log=Path(__file__).resolve().parents[2]/'artifacts/b1_stream/accelerated_layer.log'
        # Per-URI file options are required when aria2 reads an input file.
        download_input=f'{resolved}\n dir={cache}\n out={target.name}\n checksum=sha-256={DIGEST.split(":")[1]}\n'
        with log.open('wb') as output:
            subprocess.run(['aria2c','--input-file=-',
            '--continue=true','--auto-file-renaming=false','--file-allocation=none',
            '--max-connection-per-server=16','--split=16','--min-split-size=16M',
            '--retry-wait=2','--max-tries=5','--connect-timeout=15','--timeout=30','--lowest-speed-limit=512K',
            '--summary-interval=20','--enable-color=false','--show-console-readout=false',
            '--checksum=sha-256='+DIGEST.split(':')[1]],input=download_input.encode(),stdout=output,stderr=output,check=True)
    h=hashlib.sha256()
    with target.open('rb') as source:
        for block in iter(lambda:source.read(16*1024*1024),b''):h.update(block)
    if target.stat().st_size!=TOTAL or h.hexdigest()!=DIGEST.split(':')[1]:
        raise RuntimeError('Downloaded public layer failed size/SHA verification')
    with target.open('rb') as source:
        subprocess.run(['ctr','-n','moby','content','ingest','--expected-digest',DIGEST,
                        '--expected-size',str(TOTAL),'straw-verified-nvidia-layer'],stdin=source,check=True)
    # Only our verified temporary download is removed; Docker owns its copy.
    target.unlink()
    control=target.with_name(target.name+'.aria2')
    if control.exists():control.unlink()
    print('Verified NVIDIA layer imported; completing image pull',flush=True)
    subprocess.run(['docker','pull',IMAGE],check=True)


if __name__=='__main__': main()
