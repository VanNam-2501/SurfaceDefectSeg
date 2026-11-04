"""Measure received MJPEG preview cadence independently of inference cadence."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8033')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/verification/upgrade_v2/preview_cadence.json')
    args=parser.parse_args()
    state={}
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        with urlopen(args.url+'/api/video',timeout=5) as response:state=json.load(response)
        if state.get('frames_received',0)>=10 and state['status'] in ('calibrating','running'):break
        time.sleep(.1)
    else:raise RuntimeError('Start a video before measuring preview')
    times,numbers=[],[]
    with urlopen(args.url+'/api/video/stream.mjpg?session_id='+state['id'],timeout=10) as response:
        while len(times)<90:
            boundary=response.readline()
            if boundary==b'--frame--\r\n' or not boundary:break
            if boundary!=b'--frame\r\n':continue
            headers={}
            while True:
                line=response.readline()
                if line==b'\r\n':break
                key,value=line.decode('ascii').strip().split(':',1);headers[key.lower()]=value.strip()
            remaining=int(headers['content-length']);parts=[]
            while remaining:
                block=response.read(remaining)
                if not block:raise EOFError('Preview packet truncated')
                parts.append(block);remaining-=len(block)
            jpeg=b''.join(parts)
            if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):raise AssertionError('Invalid JPEG packet')
            response.read(2)
            times.append(time.perf_counter());numbers.append(int(headers['x-frame-number']))
    if len(times)<60:raise AssertionError('Insufficient preview frames')
    # Discard connection startup; measure on the same receiver clock.
    intervals=[(b-a)*1000 for a,b in zip(times[10:],times[11:])]
    fps=(len(times)-11)/(times[-1]-times[10])
    if fps<20 or any(b<=a for a,b in zip(numbers,numbers[1:])):raise AssertionError('Preview cadence or sequence failed')
    with urlopen(args.url+'/api/video') as response:final=json.load(response)
    result={'status':'PASS','scope':'Local HTTP MJPEG delivery on 30 FPS synthetic replay; not camera exposure or display refresh rate',
        'frames':len(times),'received_fps':fps,'interarrival_p50_ms':sorted(intervals)[len(intervals)//2],
        'interarrival_p95_ms':sorted(intervals)[int((len(intervals)-1)*.95)],
        'capture_frames_at_end':final['frames_received'],'analysis_steps_at_end':final['analysis_steps'],
        'source_kind':final['kind'],'session_id':state['id'],'sequence_strictly_increasing':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
