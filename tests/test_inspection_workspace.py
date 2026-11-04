"""Regression coverage for quality, product identity, persistence and research scoping."""
from pathlib import Path
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'artifacts/test_tmp').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from apps.stream_demo.image_quality import assess, QualityConfig
from apps.stream_demo.video_tracking import VideoGate, GateConfig
from apps.stream_demo.inspection_store import InspectionStore
from apps.stream_demo.inspection_service import InspectionService
from apps.stream_demo.research_catalog import ResearchCatalog
from apps.stream_demo.video_session import redact_source, VideoSession
from apps.stream_demo import preview_web


def textured(size=96):
    data = np.random.default_rng(42).integers(40, 215, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(data)


class FakeResident:
    identity = {"contract_sha256": "fixture-contract", "engine_sha256": "fixture-engine", "checkpoint_sha256": "fixture-checkpoint"}
    def __init__(self, *args, **kwargs): self.calls = 0
    def submit(self, path, event_key, metadata): self.calls += 1;return "a" * 32
    def status(self): return {"state": "ready"}
    def result(self, request_id):
        return {"status": "completed", "class": "bent", "defective": True, "confidence": .9,
                "contract_sha256": "fixture-contract", "server_event_to_result_ms": 25}


class QualityTests(unittest.TestCase):
    def test_dark_flat_and_blurred_are_not_good_decisions(self):
        self.assertIn("too_dark", assess(Image.new("RGB", (96, 96), "black"))["reasons"])
        self.assertIn("blurred", assess(textured().filter(ImageFilter.GaussianBlur(12)))["reasons"])
        self.assertTrue(assess(textured())["acceptable"])

    def test_crop_border_does_not_hide_blurred_center(self):
        image = Image.new("RGB", (256, 256), "black")
        image.paste(Image.new("RGB", (224, 224), (128, 128, 128)), (16, 16))
        self.assertIn("blurred", assess(image)["reasons"])

    def test_quality_settings_reject_nan_and_reversed_brightness(self):
        for value in ({"min_sharpness": float("nan")}, {"min_brightness": 250, "max_brightness": 200}, {"extra": 1}):
            with self.assertRaises(ValueError): QualityConfig.parse(value)

    def test_explicit_clipping_requires_review(self):
        self.assertIn("clipped_product", assess(textured(), clipped=True)["reasons"])


class TrackingTests(unittest.TestCase):
    def make_gate(self):
        gate = VideoGate(GateConfig(calibration_frames=3))
        for i in range(3): gate.process(Image.new("RGB", (320, 180), (110, 110, 110)), i)
        return gate

    def frame(self, x=None, brightness=0):
        image = Image.new("RGB", (320, 180), (110+brightness,) * 3)
        if x is not None:
            tile = textured(65).point(lambda v: 125 + v // 2)
            image.paste(tile, (x, 55))
        return image

    def test_uniform_exposure_change_does_not_trigger(self):
        gate = self.make_gate()
        for i in range(20): self.assertEqual(gate.process(self.frame(brightness=25), i)["events"], [])

    def test_crossing_once_selects_a_source_frame_with_same_track(self):
        gate = self.make_gate();events=[]
        for i,x in enumerate(range(280,-70,-7)):
            events.extend(gate.process(self.frame(x), i*100000000)["events"])
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["track_id"],1)
        self.assertLessEqual(events[0]["pts_ns"],events[0]["trigger_pts_ns"])
        self.assertTrue(events[0]["quality"]["acceptable"])

    def test_stationary_product_after_crossing_is_not_repeated(self):
        gate=self.make_gate();events=[]
        for i,x in enumerate(list(range(280,125,-7))+[125]*20): events.extend(gate.process(self.frame(x),i)["events"])
        self.assertEqual(len(events),1)

    def test_recalibration_rejects_active_tracks(self):
        gate=self.make_gate();gate.process(self.frame(200),1)
        with self.assertRaises(ValueError):gate.recalibrate()

    def test_gate_rejects_invalid_regions_and_direction(self):
        for spec in ({"roi":[.8,0,.5,1]}, {"direction":"unknown"}, {"line":float("nan")}):
            with self.assertRaises(ValueError):GateConfig.parse(spec)

    def test_sharp_frame_is_selected_before_a_blurred_crossing(self):
        gate=self.make_gate();events=[]
        for i,x in enumerate(range(280,100,-7)):
            frame=self.frame(x)
            if x<150:frame=frame.filter(ImageFilter.GaussianBlur(3))
            events.extend(gate.process(frame,i*100000000)['events'])
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]['quality']['acceptable'])
        self.assertLess(events[0]['pts_ns'],events[0]['trigger_pts_ns'])

    def test_unconfirmed_exit_is_reviewed_instead_of_silently_lost(self):
        gate=self.make_gate();events=[]
        for i,x in enumerate([260,250,240,230,220,210,200,190,180]):gate.process(self.frame(x),i)
        for i in range(8):events.extend(gate.process(self.frame(),100+i)['events'])
        self.assertEqual(len(events),1)
        self.assertIn('roi_crossing_unconfirmed',events[0]['quality']['reasons'])

    def test_candidate_expires_while_a_product_waits_before_crossing(self):
        gate=self.make_gate();events=[];index=0
        positions=list(range(280,155,-7))+[160]*110+list(range(160,105,-7))
        for x in positions:
            events.extend(gate.process(self.frame(x),index*100000000)['events']);index+=1
        self.assertEqual(len(events),1)
        self.assertLessEqual(events[0]['trigger_pts_ns']-events[0]['pts_ns'],800000000)

    def test_missing_preview_frame_is_not_replaced_by_a_newer_frame(self):
        session=VideoSession(ROOT,FakeResident(),None,'','','b1_stream_512_pose_v1')
        session.frame_cache.extend([(1,b'first'),(2,b'second')]);session.jpeg=b'second'
        self.assertEqual(session.frame(1),b'first')
        self.assertIsNone(session.frame(999))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/test_tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.resident=FakeResident()
        self.service=InspectionService(self.root,self.resident);self.addCleanup(self.service.close)
        self.session=self.service.store.session('fixture','fixture')
        self.image=self.root/'input.png';textured().save(self.image)

    def test_duplicate_event_persists_once_and_calls_gpu_once(self):
        a=self.service.submit(self.image,'event',self.session,1,{})
        b=self.service.submit(self.image,'event',self.session,1,{})
        self.service.collect_once()
        self.assertEqual(a['id'],b['id']);self.assertEqual(self.resident.calls,1)
        self.assertEqual(self.service.store.get(a['id'])['decision'],'defect')
        self.assertEqual(InspectionStore(self.service.store.path).summary()['total'],1)

    def test_bad_quality_is_recorded_without_gpu_call(self):
        Image.new('RGB',(96,96),'black').save(self.image)
        row=self.service.submit(self.image,'dark',self.session,1,{})
        self.assertEqual(row['decision'],'review');self.assertEqual(self.resident.calls,0)

    def test_model_mismatch_cannot_be_recorded_as_a_class(self):
        row=self.service.submit(self.image,'wrong-model',self.session,1,{})
        with patch.object(self.resident,'result',return_value={'status':'completed','contract_sha256':'other'}):self.service.collect_once()
        self.assertEqual(self.service.store.get(row['id'])['decision'],'error')

    def test_filtered_csv_does_not_execute_track_formulas(self):
        self.service.submit(self.image,'formula',self.session,'=1+1',{})
        self.service.collect_once()
        csv=self.service.store.export_csv(session_id=self.session).decode('utf-8-sig')
        self.assertIn("'=1+1",csv)
        self.assertEqual(self.service.store.query(search="' OR 1=1 --")['total'],0)

    def test_paths_outside_project_are_rejected(self):
        with self.assertRaises(ValueError):self.service.submit(self.root.parent/'other.png','bad',self.session,1,{})

    def test_rtsp_source_redacts_password_and_query(self):
        self.assertEqual(redact_source('rtsp://user:secret@127.0.0.1:8554/camera?token=secret'),'rtsp://127.0.0.1:8554/camera')

    def test_counts_are_not_limited_by_page_size(self):
        for i in range(3):self.service.submit(self.image,'event'+str(i),self.session,i,{})
        self.service.collect_once()
        self.assertEqual(len(self.service.store.query(limit=1)['items']),1)
        self.assertEqual(sum(self.service.store.session_counts(self.session).values()),3)


class WorkspaceHTTPTests(unittest.TestCase):
    def test_photo_history_export_and_cross_origin_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/test_tmp') as folder:
            root=Path(folder)
            with patch.object(preview_web,'ROOT',root),patch.object(preview_web,'ResidentInference',FakeResident):
                handler=preview_web.make_handler(root/'preview','','')
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            try:
                data=io.BytesIO();textured().save(data,format='PNG')
                with urlopen(Request(base+'/api/photo',data=data.getvalue(),method='POST')) as response:row=json.load(response)
                for _ in range(40):
                    with urlopen(base+'/api/history/'+row['id']) as response:row=json.load(response)
                    if row['state']=='completed':break
                    time.sleep(.02)
                self.assertEqual(row['decision'],'defect')
                with urlopen(base+'/api/history?decision=defect') as response:self.assertEqual(json.load(response)['total'],1)
                with urlopen(base+'/api/history/export.csv') as response:self.assertIn(b'fixture-engine',response.read())
                with urlopen(base+row['image_url']+'?thumb=1') as response:
                    thumb=Image.open(io.BytesIO(response.read()));self.assertLessEqual(max(thumb.size),160);thumb.close()
                with self.assertRaises(HTTPError) as caught:urlopen(Request(base+'/api/video/stop',data=b'{}',headers={'Origin':'https://unrelated.example'}))
                self.assertEqual(caught.exception.code,403)
            finally:
                handler.workspace_api.close();server.shutdown();server.server_close();thread.join()

    def test_research_catalog_preserves_scopes_and_flags_placeholder_review(self):
        if not (ROOT/'straw_resume/artifacts/straw_global_local/confirmatory_kaggle_v1/report/test_runs.csv').is_file():
            self.skipTest('Research evidence is supplied in the full handover bundle')
        catalog=ResearchCatalog(ROOT).snapshot()
        self.assertEqual(len(catalog['runs']),21)
        self.assertEqual(len(catalog['summary']),7)
        self.assertFalse(catalog['human_review_verified'])
        self.assertTrue(any(f['name']=='COMPARISON_FULL_REPORT.md' for f in catalog['files']))
        self.assertTrue(all(f['group']=='legacy' for f in catalog['files'] if f['path'].startswith('submission/tables/')))


if __name__=='__main__':unittest.main()
