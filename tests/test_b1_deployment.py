"""Check preprocessing equivalence and API busy/size guards without real GPU work."""
import asyncio
import io
from pathlib import Path
import sys
import unittest
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.straw_global_local.data import render_arrays,tensor_views
from src.straw_global_local.deployment import normalize,image_array,decode,softmax
from apps.web_demo.backend.b1_app import app,predict
from fastapi import HTTPException


class DeploymentTests(unittest.TestCase):
    def test_normalization_matches_source_b1(self):
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(93,157,3),dtype=np.uint8))
        expected=tensor_views(render_arrays(image,'B1'),'B1')['global'][None].numpy()
        actual=normalize(image_array(image,512))
        np.testing.assert_allclose(actual,expected,atol=3e-7,rtol=1e-6)

    def test_softmax_extreme_logits_remains_finite(self):
        p=softmax([[10000,0,-10000,3,1]])
        self.assertTrue(np.isfinite(p).all());self.assertAlmostEqual(float(p.sum()),1.)

    def test_bad_input_rejected(self):
        with self.assertRaises(Exception):decode(b'not-an-image')

    def test_cancelled_request_does_not_free_busy_gpu(self):
        class Upload:
            async def read(self,size):return b'image'
        def slow(raw):time.sleep(.08);return {'timings_ms':{}}
        async def scenario():
            app.state.worker=ThreadPoolExecutor(max_workers=1)
            app.state.busy=asyncio.Lock();app.state.inflight=None
            request=SimpleNamespace(state=SimpleNamespace(started=time.perf_counter()))
            with patch('apps.web_demo.backend.b1_app.inspect',slow):
                task=asyncio.create_task(predict(request,Upload(),1))
                await asyncio.sleep(.01);task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                with self.assertRaises(HTTPException) as caught:await predict(request,Upload(),2)
                self.assertEqual(caught.exception.status_code,429)
                await app.state.inflight
            app.state.worker.shutdown(wait=True)
        asyncio.run(scenario())

    def test_only_one_request_can_wait(self):
        class Upload:
            async def read(self,size):return b'image'
        def slow(raw):time.sleep(.02);return {'timings_ms':{}}
        async def scenario():
            app.state.worker=ThreadPoolExecutor(max_workers=1)
            app.state.busy=asyncio.Lock();app.state.inflight=None;app.state.admitted=0
            request=SimpleNamespace(state=SimpleNamespace(started=time.perf_counter()))
            with patch('apps.web_demo.backend.b1_app.inspect',slow):
                results=await asyncio.gather(*(predict(request,Upload(),i) for i in range(3)),return_exceptions=True)
                self.assertEqual(sum(isinstance(r,HTTPException) and r.status_code==429 for r in results),1)
                self.assertEqual(sum(isinstance(r,dict) for r in results),2)
                self.assertEqual(app.state.admitted,0)
            app.state.worker.shutdown(wait=True)
        asyncio.run(scenario())

    def test_api_guards_and_frame_association(self):
        class Runtime:
            def run(self,x):return np.array([[0.,10.,1.,2.,3.]],np.float32)
        app.state.runtime=Runtime(); app.state.config={'size':512,'temperature':1.}
        app.state.worker=ThreadPoolExecutor(max_workers=1)
        app.state.admitted=0;app.state.inflight=None
        app.state.busy=asyncio.Lock()
        client=TestClient(app)
        stream=io.BytesIO();Image.new('RGB',(32,32),'white').save(stream,'JPEG')
        result=client.post('/predict',files={'file':('image.jpg',stream.getvalue(),'image/jpeg')},data={'frame_id':19})
        self.assertEqual(result.status_code,200);self.assertEqual(result.json()['frame_id'],19)
        self.assertEqual(result.json()['class'],'bent')
        bad=client.post('/predict',files={'file':('bad.jpg',b'bad','image/jpeg')})
        self.assertEqual(bad.status_code,422)
        class Busy:
            def locked(self):return True
        app.state.busy=Busy()
        app.state.admitted=2
        response=client.post('/predict',files={'file':('image.jpg',stream.getvalue(),'image/jpeg')})
        self.assertEqual(response.status_code,429)
        app.state.admitted=0
        app.state.worker.shutdown(wait=True)


if __name__=='__main__':unittest.main()
