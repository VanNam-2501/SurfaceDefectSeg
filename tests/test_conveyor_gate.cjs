const assert=require('node:assert/strict');
const {ConveyorGate}=require('../apps/stream_demo/conveyor_gate.js');
const w=320,h=138;
function frame(x=null,brightness=0,noise=false){const a=new Uint8ClampedArray(w*h*4);for(let y=0;y<h;y++)for(let i=0;i<w;i++){const k=(y*w+i)*4;let v=120+brightness;if(x!==null&&i>=x&&i<x+40&&y>=32&&y<95)v=220+brightness;if(noise&&i===20&&y===45)v=255;a[k]=a[k+1]=a[k+2]=v;a[k+3]=255;}return a;}
const gate=new ConveyorGate(w,h);gate.calibrate(frame());
for(let i=0;i<60;i++)assert.equal(gate.analyze(frame(null,0,true)).events.length,0,'empty/noisy belt must not call AI');
for(let i=0;i<10;i++)assert.equal(gate.analyze(frame(null,25)).events.length,0,'uniform lighting change must not trigger');
let total=0,event;
for(let x=260;x>=-45;x-=3){const result=gate.analyze(frame(x));if(x+20>160)assert.equal(result.events.length,0,'AI cannot fire before ROI crossing');for(const e of result.events){event=e;total++;}}
assert.equal(total,1,'one moving product must produce one event');assert.equal(event.gate.confirmation_window,5);assert.ok(event.gate.confirmed_frames>=4);assert.ok(event.gate.roi_overlap>=.8);
const tolerant=new ConveyorGate(w,h);tolerant.calibrate(frame());let gapEvents=0,index=0;
for(let x=260;x>=-45;x-=3,index++)if(index!==8)gapEvents+=tolerant.analyze(frame(x)).events.length;else tolerant.analyze(frame());
assert.equal(gapEvents,1,'one missed mask frame is tolerated inside the five-frame vote');
for(let i=0;i<10;i++)gate.analyze(frame());
let second=0;for(let x=260;x>=-45;x-=3)second+=gate.analyze(frame(x)).events.length;
assert.equal(second,1,'a later product must be independently inspected');
const stopped=new ConveyorGate(w,h);stopped.calibrate(frame());let count=0;for(let x=260;x>=139;x-=3)count+=stopped.analyze(frame(x)).events.length;for(let i=0;i<100;i++)count+=stopped.analyze(frame(139)).events.length;assert.equal(count,1,'stationary product at inspection station must not retrigger');
console.log('PASS: idle/noise, brightness, pre-ROI suppression, one call/product, rearm, stationary deduplication');
