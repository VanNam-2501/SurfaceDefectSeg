'use strict';
importScripts('/conveyor_gate.js');
let gate;
self.onmessage=event=>{
  const message=event.data;
  if(message.kind==='calibrate'){
    gate=new self.ConveyorGate(320,138,{roi:{x:120,y:19,w:80,h:89},line:160,band:[19,108]});
    gate.calibrate(message.rgba);
    return;
  }
  if(message.kind==='analyze'){
    const analysis=gate.analyze(message.rgba);
    self.postMessage({kind:'analysis',analysis},[analysis.mask.buffer]);
  }
};
