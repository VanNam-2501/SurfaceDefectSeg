/* Pixel-only gate: robust background subtraction -> components -> tracks -> ROI crossing.
 * It never receives product positions, filenames, class labels or model outputs.
 */
(function(root){
'use strict';
class ConveyorGate {
  constructor(width,height,options={}){
    this.w=width; this.h=height;
    this.roi=options.roi||{x:width*.38,y:height*.12,w:width*.24,h:height*.7};
    this.line=options.line??Math.round(width*.5);
    this.band=options.band||[Math.floor(height*.12),Math.floor(height*.8)];
    // Grayscale 0–255. This is a minimum signal floor, with lighting offset
    // estimated robustly from a background-only strip on every frame.
    this.threshold=options.threshold??18;
    this.minArea=options.minArea??120;
    this.minWidth=options.minWidth??12;
    this.minHeight=options.minHeight??20;
    this.minFillRatio=options.minFillRatio??.45;
    this.trackDistance=options.trackDistance??18;
    this.confirmationWindow=options.confirmationWindow??5;
    this.minConfirmedFrames=options.minConfirmedFrames??4;
    this.minRoiOverlap=options.minRoiOverlap??.8;
    this.maxMissedFrames=options.maxMissedFrames??1;
    this.tracks=[]; this.nextId=1; this.frames=0;
    this.background=null; this.previous=null;
  }
  gray(rgba){
    const gray=new Float32Array(this.w*this.h);
    for(let i=0;i<gray.length;i++)gray[i]=.299*rgba[i*4]+.587*rgba[i*4+1]+.114*rgba[i*4+2];
    return gray;
  }
  calibrate(rgba){this.background=this.gray(rgba);this.previous=this.background.slice();this.tracks=[];}
  close(mask){
    const w=this.w,h=this.h,d=new Uint8Array(mask.length),e=new Uint8Array(mask.length);
    for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
      let on=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)on|=mask[(y+dy)*w+x+dx];d[y*w+x]=on;
    }
    for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
      let on=1;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)on&=d[(y+dy)*w+x+dx];e[y*w+x]=on;
    }
    return e;
  }
  components(mask){
    const seen=new Uint8Array(mask.length),result=[],stack=new Int32Array(mask.length),w=this.w,h=this.h,r=this.roi;
    for(let start=0;start<mask.length;start++){
      if(!mask[start]||seen[start])continue;
      let top=0,area=0,roiPixels=0,x0=w,x1=0,y0=h,y1=0;
      stack[top++]=start;seen[start]=1;
      while(top){
        const i=stack[--top],x=i%w,y=Math.floor(i/w);area++;
        if(x>=r.x&&x<r.x+r.w&&y>=r.y&&y<r.y+r.h)roiPixels++;
        x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);
        for(const n of [x>0?i-1:-1,x<w-1?i+1:-1,y>0?i-w:-1,y<h-1?i+w:-1])
          if(n>=0&&mask[n]&&!seen[n]){seen[n]=1;stack[top++]=n;}
      }
      const bw=x1-x0+1,bh=y1-y0+1,fill=area/(bw*bh);
      if(area>=this.minArea&&bw>=this.minWidth&&bh>=this.minHeight&&fill>=this.minFillRatio)
        result.push({x:x0,y:y0,w:bw,h:bh,area,cx:(x0+x1)/2,cy:(y0+y1)/2,fill,roiOverlap:roiPixels/area});
    }
    return result;
  }
  analyze(rgba){
    if(!this.background)throw new Error('Background must be calibrated from an empty scene');
    const gray=this.gray(rgba),raw=new Uint8Array(gray.length),reference=[];
    // Median offset is less sensitive than a mean to a product edge or bright speck.
    for(let y=2;y<Math.min(10,this.h);y++)for(let x=2;x<this.w-2;x+=4){
      const i=y*this.w+x;reference.push(gray[i]-this.background[i]);
    }
    reference.sort((a,b)=>a-b);
    const middle=Math.floor(reference.length/2),brightness=reference.length
      ?(reference.length%2?reference[middle]:(reference[middle-1]+reference[middle])/2):0;
    let moving=0,foreground=0,roiCount=0,roiForeground=0;
    const roi=this.roi;
    for(let y=this.band[0];y<this.band[1];y++)for(let x=0;x<this.w;x++){
      const i=y*this.w+x,difference=Math.abs(gray[i]-this.background[i]-brightness);
      raw[i]=difference>this.threshold?1:0;foreground+=raw[i];
      if(Math.abs(gray[i]-this.previous[i])>12)moving++;
      if(x>=roi.x&&x<roi.x+roi.w&&y>=roi.y&&y<roi.y+roi.h){roiCount++;roiForeground+=raw[i];}
    }
    const mask=this.close(raw),blobs=this.components(mask),events=[];
    this.previous=gray;this.frames++;
    for(const track of this.tracks){
      track.missed++;
      track.hitHistory.push(0);
      if(track.hitHistory.length>this.confirmationWindow)track.hitHistory.shift();
    }
    const assigned=new Set();
    for(const blob of blobs){
      let track=null,best=this.trackDistance;
      for(const candidate of this.tracks){
        if(assigned.has(candidate.id))continue;
        const predictedX=candidate.box.cx+candidate.vx,predictedY=candidate.box.cy+candidate.vy;
        const distance=Math.hypot(predictedX-blob.cx,predictedY-blob.cy);
        if(distance<best){track=candidate;best=distance;}
      }
      if(!track){
        track={id:this.nextId++,box:blob,vx:0,vy:0,observations:0,hitHistory:Array(this.confirmationWindow).fill(0),
          triggered:false,missed:0,overlap:blob.roiOverlap};
        this.tracks.push(track);
      }
      assigned.add(track.id);
      const previousX=track.box.cx,previousY=track.box.cy,dx=blob.cx-previousX,dy=blob.cy-previousY;
      track.vx=.5*track.vx+.5*dx;track.vy=.5*track.vy+.5*dy;
      track.box=blob;track.observations++;track.missed=0;
      track.hitHistory[track.hitHistory.length-1]=1;
      track.confirmedFrames=track.hitHistory.reduce((sum,hit)=>sum+hit,0);
      track.overlap=blob.roiOverlap;
      const crossed=previousX>this.line&&blob.cx<=this.line&&dx<0;
      track.crossed=crossed;
      track.state=track.triggered?'PASSED':track.confirmedFrames>=this.minConfirmedFrames?'CONFIRMED':'CANDIDATE';
      if(!track.triggered&&crossed&&track.hitHistory.length===this.confirmationWindow
          &&track.confirmedFrames>=this.minConfirmedFrames&&track.overlap>=this.minRoiOverlap){
        track.triggered=true;track.state='TRIGGERED';
        events.push({track_id:track.id,box:blob,gate:{crossed:true,observations:track.observations,
          confirmed_frames:track.confirmedFrames,confirmation_window:this.confirmationWindow,
          roi_overlap:track.overlap,center_x:blob.cx/this.w,frame_number:this.frames,
          foreground_pixels:blob.area,motion_pixels:moving}});
      }
    }
    this.tracks=this.tracks.filter(track=>track.missed<=this.maxMissedFrames);
    return {events,tracks:this.tracks.filter(track=>track.missed===0),mask,frames:this.frames,
      motionRatio:moving/(this.w*(this.band[1]-this.band[0])),
      foregroundRatio:foreground/(this.w*(this.band[1]-this.band[0])),
      roiOccupancy:roiForeground/(roiCount||1),brightnessShift:brightness};
  }
}
if(typeof module!=='undefined'&&module.exports)module.exports={ConveyorGate};else root.ConveyorGate=ConveyorGate;
})(typeof window!=='undefined'?window:globalThis);
