'use strict';
// Self-contained charts: no CDN, simulated values, or inference calls.
if(location.pathname==='/performance'){
 const ns='http://www.w3.org/2000/svg', byId=id=>document.getElementById(id);
 let range=300, lastData=null, timer=null;
 const fmt=(value,digits=0)=>value!==null&&value!==undefined&&Number.isFinite(Number(value))?Number(value).toFixed(digits):'—';
 const text=(id,value)=>{byId(id).textContent=value;};
 const node=(tag,attrs={},value)=>{const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(value!==undefined)n.textContent=value;return n;};
 function chart(id,samples,series,unit,minimum){
  const svg=byId(id),w=600,h=215,left=52,right=18,top=18,bottom=34;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();
  const values=samples.flatMap(p=>series.map(s=>p[s.key])).filter(v=>v!==null&&v!==undefined&&Number.isFinite(v));
  if(!values.length){svg.append(node('text',{x:w/2,y:h/2,'text-anchor':'middle',fill:'#8294a8','font-size':13},'Chưa có mẫu đo'));return;}
  const max=Math.max(minimum||1,...values)*1.15,first=samples[0].timestamp,last=samples.at(-1).timestamp,span=Math.max(1,last-first);
  const x=t=>left+(t-first)/span*(w-left-right),y=v=>h-bottom-v/max*(h-top-bottom);
  for(let i=0;i<=3;i++){const value=max*i/3,py=y(value);svg.append(node('line',{x1:left,x2:w-right,y1:py,y2:py,stroke:'#e4edf5'}),node('text',{x:left-9,y:py+4,'text-anchor':'end',fill:'#7a8fa4','font-size':10},fmt(value,value<10?1:0)));}
  svg.append(node('text',{x:left,y:11,fill:'#7a8fa4','font-size':10},unit));
  for(const time of [...new Set([first,first+span/2,last])]){svg.append(node('text',{x:x(time),y:h-8,'text-anchor':'middle',fill:'#7a8fa4','font-size':10},new Date(time*1000).toLocaleTimeString('vi-VN',{hour12:false})));}
  for(const s of series){let d='',open=false;for(const p of samples){const value=p[s.key];if(value===null||value===undefined||!Number.isFinite(value)){open=false;continue;}d+=(open?' L':' M')+x(p.timestamp).toFixed(1)+','+y(value).toFixed(1);open=true;}const path=node('path',{d,fill:'none',stroke:s.color,'stroke-width':2.5,'stroke-linejoin':'round','stroke-linecap':'round'});if(s.dash)path.setAttribute('stroke-dasharray',s.dash);svg.append(path);const valid=samples.filter(p=>p[s.key]!==null&&p[s.key]!==undefined&&Number.isFinite(p[s.key]));if(valid.length){const p=valid.at(-1);const dot=node('circle',{cx:x(p.timestamp),cy:y(p[s.key]),r:3.5,fill:s.color});dot.append(node('title',{},`${s.name}: ${fmt(p[s.key],1)} ${unit}`));svg.append(dot);}}
 }
 function render(data){
  const all=data.samples||[],last=all.at(-1),cutoff=(last?.timestamp||Date.now()/1000)-range,samples=all.filter(p=>p.timestamp>=cutoff);
  const names={idle:'Chưa mở nguồn',starting:'Đang mở nguồn',calibrating:'Đang hiệu chuẩn',running:'Đang chạy',completed:'Phiên đã kết thúc',stopped:'Phiên đã dừng',error:'Nguồn cần kiểm tra'};
  text('performanceStatus',data.error?'Không lấy được mẫu mới':(names[data.status]||data.status)+' · cập nhật mỗi giây');
  text('performanceSession',data.session_id?`${data.source} · phiên ${data.session_id.slice(0,10)} · ${samples.length} mẫu`:'RAM được đo từ lúc mở ứng dụng. Bắt đầu một phiên băng chuyền để xem CPU, GPU và tracking.');
  if(data.error){text('performanceError',data.error);byId('performanceError').hidden=false;}else byId('performanceError').hidden=true;
  if(last){text('cpuSteps',last.analysis_steps||0);const cpu=all.filter(p=>p.cpu_frame_ms!==null&&p.cpu_frame_ms!==undefined).at(-1);text('cpuTiming',`Mẫu frame gần nhất: ${fmt(cpu?.cpu_frame_ms,1)} ms · ${last.frames_received} frame nguồn · CPU tiến trình ${fmt(last.cpu_one_core_percent,1)}%`);text('gpuCalls',last.gpu_submitted);text('gpuCallDetail',`${last.gpu_completed} hoàn tất · ${last.gpu_pending} đang chờ · ${last.gpu_failed} lỗi`);text('processRam',fmt(last.ram_mib,1)+' MiB');text('activeTracks',last.active_tracks);text('peakTracks',`${last.active_tracks} đang quan sát · ${last.predicted_tracks} dự báo · cao nhất trong cửa sổ ${Math.max(0,...samples.map(p=>p.active_tracks))}`);}
  chart('cpuChart',samples,[{key:'cpu_frame_ms',name:'CPU / frame',color:'#377fd0'}],'ms',10);
  chart('gpuChart',samples,[{key:'gpu_submitted',name:'Đã gửi',color:'#377fd0',dash:'5 4'},{key:'gpu_completed',name:'Hoàn tất',color:'#219b80'}],'ảnh',3);
  chart('ramChart',samples,[{key:'ram_mib',name:'RAM tiến trình',color:'#8a68c9'}],'MiB',32);
  chart('tracksChart',samples,[{key:'active_tracks',name:'Quan sát',color:'#219b80'},{key:'predicted_tracks',name:'Dự báo',color:'#d49d35',dash:'5 4'}],'track',3);
 }
 document.querySelectorAll('[data-performance-range]').forEach(button=>button.onclick=()=>{range=Number(button.dataset.performanceRange);document.querySelectorAll('[data-performance-range]').forEach(b=>b.classList.toggle('selected',b===button));if(lastData)render(lastData);});
 async function refresh(){try{const response=await fetch('/api/performance',{cache:'no-store'});if(!response.ok)throw new Error('Không đọc được số liệu hiệu năng.');lastData=await response.json();render(lastData);}catch(error){text('performanceStatus','Mất kết nối · đang thử lại');text('performanceError',error.message);byId('performanceError').hidden=false;}finally{timer=setTimeout(refresh,1000);}}
 window.addEventListener('pagehide',()=>clearTimeout(timer));refresh();
}
