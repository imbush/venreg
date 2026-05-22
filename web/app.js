let man=null, cur=1, SF=0.5;           // SF = work_xy/512 (slice px -> working voxel)
let st={A:{rep:'vesselness',z:0,zoom:1,px:0,py:0},
        B:{rep:'vesselness',z:0,zoom:1,px:0,py:0,overlay:'__live__',gain:1.0,pickMode:'normal'}};
let lms={}, pendingA=null, pendingMark=null;
const imgs={};
const off1=document.createElement('canvas'),o1=off1.getContext('2d');
const off2=document.createElement('canvas'),o2=off2.getContext('2d');
const $=id=>document.getElementById(id);
function getImg(src,cb){if(imgs[src])return imgs[src];const im=new Image();if(cb)im.onload=cb;im.src=src;imgs[src]=im;return im;}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}

// ---------- landmark similarity fit (client preview, with reflection) ----------
function pairsList(){return Object.keys(lms).map(Number).sort((a,b)=>a-b)
  .filter(p=>lms[p].A&&lms[p].B).map(p=>({id:p,a:lms[p].A,b:lms[p].B}));}
function fitOne(P,reflect){
  const axf=p=>reflect?-p.a.x:p.a.x;let mux=0,muy=0,mbx=0,mby=0;
  P.forEach(p=>{mux+=axf(p);muy+=p.a.y;mbx+=p.b.x;mby+=p.b.y;});const n=P.length;mux/=n;muy/=n;mbx/=n;mby/=n;
  let saa=0,sab=0,scr=0;P.forEach(p=>{const ax=axf(p)-mux,ay=p.a.y-muy,bx=p.b.x-mbx,by=p.b.y-mby;
    saa+=ax*ax+ay*ay;sab+=ax*bx+ay*by;scr+=ax*by-ay*bx;});if(saa<1e-6)return null;
  const a=sab/saa,b=scr/saa,tx=mbx-(a*mux-b*muy),ty=mby-(b*mux+a*muy);
  const m00=reflect?-a:a,m01=-b,m10=reflect?-b:b,m11=a;
  let rss=0,res=[];P.forEach(p=>{const px=m00*p.a.x+m01*p.a.y+tx,py=m10*p.a.x+m11*p.a.y+ty;
    const e=Math.hypot(px-p.b.x,py-p.b.y);res.push({id:p.id,e});rss+=e*e;});
  return {m00,m01,m10,m11,tx,ty,scale:Math.hypot(a,b),rot:Math.atan2(b,a)*180/Math.PI,reflect,rms:Math.sqrt(rss/n),n,res};
}
function fitSim(){const P=pairsList();if(P.length<2)return null;const f0=fitOne(P,false),f1=fitOne(P,true);
  if(!f0)return f1;if(!f1)return f0;return f1.rms<f0.rms?f1:f0;}
function fitZ(){const P=pairsList();if(P.length<2||new Set(P.map(p=>p.a.z)).size<2)return null;
  const n=P.length;let sx=0,sy=0,sxx=0,sxy=0;P.forEach(p=>{sx+=p.a.z;sy+=p.b.z;sxx+=p.a.z*p.a.z;sxy+=p.a.z*p.b.z;});
  const m=(n*sxy-sx*sy)/(n*sxx-sx*sx);return {m,c:(sy-m*sx)/n};}
function liveAz(bz){const zf=fitZ();if(zf&&Math.abs(zf.m)>1e-6)return Math.round((bz-zf.c)/zf.m);
  return Math.round(bz*(man.A.nz-1)/(man.B.nz-1));}

// ---------- panels ----------
function panel(v){const nz=man[v].nz;
 let ov='';
 if(v==='B'){ov=`<div class="zrow"><span style="font-size:12px">overlay:</span>
     <select id="ovB" onchange="setOverlay(this.value)" style="flex:1">
       <option value="__live__">live (rigid preview)</option><option value="">none</option>
       <option value="regA_rigid">rigid result (server)</option>
       <option value="regA_warp">warp result (server)</option></select>
     <span style="font-size:12px">A</span><input type="range" id="ovop" oninput="setGain(this.value)" min="0" max="3" step="0.05" value="1" style="width:70px"></div>
   <div class="zrow"><button class="tbtn" id="pickmode" onclick="togglePick()">overlay-pick: OFF</button>
     <span class="hint">click green(A) then red(B) on overlay → correction pair</span></div>
   <div id="fit" class="hint"></div>`;}
 return `<div class="panel" id="p${v}">
   <h2>${v} — ${man[v].name} · ${nz} slices</h2>
   <div class="reps" data-v="${v}">
     ${['fluorescence','vesselness','mask','intersection'].map(r=>`<button data-r="${r}" class="${r==='vesselness'?'on':''}">${r}</button>`).join('')}
     <button class="tbtn" onclick="resetView('${v}')">reset view</button><span class="hint" id="zoom${v}"></span></div>
   ${ov}
   <canvas id="cv${v}" width="512" height="512"></canvas>
   <div class="zrow"><button class="nav" onclick="step('${v}',-1)">◀</button>
     <input type="range" id="z${v}" min="0" max="${nz-1}" value="0">
     <button class="nav" onclick="step('${v}',1)">▶</button><span id="zl${v}">z 0/${nz-1}</span></div>
 </div>`;}
function srcFor(v,rep,z){return `slices/${v}/${rep}/z${String(z).padStart(3,'0')}.png`;}

function draw(v){
 const s=st[v],ctx=$('cv'+v).getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,512,512);
 $('zoom'+v).textContent=`zoom ${s.zoom.toFixed(1)}x`;
 const base=getImg(srcFor(v,s.rep,s.z),()=>draw(v));
 const overlayOn=v==='B'&&s.overlay;
 if(!overlayOn){if(base.complete){ctx.setTransform(s.zoom,0,0,s.zoom,s.px,s.py);ctx.drawImage(base,0,0,512,512);}
   ctx.setTransform(1,0,0,1,0,0);drawMarks(v,ctx);return;}
 if(!base.complete)return;
 off1.width=off1.height=512;o1.setTransform(s.zoom,0,0,s.zoom,s.px,s.py);o1.clearRect(-9999,-9999,99999,99999);o1.drawImage(base,0,0,512,512);
 const Bd=o1.getImageData(0,0,512,512).data;off2.width=off2.height=512;o2.setTransform(1,0,0,1,0,0);o2.clearRect(0,0,512,512);
 let Ad=null,info='';
 if(s.overlay==='__live__'){const F=fitSim();
   if(F){const aZ=Math.max(0,Math.min(man.A.nz-1,liveAz(s.z)));const A=getImg(srcFor('A',s.rep,aZ),()=>draw('B'));
     if(A.complete){o2.setTransform(s.zoom*F.m00,s.zoom*F.m10,s.zoom*F.m01,s.zoom*F.m11,s.zoom*F.tx/SF+s.px,s.zoom*F.ty/SF+s.py);
       o2.drawImage(A,0,0,512,512);Ad=o2.getImageData(0,0,512,512).data;}
     const ok=F.rms<6;info=`live preview: ${F.n} pairs · ${F.reflect?'<b>FLIP</b>+':''}scale ${F.scale.toFixed(3)} · rot ${F.rot.toFixed(1)}° · RMS <b class="${ok?'good':'bad'}">${F.rms.toFixed(1)}px</b>`;
   } else info='live: need ≥2 pairs';
 } else {const A=getImg(`slices/${s.overlay}/${s.rep}/z${String(s.z).padStart(3,'0')}.png`,()=>draw('B'));
   if(A.complete){o2.setTransform(s.zoom,0,0,s.zoom,s.px,s.py);o2.drawImage(A,0,0,512,512);Ad=o2.getImageData(0,0,512,512).data;}
   info=`${s.overlay} (server result)`;}
 const out=ctx.createImageData(512,512),o=out.data,g=s.gain;
 for(let i=0;i<512*512;i++){o[i*4]=Bd[i*4];o[i*4+1]=Ad?Math.min(255,Ad[i*4]*g):0;o[i*4+2]=0;o[i*4+3]=255;}
 ctx.setTransform(1,0,0,1,0,0);ctx.putImageData(out,0,0);drawMarks(v,ctx);$('fit').innerHTML=info;
}
function drawMarks(v,ctx){const s=st[v];
 if(v==='B'&&pendingMark){const cx=s.zoom*(pendingMark.x/SF)+s.px,cy=s.zoom*(pendingMark.y/SF)+s.py;
   ctx.strokeStyle='#ff0';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,10,0,7);ctx.stroke();ctx.fillStyle='#ff0';ctx.fillText('pick B',cx+12,cy);}
 for(const p in lms){const e=lms[p][v];if(!e)continue;const near=Math.abs(e.z-s.z)<=1;
   const cx=s.zoom*(e.x/SF)+s.px,cy=s.zoom*(e.y/SF)+s.py;ctx.lineWidth=2;ctx.strokeStyle=v==='A'?'#6f6':'#f88';ctx.globalAlpha=near?1:0.3;
   ctx.beginPath();ctx.arc(cx,cy,8,0,7);ctx.stroke();ctx.beginPath();ctx.moveTo(cx-12,cy);ctx.lineTo(cx+12,cy);ctx.moveTo(cx,cy-12);ctx.lineTo(cx,cy+12);ctx.stroke();
   ctx.fillStyle=ctx.strokeStyle;ctx.fillText('#'+p,cx+9,cy-9);ctx.globalAlpha=1;}}
function step(v,d){const nz=man[v].nz;st[v].z=Math.max(0,Math.min(nz-1,st[v].z+d));$('z'+v).value=st[v].z;$('zl'+v).textContent=`z ${st[v].z}/${nz-1}`;draw(v);}
function resetView(v){st[v].zoom=1;st[v].px=0;st[v].py=0;draw(v);}
function setOverlay(val){st.B.overlay=val;draw('B');}
function setGain(val){st.B.gain=+val;draw('B');}
function togglePick(){st.B.pickMode=st.B.pickMode==='overlay'?'normal':'overlay';pendingA=null;pendingMark=null;
  $('pickmode').textContent='overlay-pick: '+(st.B.pickMode==='overlay'?'ON':'OFF');$('pickmode').style.background=st.B.pickMode==='overlay'?'#2a7':'#333';draw('B');}

function canvasXY(v,ev){const cv=$('cv'+v),r=cv.getBoundingClientRect();return [(ev.clientX-r.left)/r.width*512,(ev.clientY-r.top)/r.height*512];}
function toWork(v,cx,cy){const s=st[v];return [((cx-s.px)/s.zoom)*SF,((cy-s.py)/s.zoom)*SF];}
function setupCanvas(v){const cv=$('cv'+v);let down=false,moved=false,sx,sy,opx,opy;
 cv.addEventListener('wheel',e=>{e.preventDefault();const s=st[v];const[cx,cy]=canvasXY(v,e);const f=e.deltaY<0?1.15:1/1.15;
   const nz=Math.max(1,Math.min(20,s.zoom*f));s.px=cx-(cx-s.px)*(nz/s.zoom);s.py=cy-(cy-s.py)*(nz/s.zoom);s.zoom=nz;draw(v);},{passive:false});
 cv.addEventListener('mousedown',e=>{down=true;moved=false;[sx,sy]=canvasXY(v,e);opx=st[v].px;opy=st[v].py;});
 window.addEventListener('mousemove',e=>{if(!down)return;const[cx,cy]=canvasXY(v,e);if(Math.hypot(cx-sx,cy-sy)>3){moved=true;st[v].px=opx+(cx-sx);st[v].py=opy+(cy-sy);draw(v);}});
 window.addEventListener('mouseup',e=>{if(!down)return;down=false;if(!moved){const[cx,cy]=canvasXY(v,e);const[fx,fy]=toWork(v,cx,cy);
   if(fx>=0&&fx<man.work_xy&&fy>=0&&fy<man.work_xy)placeLandmark(v,fx,fy);}});}
function placeLandmark(v,fx,fy){
 if(v==='B'&&st.B.pickMode==='overlay'){const F=fitSim();if(!F){alert('Add ≥2 normal pairs first.');return;}
   if(pendingA===null){const det=F.m00*F.m11-F.m01*F.m10;const ix=(F.m11*(fx-F.tx)-F.m01*(fy-F.ty))/det;const iy=(-F.m10*(fx-F.tx)+F.m00*(fy-F.ty))/det;
     pendingA={x:Math.round(ix),y:Math.round(iy),z:liveAz(st.B.z)};pendingMark={x:fx,y:fy};draw('B');}
   else{if(!lms[cur])lms[cur]={};lms[cur].A=pendingA;lms[cur].B={x:Math.round(fx),y:Math.round(fy),z:st.B.z};pendingA=null;pendingMark=null;renderTable();newPair();draw('A');draw('B');}
   return;}
 if(!lms[cur])lms[cur]={};lms[cur][v]={x:Math.round(fx),y:Math.round(fy),z:st[v].z};draw('A');draw('B');renderTable();}

// ---------- pairs / table ----------
function newPair(){const ids=Object.keys(lms).map(Number);cur=(ids.length?Math.max(...ids):0)+1;$('curpair').textContent=cur;}
function setPair(p){cur=p;$('curpair').textContent=cur;}
function delPair(p){delete lms[p];renderTable();draw('A');draw('B');}
function clearAll(){if(confirm('Clear all landmarks?')){lms={};renderTable();draw('A');draw('B');}}
function renderTable(){const tb=document.querySelector('#lmtab tbody');tb.innerHTML='';const F=fitSim();const r={};if(F)F.res.forEach(x=>r[x.id]=x.e);
 Object.keys(lms).map(Number).sort((a,b)=>a-b).forEach(p=>{const e=lms[p],a=e.A,b=e.B;const tr=document.createElement('tr');
   tr.innerHTML=`<td><a href="#" onclick="setPair(${p});return false">#${p}</a></td><td class="a">${a?`${a.x},${a.y},${a.z}`:'—'}</td>
     <td class="b">${b?`${b.x},${b.y},${b.z}`:'—'}</td><td>${r[p]!=null?r[p].toFixed(1):'—'}</td>
     <td><button class="danger" style="padding:1px 6px" onclick="delPair(${p})">×</button></td>`;tb.appendChild(tr);});
 if(st.B&&st.B.overlay==='__live__'&&man)draw('B');}

function lmArray(){return pairsList().map(p=>[p.a.x,p.a.y,p.a.z,p.b.x,p.b.y,p.b.z]);}
async function generate(mode){const P=lmArray();if(P.length<2){alert('Need ≥2 complete pairs.');return;}
 $('regstatus').textContent=`generating ${mode}… (warp can take ~1 min)`;
 const r=await post('/api/register',{landmarks:P,mode});
 if(r.error){$('regstatus').textContent='error: '+r.error;return;}
 Object.keys(imgs).forEach(k=>{if(k.includes('regA_'+mode))delete imgs[k];}); // bust cache
 $('ovB').value='regA_'+mode;st.B.overlay='regA_'+mode;draw('B');
 $('regstatus').innerHTML=`${mode} done · ${r.reflect?'FLIP · ':''}${r.scale?'scale '+r.scale.toFixed(3)+' · ':''}B→A oriented-inlier <b>${r.BtoA_oriented_inlier}</b> (chance ~0.10) · showing it now`;}
async function save(what){const r=await post('/api/save',{what,mode:$('savemode').value,landmarks:lmArray()});
 $('savestatus').textContent=r.error?('error: '+r.error):('saved → '+r.saved);}
async function loadExampleLm(){const r=await fetch('example_landmarks.json').catch(()=>null);
 if(!r||!r.ok){alert('no example landmarks bundled');return;}const d=await r.json();lms={};Object.assign(lms,d);newPair();renderTable();draw('A');draw('B');}

// ---------- load flow ----------
async function init(){
 const lst=await (await fetch('/api/list')).json();
 const opt=lst.files.map(f=>`<option value="${f}">${f}</option>`).join('');
 $('fileA').innerHTML=opt;$('fileB').innerHTML=opt;if(lst.files[1])$('fileB').selectedIndex=1;
 $('loadbtn').onclick=loadPair;}
async function loadPair(){
 $('loadstatus').textContent='loading & preprocessing… (~10–30 s)';
 const r=await post('/api/load',{fileA:$('fileA').value,fileB:$('fileB').value});
 if(r.error){$('loadstatus').textContent='error: '+r.error;return;}
 man=r;SF=man.work_xy/512;Object.keys(imgs).forEach(k=>delete imgs[k]);
 $('wrap').innerHTML=panel('A')+panel('B');$('controls').style.display='';
 ['A','B'].forEach(v=>{document.querySelector(`#p${v} .reps`).addEventListener('click',e=>{if(e.target.dataset.r){st[v].rep=e.target.dataset.r;
     document.querySelectorAll(`#p${v} .reps button`).forEach(b=>b.classList.toggle('on',b===e.target));draw(v);}});
   $('z'+v).addEventListener('input',e=>{st[v].z=+e.target.value;$('zl'+v).textContent=`z ${st[v].z}/${man[v].nz-1}`;draw(v);});
   setupCanvas(v);draw(v);});
 window.onkeydown=e=>{if(e.key==='ArrowUp')step('B',-1);if(e.key==='ArrowDown')step('B',1);};
 $('loadstatus').textContent='loaded ✓';renderTable();}
init();
