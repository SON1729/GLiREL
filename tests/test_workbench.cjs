'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.join(__dirname,'..'),data=JSON.parse(fs.readFileSync(path.join(root,'data/recorded.json'),'utf8'));
const els=new Map(),events={},messages=[],intervals=new Map();let uid=0;
const defaults={q1:1,q2:.5,countN:3,countM:5};
const el=id=>{if(!els.has(id))els.set(id,{innerHTML:'',textContent:'',value:defaults[id]??'',disabled:false,files:[],classList:{add(){},toggle(){}},getBoundingClientRect:()=>({height:900})});return els.get(id)};
const parent={postMessage:m=>messages.push(m)};
const ctx=vm.createContext({console,JSON,Math,Map,Set,Number,String,Array,Error,crypto:{randomUUID:()=>`request-${++uid}`},
 document:{getElementById:el,querySelectorAll:()=>[],addEventListener:(n,fn)=>events[n]=fn,body:{}},window:{parent,addEventListener:(n,fn)=>events['window:'+n]=fn},
 ResizeObserver:class{observe(){}},requestAnimationFrame:fn=>fn(),setInterval:fn=>{const id=++uid;intervals.set(id,fn);return id},clearInterval:id=>intervals.delete(id),setTimeout:()=>{},URL:{},Blob:class{}});
vm.runInContext(fs.readFileSync(path.join(root,'frontend/algorithm.js'),'utf8'),ctx);
const run=s=>vm.runInContext(s,ctx);
function render(b=data,response=null,version='v1'){events['window:message']({source:parent,data:{type:'streamlit:render',args:{bundle:b,library:data,backend:'unconfigured',response,version}}})}
function click(act,extra={}){const button={dataset:{act,...extra},disabled:false};events.click({target:{closest:()=>button}})}
render();assert.equal(messages[0].type,'streamlit:componentReady');assert(el('editor').innerHTML.includes('Chọn thực thể'));
assert.deepEqual(Array.from(ctx.window.GLIREL_TEST.tokenize("Marie Curie was born in Warsaw.")),['Marie','Curie','was','born','in','Warsaw','.']);
run('S.k=1');assert.equal(ctx.window.GLIREL_TEST.selection(0).length,2);
run('S.k=-1');assert.equal(ctx.window.GLIREL_TEST.selection(0).length,6);
run('S.k=1');
for(let c=0;c<5;c++)for(const tab of ['flow','results','compare','math'])for(let step=0;step<(tab==='flow'?8:1);step++){
 run(`loadCase(S.library.cases[${c}],S.library);S.tab=${JSON.stringify(tab)};S.step=${step};render()`);
 assert(!el('view').innerHTML.includes('NaN'));assert(el('view').innerHTML.length>150);
}
// Sentence editing immediately invalidates every previous score.
run('loadCase(S.library.cases[0],S.library);S.tab="flow";render()');el('sentence').value='Marie Curie was born in Warsaw.';
events.input({target:{id:'sentence',value:el('sentence').value}});
assert.equal(run('candidates().length'),0);assert.equal(run('S.textChanged'),true);assert(el('view').innerHTML.includes('Câu đang được sửa'));
click('tokenize');assert.equal(run('S.example.entities.length'),0);
click('word',{i:'0'});click('word',{i:'1'});click('add-entity');
click('word',{i:'5'});click('word',{i:'5'});click('add-entity');
assert.equal(run('entity(0)'),'Marie Curie');assert.equal(run('entity(1)'),'Warsaw');
el('new-label').value='place of birth';click('add-label');
el('sentence').value=run('S.raw');click('run');
assert.equal(run('S.busy'),true);
const request=messages.findLast(x=>x.type==='streamlit:setComponentValue');assert.equal(request.value.action,'predict');
assert.equal(request.value.example.tokens[5],'Warsaw');
render(data,{id:request.value.id,status:'error',message:'Backend chưa cấu hình'});
assert.equal(run('S.busy'),false);assert(el('editor').innerHTML.includes('Backend chưa cấu hình'));assert.equal(run('candidates().length'),0);
// A known-good inference reply replaces the record, using its unique request id.
run('S.busy=true;S.requestId="success";S.annotated=false');
render(data,{id:'success',status:'ok'},'v2');assert.equal(run('S.busy'),false);assert(run('candidates().length')>0);assert.equal(run('S.annotated'),false);
click('pin');assert.equal(run('S.pins.length'),1);
click('step',{i:'6'});run('S.tab="flow";renderView()');click('point',{h:'0',t:'1',label:'employer'});
assert(el('view').innerHTML.includes('logit suy ngược'));click('gold-point');assert.equal(run('S.annotated'),true);
// Scores at the threshold are excluded; top-k is per directed pair.
run('S.threshold=candidates()[0].score');assert(!run('selection().some(x=>x.score===S.threshold)'));
// Playback terminates at step 8, instead of looping or accumulating timers.
run('S.step=0;S.tab="flow"');click('play');for(let i=0;i<8;i++)for(const fn of [...intervals.values()])fn();
assert.equal(run('S.step'),7);assert.equal(run('S.playing'),false);assert.equal(intervals.size,0);
// HTML escaping and malformed import handling.
const bad=structuredClone(data);bad.cases[0].candidates_before_display_thresholds[0].score=Infinity;assert.throws(()=>ctx.window.GLIREL_TEST.normalize(bad));
const escaped=structuredClone(data);escaped.cases[0].example.tokens[0]='<img onerror=alert(1)>';
render(escaped,null,'v3');assert(!el('view').innerHTML.includes('<img'));assert(el('view').innerHTML.includes('&lt;img'));
console.log('PASS: workbench renders; every case/stage/view; English tokenization; entity click selection; stale-score invalidation; inference request/reply/error; comparison; gold; threshold; playback; escaping.');
