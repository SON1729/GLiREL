'use strict';
// Tests the component's data/decision/render logic without running a browser.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const root=path.join(__dirname,'..');
const source=fs.readFileSync(path.join(root,'frontend','app.js'),'utf8');
const data=JSON.parse(fs.readFileSync(path.join(root,'data','recorded.json'),'utf8'));
const elements=new Map(),events={},messages=[];
const defaults={q1:1,q2:.5,countN:3,countM:5};
const element=id=>{if(!elements.has(id))elements.set(id,{innerHTML:'',textContent:'',value:defaults[id]||'',getBoundingClientRect:()=>({height:1000})});return elements.get(id)};
const parent={postMessage:m=>messages.push(m)};
const context=vm.createContext({console,JSON,Math,Map,Set,Number,String,Array,Error,
 document:{getElementById:element,querySelectorAll:()=>[],addEventListener:(name,fn)=>events[name]=fn,body:{}},
 window:{parent,addEventListener:(name,fn)=>events['window:'+name]=fn},
 ResizeObserver:class{observe(){}},requestAnimationFrame:fn=>fn(),
 setInterval:()=>1,clearInterval:()=>{},setTimeout:()=>{},URL:{},Blob:class{}});
vm.runInContext(source,context);
assert.equal(messages[0].type,'streamlit:componentReady');
events['window:message']({source:parent,data:{type:'streamlit:render',args:{bundle:data,builtin:true,version:'one'}}});
assert.match(element('app').innerHTML,/GLiREL/);
assert.match(element('content').innerHTML,/North Labs/);
const run=s=>vm.runInContext(s,context);
const T=context.window.GLIREL_TEST;

// Strict threshold and per-pair top-k match the Python core.
run('state.case=0;state.k=1;');
assert.equal(T.selection(0).length,2);
run('state.k=-1');
assert.equal(T.selection(0).length,6);
const first=run('candidates()[0].score');
assert(!T.selection(first).some(p=>p.score===first));
run('state.k=1');
assert.equal(run('gold().size'),1);

// Every recorded case, every stage and all five views must render successfully.
for(let i=0;i<5;i++){
 for(const tab of ['io','pipeline','errors','lab','about']){
  for(let step=0;step<(tab==='pipeline'?8:1);step++){
   run(`state.case=${i};state.tab=${JSON.stringify(tab)};state.step=${step};renderBody()`);
   assert(element('content').innerHTML.length>200);
   assert(!element('content').innerHTML.includes('NaN'));
  }
 }
}
run('state.lab=0;state.tab="lab";renderBody()');
assert.match(element('toy-result').innerHTML,/0\.7773/);
run('state.lab=1;renderBody()');
assert.match(element('count-result').innerHTML,/30/);

// Import validation catches malformed spans, duplicate candidates and self-pairs.
let bad=structuredClone(data);
bad.cases[0].example.spans[1]=bad.cases[0].example.spans[0];
assert.throws(()=>T.normalize(bad));
bad=structuredClone(data);
bad.cases[0].candidates_before_display_thresholds.push(bad.cases[0].candidates_before_display_thresholds[0]);
assert.throws(()=>T.normalize(bad));
bad=structuredClone(data);
bad.cases[0].candidates_before_display_thresholds[0].score=Infinity;
assert.throws(()=>T.normalize(bad));
bad=structuredClone(data);
bad.cases[0].candidates_before_display_thresholds[0].tail_pos=bad.cases[0].candidates_before_display_thresholds[0].head_pos;
assert.throws(()=>T.normalize(bad));

// Imported text is escaped before HTML rendering, not executed as markup.
const escaped=structuredClone(data);
escaped.cases[0].example.tokens[0]='<img src=x onerror=alert(1)>';
events['window:message']({source:parent,data:{type:'streamlit:render',args:{bundle:escaped,builtin:false,version:'two'}}});
run('state.case=0;state.tab="io";renderBody()');
assert(!element('content').innerHTML.includes('<img'));
assert(element('content').innerHTML.includes('&lt;img'));
console.log('PASS: handshake; strict threshold; top-k per pair; 5 cases × 8 stages; all views; both labs; invalid imports; HTML escaping.');
