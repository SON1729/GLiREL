'use strict';
// The diagrams follow this source revision. A trace is authoritative about its own flags/shapes.
const GLIREL_SOURCE='https://github.com/jackboyla/GLiREL/blob/1f485a268c2144cdc0221bcf0b24152f93c0a155/';
function moduleLink(path,part){return `<a class="source-link" href="${GLIREL_SOURCE}${path}" target="_blank" rel="noreferrer">↗ ${part}</a>`}
function shapeOf(name){return trace()[name]?.shape||[]}
function shapeText(s){return '['+s.join(' × ')+']'}
function cfg(){return S.bundle.config||{}}
function relationLabels(){const e=S.example,t=trace(),axis=t.label_axis||t.prompt_and_text_units?.[0]?.slice(0,2*e.labels.length).filter((_,i)=>i%2===1);return axis&&axis.length===e.labels.length&&axis.every(l=>e.labels.includes(l))?axis:e.labels}
function knownFlag(name){return typeof cfg()[name]==='boolean'?cfg()[name]?'bật':'tắt':'chưa ghi'}
function flagsView(){return `<div class="config-flags"><span>refine_relation: <b>${knownFlag('refine_relation')}</b></span><span>refine_prompt: <b>${knownFlag('refine_prompt')}</b></span></div>`}
function exactAxes(){
 const t=trace(),e=S.example,M=e.labels.length;
 const units=t.prompt_and_text_units?.[0];
 const labels=t.label_axis||units?.slice(0,2*M).filter((_,i)=>i%2===1);
 // New traces carry the actual model pair indices. Legacy traces only contain a full all-pairs grid.
 const pairs=t.pair_axis||((candidates().length===e.entities.length*(e.entities.length-1)*M)?pairOptions().map(([h,v])=>[e.entities[h].slice(0,2),e.entities[v].slice(0,2)]):null);
 if(!pairs||!labels||t.logits?.shape?.[1]!==pairs.length||t.logits?.shape?.[2]!==labels.length)return null;
 // Only bind the old flattened prefix after every saved logit agrees with the candidate at that axis.
 const z=t.logits.first_values||[];
 for(let i=0;i<z.length;i++){
  const pair=pairs[Math.floor(i/labels.length)];if(!pair)return null;
  const p=candidates().find(p=>p.head_pos[0]===pair[0][0]&&p.head_pos[1]===pair[0][1]+1&&p.tail_pos[0]===pair[1][0]&&p.tail_pos[1]===pair[1][1]+1&&p.label===labels[i%labels.length]);
  if(!p||Math.abs(1/(1+Math.exp(-z[i]))-p.score)>1e-6)return null;
 }
 return {pairs,labels,verifiedPrefix:z.length,explicit:!!t.pair_axis};
}
function storedLogit(p){const axes=exactAxes();if(!axes)return null;
 const k=axes.pairs.findIndex(pair=>pair[0][0]===p.head_pos[0]&&pair[0][1]+1===p.head_pos[1]&&pair[1][0]===p.tail_pos[0]&&pair[1][1]+1===p.tail_pos[1]);
 const m=axes.labels.indexOf(p.label),flat=k*axes.labels.length+m;if(k<0||m<0)return null;
 const full=trace().logit_matrix?.[k]?.[m];
 if(Number.isFinite(full)&&Math.abs(1/(1+Math.exp(-full))-p.score)<1e-6)return {value:full,k,m,origin:'matrix'};
 const value=trace().logits.first_values?.[flat];
 return Number.isFinite(value)?{value,k,m,origin:'prefix'}:null;
}
function traceSummary(){const t=trace(),e=S.example,P=2*e.labels.length+1;
 const checks=[
  ['Prompt U = 2M+1+N',t.prompt_and_text_units?.[0]?.length===P+e.tokens.length],
  ['TokenRep: U đơn vị',t.pooled_embeddings?.shape?.[1]===P+e.tokens.length],
  ['BiLSTM: N token câu',(t.text_after_rnn||t.contextual_text_after_rnn)?.shape?.[1]===e.tokens.length],
  ['Scorer: K×M ô',t.logits?.shape?.[2]===e.labels.length]
 ];
 return `<div class="trace-checks">${checks.map(([s,ok])=>`<span class="${ok?'verified':'unverified'}">${ok?'✓':'?'} ${s}</span>`).join('')}</div>`;
}
function tensorFlow(label,shape,body='',tone='text'){
 return `<div class="tensor-op tone-${tone}"><div class="tensor-op-head"><b>${label}</b><code>${esc(shapeText(shape))}</code></div>${body}</div>`;
}
function splitDiagram(){
 const M=S.example.labels.length,P=2*M+1,d=cfg().hidden_size||'d';
 const textShape=shapeOf('text_after_rnn').length?shapeOf('text_after_rnn'):shapeOf('contextual_text_after_rnn');
 const strategy=cfg().label_embed_strategy||'chưa ghi';
 const rule=strategy==='both'?'pₜ = (E[:,2t,:] + E[:,2t+1,:]) / 2':strategy==='label'?'pₜ = E[:,2t+1,:]':strategy==='ent_token'?'pₜ = E[:,2t,:]':'Cần label_embed_strategy trong config để xác nhận.';
 return `<div class="split-junction">E · ${esc(shapeText(shapeOf('pooled_embeddings')))}<div>tách tại vị trí P = ${P}</div></div><div class="split-flow">${tensorFlow('Văn bản · sau [SEP]',[1,S.example.tokens.length,d],`<div class="equation">E[:, ${P}:${P+S.example.tokens.length}, :]</div>${tokens()}${connector}<b>BiLSTM · 1 lớp, 2 chiều</b><div class="equation">${typeof d==='number'?d/2:'d/2'} mỗi chiều → ${d} chiều</div>${summary('text_after_rnn')}`,'text')}${tensorFlow('Nhãn · trước [SEP]',[1,M,d],`<div class="equation">${esc(rule)}</div>${relationLabels().map((l,i)=>`<span class="chip">t=${i} · ${esc(l)}</span>`).join('')}<p class="hint">Chưa qua FFN nhãn ở điểm tách này; xem bước 6.</p>`,'label')}</div>`;
}
function entityDiagram(){const e=S.example,d=cfg().hidden_size||'d',mode=cfg().span_marker_mode;
 if(e.entities.length<2)return pending('Cần ít nhất hai thực thể');
 const [h,t]=S.pair,span=e.entities[h],known=mode==='markerv1'||mode==='markerv2';
 let entityTransform;
 if(mode==='markerv2')entityTransform=`<div class="equation">e = FFN(mean(R[${span[0]}:${span[1]+1}]))</div><p class="hint">markerv2 lấy trung bình trong span rồi chiếu.</p>`;
 else entityTransform=`<div class="boundary-grid">${tensorFlow('Biên đầu', [d],`<code>R[${span[0]}] → FFN_start</code>`,'text')}${tensorFlow('Biên cuối',[d],`<code>R[${span[1]}] → FFN_end</code>`,'text')}</div><div class="equation">[FFN_start(R[${span[0]}]); FFN_end(R[${span[1]}])]<br>→ ReLU → FFN_span → e<sub>${h+1}</sub> ∈ R<sup>${d}</sup></div>`;
 return `${known?`<span class="evidence-badge">Config xác nhận: ${esc(mode)}</span>`:`<div class="evidence-note">Trace chưa lưu <code>span_marker_mode</code>. Hình dưới minh họa <b>markerv1 theo config FewRel của tác giả</b>; cần config của checkpoint để xác nhận.</div>`}<div class="span-sentence">${tokens()}</div>${box(`Tạo biểu diễn ${esc(entity(h))}`,entityTransform)}${connector}<div class="flow-pair">${box('Head · thứ tự thứ nhất',pill(h))}<div class="connector">⊕</div>${box('Tail · thứ tự thứ hai',pill(t))}</div><div class="equation">[e<sub>${h+1}</sub> ; e<sub>${t+1}</sub>] ∈ R<sup>${typeof d==='number'?2*d:'2d'}</sup> → FFN_pair → κ<sub>${h+1},${t+1}</sub> ∈ R<sup>${d}</sup></div>${pairButtons()}<p class="hint">“;” là nối vector. ${mode==='markerv2'?'FFN_span và FFN_pair có tham số riêng.':'FFN_start, FFN_end, FFN_span và FFN_pair có tham số riêng.'} Dropout tắt khi eval.</p>${summary('pair_projection')}`;
}
function labelDiagram(){const strategy=cfg().label_embed_strategy,d=cfg().hidden_size||'d';
 const selected=strategy==='both'?'(marker + tên nhãn) / 2':strategy==='label'?'vector tên nhãn':strategy==='ent_token'?'vector marker [REL]':'chưa xác nhận (thiếu cấu hình)';
 return `<span class="evidence-badge">label_embed_strategy = ${esc(strategy||'chưa ghi')}</span>${relationLabels().map((l,i)=>`<div class="label-lane"><div><span class="unit marker">[REL]<small>vị trí ${2*i}</small></span><span class="unit label-unit">${esc(l)}<small>vị trí ${2*i+1}</small></span></div><div class="lane-arrow">→</div><div class="lane-result"><b>${esc(selected)}</b><code>p<sub>${i}</sub> ∈ R<sup>${d}</sup></code></div></div>`).join('')}<div class="operator-chain"><span>Linear ${d}→${typeof d==='number'?4*d:'4d'}</span><b>→</b><span>Dropout · tắt ở eval</span><b>→</b><span>ReLU</span><b>→</b><span>Linear →${d}</span></div>${summary('label_projection')}<p class="hint">Mỗi tên nhãn là một đơn vị, kể cả tên có nhiều từ. Với pooling=first, không lấy trung bình mọi subtoken trong tên.</p>`;
}
function refinementDiagram(){const r=cfg().refine_relation,p=cfg().refine_prompt;
 if(r===false&&p===false)return `<div class="bypass">κ, q <span>→</span> <b>Refinement: bỏ qua (cả hai cờ = false)</b> <span>→</span> scorer</div>`;
 return `<div class="refinement"><b>Refinement trước scorer</b>${flagsView()}${r?'<p>κ ← Refine_relation(κ, token sau BiLSTM)</p>':''}${p?'<p>q ← Refine_prompt(q, κ sau bước trên)</p>':''}${typeof r!=='boolean'||typeof p!=='boolean'?'<p>Thiếu cờ trong config: không tự khẳng định nhánh nào đã chạy.</p>':''}<small>Mã commit này khởi tạo ReadProcess với read_only=True: cross-attention + residual/LayerNorm/FFN; không chạy nhánh self-attention bên trong.</small></div>`;
}
function scorerDescription(){const type=cfg().scorer;
 if(type==='dot')return {name:'Tích vô hướng → sigmoid',equation:'zᵤᵥₜ = Σⱼ κᵤᵥⱼ qₜⱼ ; s = 1 / (1 + exp(−z))'};
 if(type==='dot_norm')return {name:'Dot + các bias học được → sigmoid',equation:'z = κ·q + bias + bias_type(q) + bias_pair(κ) ; s = sigmoid(z)'};
 if(type==='dot_thresh')return {name:'Dot có scale và bias học được',equation:'z = (κ·proj_type(q) + β(q)) × softplus(α(q)); s = sigmoid(z)'};
 if(type==='concat_proj')return {name:'Ghép đặc trưng → MLP → sigmoid',equation:'z = MLP([κ⊙q ; κ−q ; κ ; q]); s = sigmoid(z)'};
 return {name:'Scorer chưa được xác nhận',equation:'Cần trường scorer trong config; không mặc định là dot.'};
}
function sourceForStep(i){return [
 ['glirel/modules/base.py','preprocess_spans / collate_fn'],['glirel/model.py','compute_score · ghép prompt'],
 ['glirel/modules/token_rep.py','TokenRepLayer.forward'],['glirel/model.py','compute_score · tách hai nhánh'],
 ['glirel/modules/rel_rep.py','SpanMarkerV1/V2 · RelMarkerv0'],['glirel/model.py','prompt_rep_layer / both'],
 ['glirel/modules/layers.py','RefineLayer / ScorerLayer'],['glirel/model.py','predict / batch_predict_relations']
 ][i]}
function stageData(){
 const e=S.example,t=trace(),N=e.entities.length,M=e.labels.length,P=2*M+1,d=cfg().hidden_size||'d',prompt=promptUnits();
 const hs=t.encoder?.hidden_state?.shape||['B','L','D'],D=hs[2],pooling=cfg().subtoken_pooling||'chưa ghi';
 const zShape=t.logits?.shape||[1,'K',M],scorer=scorerDescription();
 const specs=[
 {title:'Đầu vào của một lượt suy luận',sub:'Câu đã tách token + span thực thể + tên nhãn ứng viên.',input:'N token, E thực thể có span [a,b], M tên nhãn.',op:'Dùng span đã cung cấp. Giao diện chọn span thay cho NER bên ngoài.',out:`N=${e.tokens.length} · E=${N} · M=${M}.`,why:'GLiREL giải quyết phân loại quan hệ giữa các thực thể đã biết. Việc bạn bấm chọn tên là chuẩn bị đầu vào, chưa phải GLiREL suy ra thực thể.',math:'Chỉ số token bắt đầu 0. Input [a,b] bao gồm b; một thực thể có thể gồm nhiều token. E1/E2 là ID giao diện.',visual:box('Token gốc · vị trí trong câu',tokens())+connector+box('Span thực thể',e.entities.map((_,i)=>pill(i)).join(' '))+connector+box('Tên nhãn ứng viên',e.labels.map(l=>`<span class="chip">${esc(l)}</span>`).join(' '))},
 {title:'Ghép prompt và chạy tokenizer',sub:'Giữ nguyên từng tên nhãn như một đơn vị trước subtokenization.',input:'Danh sách nhãn có thứ tự và token câu.',op:'[REL], nhãn₀, …, [REL], nhãnₘ₋₁, [SEP], câu; tokenizer is_split_into_words=True.',out:`P=2M+1=${P}; U=P+N=${P+e.tokens.length} đơn vị. L subtoken có thể khác U.`,why:'Prompt và câu đi qua cùng encoder để tương tác với nhau. ID subtoken do tokenizer tạo, không phải ID thực thể hay chỉ số token câu.',math:'Prompt chiếm vị trí [0,P). Câu nằm ở [P,P+N). [CLS]/[SEP] do tokenizer thêm có thể hiện thêm trong bản ghi subtoken.',visual:box('Đơn vị trước tokenizer',`<div class="token-stream">${prompt.map((x,i)=>`<span class="unit ${i<2*M?(i%2===0?'marker':'label-unit'):''}" style="--i:${i}"><small>u=${i}</small>${esc(x)}</span>`).join('')}</div>`)+connector+box('Subtoken + ID thực tế',(t.encoder?.subtokens?.[0]?`<div class="token-stream">${t.encoder.subtokens[0].map((x,i)=>`<span class="unit"><small>vị trí ${i} · ID ${t.encoder.input_ids?.[0]?.[i]??'?'}</small>${esc(x)}</span>`).join('')}</div>`:pending('Bản ghi chưa chứa tokenizer output')))},
 {title:'Encoder → pooling → phép chiếu',sub:'Tensor “pooled_embeddings” được ghi sau cả pooling và Linear.',input:'Subtoken ID cho toàn bộ prompt + câu.',op:`DeBERTa → pooling=${pooling} theo word_ids → Linear nếu D khác d.`,out:`H ${shapeText(hs)} → E ${shapeText(shapeOf('pooled_embeddings'))}.`,why:'Encoder cho vector theo ngữ cảnh ở mỗi subtoken. Pooling first giữ subtoken đầu của từng đơn vị; Linear đổi số chiều, không phải chỉ lấy bớt tọa độ.',math:`B: số câu; L: số subtoken; U=${P+e.tokens.length} đơn vị; D=${D}; d=${d}. Linear D→d chỉ có khi D≠d. Các con số dưới là tensor sau phép chiếu.`,visual:tensorFlow('DeBERTa · chung cho nhãn và câu',hs,schematic(),'encoder')+connector+tensorFlow(`Pooling ${esc(pooling)} · giữ theo word_ids`,[1,P+e.tokens.length,D],`<p class="hint">Mỗi đơn vị trước tokenizer nhận một vector ${esc(D)} chiều. ID special không thuộc đơn vị được bỏ khi pooling.</p>`,'text')+connector+tensorFlow(`${D===d?'Không cần Linear':'Linear '+D+' → '+d} · TokenRepLayer output`,shapeOf('pooled_embeddings'),summary('pooled_embeddings'),'encoder')},
 {title:'Tách prompt và câu tại [SEP]',sub:'BiLSTM chỉ xử lý phần câu; nhãn lấy vector từ phần prompt.',input:`E có ${P+e.tokens.length} đơn vị, mỗi đơn vị d=${d}.`,op:`Lấy câu E[:,${P}:${P+e.tokens.length},:]; lấy marker/tên nhãn trước [SEP] theo label_embed_strategy.`,out:`Văn bản sau BiLSTM ${shapeText(shapeOf('text_after_rnn').length?shapeOf('text_after_rnn'):shapeOf('contextual_text_after_rnn'))}; M=${M} biểu diễn nhãn trước FFN.`,why:'Tách tensor là phép chọn vị trí, không phải chạy hai encoder. BiLSTM một lớp hai chiều thêm xử lý tuần tự cho câu; nhãn không đi qua BiLSTM.',math:`P=2M+1=${P}. Với both: pₜ=(E[:,2t,:]+E[:,2t+1,:])/2. R=BiLSTM(E_câu). BiLSTM có d/2 trạng thái mỗi chiều rồi nối thành d=${d}.`,visual:splitDiagram()},
 {title:'Tạo vector thực thể rồi vector cặp',sub:'Đọc span trước; sau đó ghép head–tail theo thứ tự.',input:'R: vector token sau BiLSTM; span_idx và relations_idx.',op:'SpanMarker theo config → lấy vector head, tail → nối → FFN_pair.',out:`κ ${shapeText(shapeOf('pair_projection'))}; tối đa E(E−1)=${N*(N-1)} cặp.`,why:'Một cặp có hướng khác cặp đảo chiều. Mã lấy các cặp từ relations_idx, không coi hàng xếp theo điểm của bảng output là thứ tự tensor.',math:cfg().span_marker_mode==='markerv2'?'markerv2: e = FFN(mean các vector trong span). Sau đó κᵤᵥ=FFN_pair([eᵤ;eᵥ]).':'markerv1: e = FFN_span(ReLU([FFN_start(R[a]);FFN_end(R[b])])). κᵤᵥ=FFN_pair([eᵤ;eᵥ]). '+(cfg().span_marker_mode==='markerv1'?' Config xác nhận markerv1.':' Trace cũ thiếu span_marker_mode nên chưa xác nhận được biến thể.'),visual:entityDiagram()},
 {title:'Biến biểu diễn nhãn qua FFN',sub:'Từ vector prompt đã chọn sang q dùng để chấm điểm.',input:`M vector p theo strategy=${cfg().label_embed_strategy||'chưa ghi'}.`,op:'FFN nhãn: Linear → Dropout → ReLU → Linear. Dropout không tác động ở eval.',out:`q ${shapeText(shapeOf('label_projection'))}.`,why:'Đưa nhãn về không gian so khớp với cặp. FFN nhãn dùng chung trọng số cho các nhãn, nhưng khác trọng số các FFN của nhánh thực thể/cặp.',math:'Với both, trung bình marker và tên nhãn trước FFN. Với label hoặc ent_token, chỉ dùng tên nhãn hoặc marker tương ứng. qₜ=FFN_label(pₜ).',visual:labelDiagram()},
 {title:'Refinement theo config → scorer → sigmoid',sub:`scorer=${cfg().scorer||'chưa ghi'} · ${scorer.name}.`,input:'κ và q; nếu refinement bật thì dùng phiên bản đã refine.',op:'Theo thứ tự code: refine_relation với văn bản → refine_prompt với cặp → scorer → sigmoid.',out:`Logits Z ${shapeText(zShape)}; điểm S có cùng shape.`,why:'Mỗi cặp–nhãn được chấm độc lập. Refine tắt trong các bản ghi mẫu hiện tại; không tự thêm attention vào hình. Sigmoid không ép tổng các nhãn bằng 1.',math:scorer.equation+' K là số cặp được model tạo; M là số nhãn. Khi có z thực đã lưu, bấm ô để đối chiếu sigmoid(z).',visual:refinementDiagram()+flagsView()+`<div class="equation">${esc(scorer.equation)}</div>${heatmap()}`},
 {title:'Ngưỡng → top-k theo cặp → bộ ba',sub:'Đây là quyết định hiển thị; không chạy lại encoder.',input:'Điểm sigmoid ứng viên; ngưỡng τ; top-k.',op:'Giữ s>τ → xếp điểm giảm dần → lấy tối đa k nhãn riêng trên từng cặp.',out:`${selection().length} bộ ba; output span [a,b+1) loại trừ end.`,why:'Hạ ngưỡng có thể giữ thêm cả nhãn đúng và nhãn dư. Một nhãn đứng dưới top-k vẫn bị loại dù vượt ngưỡng. Kiểu thực thể ở demo chỉ là metadata, chưa áp ràng buộc loại.',math:'Top-k không áp cho toàn câu. s=τ bị loại. Đổi gold chỉ đổi đối chiếu; không tác động scorer. candidates ghi tại threshold=0 không chứa score bằng 0 nếu có underflow.',visual:S.dirty?pending():`<div class="gate"><div><small>Ứng viên theo điểm</small>${candidates().slice().sort((a,b)=>b.score-a.score).slice(0,7).map((p,i)=>`<div class="packet ${selection().some(x=>key(x)===key(p))?'':'rejected'}" style="--i:${i}">${esc(text(p.head_text))} → ${esc(p.label)} → ${esc(text(p.tail_text))}<br><b>${p.score.toFixed(4)}</b></div>`).join('')}</div><div class="gate-wall">s &gt;<br>${S.threshold.toFixed(2)}<br><br>top-k<br>/ cặp</div><div><small>Được giữ</small>${selection().slice(0,7).map((p,i)=>`<div class="packet kept" style="--i:${i}">✓ ${esc(text(p.head_text))} → ${esc(p.label)} → ${esc(text(p.tail_text))}</div>`).join('')||'<p class="hint">Không có bộ ba được giữ.</p>'}</div></div><p class="hint">Hình minh họa tối đa 7 ứng viên / kết quả; danh sách bên dưới hiển thị toàn bộ.</p>`}
 ];
 const item=specs[S.step];item.source=sourceForStep(S.step);return item;
}
