// Technical mid-project deck: editable evidence from unchanged released results.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const C={bg:'#F8F6F1',ink:'#123039',teal:'#247A70',blue:'#627FAD',coral:'#C97552',red:'#B8675C',gold:'#B78B41',muted:'#576969',line:'#D7DFDA',light:'#E7EEEA',white:'#FFFFFF'};
export const CHART_SLIDES=[6,9,10,11,12,13];
export const TABLE_SLIDES=[7,8,19];
export async function build(p){
 const read=async f=>JSON.parse(await fs.readFile(path.join(ROOT,f),'utf8'));
 const content=await read('docs/presentation/verity-midproject-content.json');
 const ops=await read('reports/reasonke/operations-final/results.json');
 const depth=await read('reports/reasonke/depth-final/results.json');
 const dev=await read('reports/reasonke/depth-development/retrieval-selection.json');
 const audit=await read('output/presentation/analysis/failure-audit.json');
 if(p.slides.items.length)throw Error('Use a new presentation; preserve previous materials.');
 p.theme.defaultFont='Calibri';
 const tx=(s,t,x,y,w,h,size=28,color=C.ink,bold=false,font='Calibri',align='left')=>{
  const a=s.shapes.add({geometry:'textbox',name:t.slice(0,90),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=t;a.text.style={fontSize:size,typeface:font,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:0};return a;
 };
 const slide=(n,sub='',dark=false)=>{
  const s=p.slides.add({width:1280,height:720,background:{fill:dark?C.ink:C.bg}});
  tx(s,content.slides[n-1].title,64,33,1152,86,n===16?39:42,dark?C.white:C.ink,true);
  if(sub)tx(s,sub,66,122,1145,52,25,dark?'#C9DDD7':C.muted);
  tx(s,'Verity',64,678,150,22,18,dark?'#C9DDD7':C.muted);
  tx(s,n>16?'Q&A · '+String(n):String(n).padStart(2,'0'),1100,678,112,22,18,dark?'#C9DDD7':C.muted,false,'Calibri','right');return s;
 };
 const foot=(s,t,dark=false)=>tx(s,t,64,625,1148,44,21,dark?'#C9DDD7':C.muted);
 const box=(s,t,x,y,w,h,accent=false,size=27)=>{
  const a=s.shapes.add({geometry:'rect',name:t.slice(0,80),position:{left:x,top:y,width:w,height:h},fill:accent?C.teal:C.light,line:{fill:accent?C.teal:C.line,width:1}});
  a.text=t;a.text.style={fontSize:size,typeface:'Calibri',color:accent?C.white:C.ink,alignment:'center',verticalAlignment:'middle',wrap:'square',insets:12};return a;
 };
 const link=(s,a,b,from='right',to='left')=>s.shapes.connect(a,b,{kind:from==='right'&&to==='left'&&Math.abs(a.position.top+a.position.height/2-b.position.top-b.position.height/2)<1?'straight':'elbow',fromSide:from,toSide:to,line:{fill:C.teal,width:2.2},tail:{type:'arrow',width:'med',length:'med'}});
 const table=(s,values,widths,{x=64,y=202,w=1152,h=354,size=26}={})=>{
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  values.forEach((row,r)=>row.forEach((_,c)=>{const cell=t.getCell(r,c);cell.fill=r?C.bg:C.light;cell.text.style={fontSize:size,typeface:'Calibri',color:C.ink,bold:r===0,verticalAlignment:'middle',insets:10};}));return t;
 };
 const image=async(s,file,alt,x,y,w,h)=>{
  const bytes=await fs.readFile(path.join(ROOT,file));
  s.images.add({dataUrl:'data:image/jpeg;base64,'+bytes.toString('base64'),contentType:'image/jpeg',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
 };
 const plot=(s,categories,series,{x=62,y=197,w=785,h=411,max,step,title='Correct answers',percent=false,stacked=false,horizontal=false,legend=true,format='0;-0;;'}={})=>s.charts.add('bar',{
  position:{left:x,top:y,width:w,height:h},categories,
  series:series.map((v,i)=>({name:v.name+'\u2002',values:v.values.map(n=>Number(n.toPrecision(12))),fill:v.fill??[C.teal,C.blue,C.gold][i],dataLabels:{deleted:v.labels===false,showValue:v.labels!==false,position:stacked?'center':'outEnd',numberFormatCode:percent?'0.0%':format,textStyle:{fontSize:stacked?24:23,bold:true,fill:stacked?C.white:C.ink}}})),
  barOptions:{direction:horizontal?'bar':'column',grouping:stacked?'stacked':'clustered',gapWidth:90},hasLegend:legend,
  legend:{position:'bottom',textStyle:{fontSize:22,fill:C.ink}},
  xAxis:{textStyle:{fontSize:23,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},
  yAxis:{min:0,max,majorUnit:step,numberFormatCode:percent?'0%':'0',textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.line,width:1},title:{text:title,textStyle:{fontSize:21,fill:C.muted}}},
  chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0}
 });
 const ot=c=>ops.conditions[c].overall,dt=c=>depth.conditions[c].overall;
 let s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
 tx(s,'Verity',70,110,1140,140,106,C.white,true);
 tx(s,'Governed knowledge updates\nfor connected factual questions',76,292,1120,142,45,C.white);
 tx(s,'System mechanisms · Controlled experiments · Failure analysis',76,523,1135,64,29,'#C9DDD7');
 tx(s,'Technical mid-project review  |  31 October 2026',76,601,1120,39,24,'#C9DDD7');

 s=slide(2,'A familiar historical update. A bounded technical objective.');
 tx(s,'20 DEC 2021 → 27 JAN 2022',65,208,665,46,28,C.muted,true);
 tx(s,'Government of India → Talace',65,271,710,65,36,C.teal,true);
 tx(s,'Owner → parent company',65,353,710,55,33,C.ink);
 tx(s,'Correct relevant answers. Preserve other scopes.\nApprove, trace and reverse the local update.',65,470,710,112,29,C.ink);
 await image(s,'output/presentation/midproject/assets/air-india-handover-pib.jpg','Genuine PIB photograph of the 27 January 2022 Air India handover',817,211,395,294);
 tx(s,'Handover snapshot; not current ownership.',817,525,395,71,23,C.muted);
 foot(s,'Government and Tata primary sources motivate the task. Experimental results use separate controlled fictional worlds.');

 s=slide(3,'State consistency, retrieval and neural generation have separate responsibilities.');
 const top=['Question','API\nCapture revision r','Retrieve\nApproved memory','GPU worker\nGenerate','Answer + evidence'].map((t,i)=>box(s,t,64+i*236,220,200,130,i===3,27));
 for(let i=0;i<4;i++)link(s,top[i],top[i+1]);
 const state=box(s,'SQLite\nVersions, events, messages',300,460,436,110,false,28);
 const operator=box(s,'Operator approval',64,460,200,110,false,27);link(s,operator,state);link(s,state,top[2],'top','bottom');
 tx(s,'Compatible history only\nPersist raw output, fact IDs,\ncheckpoint and memory revision',785,443,425,141,27,C.ink);
 foot(s,'Frozen ReasonKE_plus / Qwen2.5-7B. NF4 + FP16, greedy, 4,096 input / 512 new-token ceilings, T4.');

 s=slide(4,'A represented fact has one active version; a rollback is one atomic state change.');
 tx(s,'fact_key = (entity_id, relation_id, scope_id)',66,205,1145,68,34,C.teal,true,'Consolas');
 const life=['Approve\nExplicit authority','Supersede\nName predecessor','Rollback\nRestore predecessor'].map((t,i)=>box(s,t,64+i*396,316,360,115,i===1,29));
 link(s,life[0],life[1]);link(s,life[1],life[2]);
 tx(s,'BEGIN IMMEDIATE',65,480,345,55,29,C.ink,true,'Consolas');
 tx(s,'State + one revision + audit event pair\nCommit together or restore the prior state.',450,478,762,105,30,C.ink);
 foot(s,'Partial unique index enforces active identity. Injected late-write failures test the transaction boundary.');

 s=slide(5,'A BM25-style lexical ranker follows answer text to bridge facts.');
 tx(s,'S ← top 2 records ranked by q\nfrontier ← answers(S)\nrepeat within round limit:\n  next ← best unseen(frontier)\n  S ← S ∪ {next}\n  frontier ← answer(next)',66,210,685,315,25,C.ink,false,'Consolas');
 const chain=['Service','Owner team','On-call contact'].map((t,i)=>box(s,t,838,213+i*124,374,86,i===2,29));
 link(s,chain[0],chain[1],'bottom','top');link(s,chain[1],chain[2],'bottom','top');
 tx(s,'k₁ = 1.2   b = 0.75',66,551,675,47,28,C.teal,true);
 foot(s,'2 seeds · at most 3 expansion rounds · 4-fact context. Normal retrieval receives no gold support or target answer.');

 s=slide(6,'Development only: 32 questions per depth group, before neural profiling and final freeze.');
 const candidate=dev.candidate_family.find(v=>v.cap===4&&v.seed_count===2);
 plot(s,['Locality','1 hop','2 hops','3 hops'],[{name:'Two-stage',values:[0,1,2,3].map(k=>dev.baseline_two_stage_k4.by_depth[String(k)].complete),fill:C.blue},{name:'Iterative',values:[0,1,2,3].map(k=>candidate.by_depth[String(k)].complete),fill:C.teal}],{max:36,step:8,title:'Complete support paths'});
 tx(s,'96/128 → 128/128',887,225,325,100,35,C.teal,true);
 tx(s,'Three-hop paths\n0/32 → 32/32',887,374,325,107,30,C.ink);
 tx(s,'Six declared candidates.\nSelected: 2 seeds, 4 facts.',887,527,325,82,25,C.muted);
 foot(s,'Retrieval support is measured independently of final neural answers. Final depth support: 448/448 fact occurrences.');

 s=slide(7,'Split the edit lineage before generating related question variants.');
 table(s,[['Study','Facts / world','Max depth','Dev / final questions'],['Operations','9','2 hops','128 / 256'],['Depth challenge','12','3 hops','128 / 256']],[315,225,225,387],{h:221,size:28});
 const parent=box(s,'48 fact worlds',64,487,365,106,false,31);
 const devNode=box(s,'16 development lineages',670,452,542,70,false,29);
 const testNode=box(s,'32 final lineages',670,541,542,70,true,29);
 link(s,parent,devNode);link(s,parent,testNode);
 foot(s,'8 variants per lineage. Automatic label/graph audit. Small worlds and controlled wording bound generalisation.');

 s=slide(8,'Updated arms share approved information; response-format prompts remain model appropriate.');
 table(s,[['Arm','Generator','Supplied context'],['R0 diagnostic','ReasonKE_plus','Initial full handbook'],['R-RAG / R-ALL','ReasonKE_plus','Current retrieved / current full'],['G-RAG / G-ALL','Qwen2.5-7B','Same retrieval / current full']],[267,304,581],{h:262,size:27});
 tx(s,'accuracy = correct / all assigned',65,495,1147,56,34,C.teal,true);
 tx(s,'Whole endorsed final span. UNKNOWN receives zero accuracy credit.',65,562,1147,48,27,C.ink);
 foot(s,'Original: 1,312 generations. Depth: 544. Each main arm n=256; oracle diagnostics n=16 per model.');

 s=slide(9,'Five matched main configurations, 256 assigned questions each.');
 const codes=['R0','R-RAG','R-ALL','G-RAG','G-ALL'];
 plot(s,['Initial R','Retrieved R','Full R','Retrieved Q','Full Q'],[{name:'Correct',values:codes.map(c=>ot(c).counts.correct),fill:C.teal},{name:'Strict mismatch',values:codes.map(c=>ot(c).counts.wrong),fill:C.red,labels:false},{name:'UNKNOWN',values:codes.map(c=>ot(c).counts.abstain),fill:C.gold,labels:false}],{w:1152,max:280,step:64,stacked:true,title:'Assigned outcomes',y:191,h:408});
 foot(s,'R-ALL: 25 wrong. Retrieved Q: 15 UNKNOWN. Full Q: 13 strict mismatches, 2 UNKNOWN.\nR = ReasonKE_plus; Q = Qwen. Retrieved difference +5.86 pp, 95% lineage interval [3.91, 8.20].');

 s=slide(10,'Separate final challenge: 64 assigned questions per group, complete retrieved support.');
 plot(s,['Locality','1 hop','2 hops','3 hops'],[{name:'ReasonKE_plus',values:[0,1,2,3].map(k=>depth.conditions['R-RAG'].by_depth[String(k)].counts.correct)},{name:'Qwen',values:[0,1,2,3].map(k=>depth.conditions['G-RAG'].by_depth[String(k)].counts.correct)}],{max:72,step:16});
 tx(s,'256/256 vs 248/256',887,220,325,105,34,C.teal,true);
 tx(s,'8 Qwen abstentions\n7 in one two-hop\nwording family',887,372,325,132,28,C.ink);
 tx(s,'Three-hop difference: 1/64.\nIts interval includes zero.',887,539,325,70,25,C.muted);
 foot(s,'Perfect observed scores apply to controlled small worlds. All eight Qwen nonsuccesses are UNKNOWN.');

 s=slide(11,'Original full-memory ReasonKE_plus: all 25 wrong final values match another supplied fact.');
 const roles=audit.summary['operations:R-ALL'].exact_off_target_fact_roles;
 plot(s,['Pro owner','Wrong release','Office city','Prior contact','Hosting region','Owner contact'],[{name:'Wrong final values',values:['pro_service_owner','release_2_owner','unrelated_office_city','previous_owner_contact','hosting_region','current_owner_contact'].map(k=>roles[k]),fill:C.coral}],{w:751,max:18,step:3,title:'Failure count',legend:false,horizontal:true});
 tx(s,'Garnet Relay · release 2',860,214,352,70,29,C.ink,true);
 tx(s,'Expected service owner\nSilver unit 77',860,313,352,106,29,C.teal,true);
 tx(s,'Endorsed Pro owner\nSaffron unit 137',860,455,352,106,29,C.coral,true);
 foot(s,'All 63 updated-method nonsuccesses had complete support. These associations describe outputs, not internal causes.');

 s=slide(12,'Original ReasonKE_plus locality: compare the correct denominator for each question.');
 plot(s,['Baseline-correct\nretention (n=124)','All locality\ncorrectness (n=128)'],[{name:'Retrieved',values:[ops.conditions['R-RAG'].baseline_correct_retention.rate,ops.conditions['R-RAG'].locality.accuracy]},{name:'Full memory',values:[ops.conditions['R-ALL'].baseline_correct_retention.rate,ops.conditions['R-ALL'].locality.accuracy]}],{max:1.10,step:.25,title:'Fraction correct',percent:true});
 tx(s,'Initial locality: 124/128',887,222,325,96,31,C.ink,true);
 tx(s,'4 prior mistakes\nare excluded from\nretention’s denominator.',887,367,325,139,28,C.muted);
 tx(s,'Retrieved: 124/124 retained\n128/128 factual',887,539,325,70,25,C.teal,true);
 foot(s,'Preserving an already wrong answer is not factual correctness. Depth has no separate initial arm for a new retention estimate.');

 s=slide(13,'Depth final warm pipeline: 256 generated requests per condition, including abstentions.');
 plot(s,['Median (p50)','Tail (p95)'],[{name:'ReasonKE_plus',values:[dt('R-RAG').generated_latency.p50_seconds,dt('R-RAG').generated_latency.p95_seconds]},{name:'Qwen',values:[dt('G-RAG').generated_latency.p50_seconds,dt('G-RAG').generated_latency.p95_seconds]}],{max:22,step:5,title:'Seconds',format:'0.00'});
 tx(s,'8 more correct answers',887,220,325,111,34,C.teal,true);
 tx(s,'256 vs 248 assigned\nanswers correct',887,374,325,96,29,C.ink);
 tx(s,'Median output tokens\n212 vs 6',887,528,325,80,27,C.muted);
 foot(s,'Warm timing: selection + local IPC + generation; excludes cold load and browser network.');

 s=slide(14,'Genuine recorded Verity interface: inspect individual failures behind an aggregate score.');
 await image(s,'output/presentation/screenshots/technical-v2/verity-results-explorer.jpg','Genuine Verity results explorer filtering 25 original full-memory wrong answers',64,188,890,431);
 tx(s,'Filter outcomes\n\nInspect raw answers\n\nTrace supplied facts',981,230,231,310,27,C.ink,true);
 foot(s,'Recorded evidence, not fresh chat generation. Chat, correction editing and rollback views remain available in Q&A.');

 s=slide(15,'A complete maintenance platform with a measured, bounded conclusion.');
 const conclusions=[['Governed memory','Canonical keys, approval, conflicts, transactional rollback'],['Connected context','Iterative selection, revision-aware history and private worker'],['Reproducible evidence','Frozen protocols, recoverable ledgers and failure audit']];
 conclusions.forEach((r,i)=>{tx(s,r[0],66,211+i*93,334,72,30,C.teal,true);tx(s,r[1],427,211+i*93,785,72,28,C.ink);});
 tx(s,'Selected context supports accurate bounded answers.\nFull memory exposes association errors; coverage costs latency.',66,522,1146,92,31,C.ink,true);
 foot(s,'Current contribution: the implemented system, retrieval adaptation, datasets, evaluation and engineering delivery.');

 s=slide(16,'Planned research: does adaptation improve transfer under matched conditions?',true);
 const future=['Instruction baseline','SFT','SFT + PPO'].map((t,i)=>box(s,t,64+i*396,231,360,105,i===2,33));link(s,future[0],future[1]);link(s,future[1],future[2]);
 tx(s,'Reviewed edit data → profiled pilot → finite training runs',66,389,1146,68,31,C.white,true);
 tx(s,'Fresh entities and wording · larger memory · deeper questions',66,485,1146,68,29,'#D9E8E2');
 tx(s,'Matched retrieval and budgets; accuracy, locality, coverage, latency, resources.',66,561,1146,57,27,'#D9E8E2');
 foot(s,'A verified common parent and fresh frozen test. Existing results remain regression evidence.',true);

 s=slide(17,'Optional Q&A · deterministic state guarantees are separate from generated-answer quality.');
 tx(s,'UNIQUE active (entity_id, relation_id, scope_id)',66,209,1145,61,29,C.teal,true,'Consolas');
 tx(s,'BEGIN IMMEDIATE\n  validate replacement + owned predecessor\n  withdraw replacement; restore predecessor\n  revision ← revision + 1\n  append withdrawn + restored events\nCOMMIT  // exception: ROLLBACK',66,301,1145,259,26,C.ink,false,'Consolas');
 foot(s,'A late event-write fault must leave the full prior state intact. History replay and snapshot validation check state agreement.');

 s=slide(18,'Optional Q&A · 32 final lineage clusters, 10,000 paired bootstrap draws, seed 42.');
 tx(s,'accuracy = correct / assigned\ncoverage = answered / assigned',66,210,1145,112,32,C.teal,true);
 tx(s,'Draw whole lineage IDs with replacement.\nKeep each lineage’s variants and paired system outcomes together.\nRecompute the difference; use 2.5th and 97.5th percentiles.',66,367,1145,135,29,C.ink);
 tx(s,'Retrieved differences: original +5.86 pp [3.91, 8.20]; depth +3.13 pp [1.56, 5.08].',66,542,1145,67,26,C.ink,true);
 foot(s,'UNKNOWN stays in the denominator. Strict full-span mismatches include two Qwen correct chains with extra final text.');

 s=slide(19,'Optional Q&A · author-published ReasonKE_plus and official Qwen2.5-7B-Instruct.');
 table(s,[['Worker setting','Pinned deployment'],['Parameters, each','7,615,616,512'],['Weight / compute','NF4 double quantisation / FP16'],['Attention / decoding','SDPA / greedy'],['Input / output ceiling','4,096 / 512 tokens'],['Device','T4; one model at a time']],[300,445],{w:745,h:366,size:26});
 await image(s,'output/presentation/midproject/screenshots/air-india-chat.jpg','Genuine recorded Verity chat with source context and revision evidence',853,185,359,421);
 foot(s,'Measured T4 load peak: 6.59 GiB allocated, 6.65 GiB reserved. Recorded chat; weights remain fixed.');

 s=slide(20,'Optional Q&A · genuine sourced correction editor and actual rollback/reapproval history.');
 await image(s,'output/presentation/midproject/screenshots/air-india-editor.jpg','Genuine sourced approval draft',64,195,770,410);
 await image(s,'output/presentation/midproject/screenshots/air-india-history.jpg','Genuine rollback history showing revisions 6, 7 and 8',860,190,352,420);
 foot(s,'Recorded acceptance: 7 selected neural requests matched targets; conflict rejected; predecessor restored. Separate from held-out accuracy.');

 let elapsed=0;
 content.slides.forEach((r,i)=>{
  const clock=n=>`${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;
  const when=r.core?`Member ${r.speaker}; ${clock(elapsed)}–${clock(elapsed+r.seconds)} (${r.seconds}s).`:'Hidden, untimed technical Q&A backup; outside the 10-minute talk.';
  p.slides.items[i].speakerNotes.text=`${when}\n\n${r.speech}\n\nMechanism and interpretation\n${r.detail}\n\nWhy this matters\n${r.why}\n\nLikely questions\n${r.qa.map(v=>v[0]+'\n'+v[1]).join('\n\n')}\n\nEvidence\n${r.sources.map(v=>v.startsWith('https:')?v:'https://github.com/ShivamRaghav7/verity/blob/codex/reasonke-integration/'+v).join('\n')}`;
  elapsed+=r.seconds;
 });
 if(p.slides.items.length!==20||elapsed!==600)throw Error('Content/timing mismatch');
 return {slides:20,coreSlides:16,backupSlides:[17,18,19,20],seconds:600,chartSlides:CHART_SLIDES,tableSlides:TABLE_SLIDES};
}
