// Native editable presentation. Reads released evidence, makes no model calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const C={bg:'#FAFBFA',ink:'#152B2B',teal:'#26766D',light:'#E5EEEB',blue:'#627FAD',muted:'#526464',grid:'#D8E1DE',red:'#B8675C',amber:'#C69A46',white:'#FFFFFF'};
const read=async p=>JSON.parse(await fs.readFile(path.join(ROOT,p),'utf8'));
export const CHART_SLIDES=[12,18,19,22,23];
export const TABLE_SLIDES=[4,9,14,15,16,20,21,28,30,31,32];

export async function build(p){
  if(p.slides.items.length)throw Error('A new empty presentation is required. Preserve the source deck.');
  const content=await read('docs/presentation/verity-technical-content.json');
  const ops=await read('reports/reasonke/operations-final/results.json');
  const depth=await read('reports/reasonke/depth-final/results.json');
  const dev=await read('reports/reasonke/depth-development/retrieval-selection.json');
  const audit=await read('output/presentation/analysis/failure-audit.json');
  const ot=code=>ops.conditions[code].overall;
  const dt=code=>depth.conditions[code].overall;
  const percentage=x=>(100*x).toFixed(2)+'%';
  p.theme.defaultFont='Calibri';
  const tx=(s,t,x,y,w,h,size=26,color=C.ink,bold=false,font='Calibri',align='left')=>{
    const a=s.shapes.add({geometry:'textbox',name:t.slice(0,70),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    a.text=t;a.text.style={fontSize:size,typeface:font,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:0};return a;
  };
  const slide=(n,sub='')=>{
    const s=p.slides.add({width:1280,height:720,background:{fill:C.bg}});
    tx(s,content.slides[n-1].title,64,38,1152,67,42,C.ink,true);
    if(sub)tx(s,sub,66,112,1146,58,24,C.muted);
    tx(s,String(n),1192,670,32,27,18,C.muted);
    return s;
  };
  const foot=(s,t)=>tx(s,t,66,645,1105,51,21,C.muted);
  const head=(s,t,x,y,w)=>tx(s,t,x,y,w,42,29,C.teal,true);
  const box=(s,t,x,y,w,h,{dark=false,size=25,fill=C.light}={})=>{
    const a=s.shapes.add({geometry:'rect',name:t.slice(0,70),position:{left:x,top:y,width:w,height:h},fill:dark?C.ink:fill,line:{fill:dark?C.ink:C.grid,width:1.2}});
    a.text=t;a.text.style={fontSize:size,typeface:'Calibri',color:dark?C.white:C.ink,bold:false,alignment:'center',verticalAlignment:'middle',wrap:'square',insets:14};return a;
  };
  const link=(s,a,b,from='right',to='left',dashed=false)=>{
    const sameRow=Math.abs((a.position.top+a.position.height/2)-(b.position.top+b.position.height/2))<1;
    return s.shapes.connect(a,b,{kind:from==='right'&&to==='left'&&sameRow?'straight':'elbow',fromSide:from,toSide:to,line:{fill:C.teal,width:2,style:dashed?'dashed':'solid'},tail:{type:'arrow',width:'med',length:'med'}});
  };
  const table=(s,values,widths,{x=66,y=190,w=1145,h=390,size=24}={})=>{
    const a=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
    a.borders.assign({fill:C.grid,width:1,style:'solid'});
    values.forEach((row,r)=>row.forEach((_,c)=>{const v=a.getCell(r,c);v.fill=r===0?C.light:C.bg;v.text.style={fontSize:size,typeface:'Calibri',color:C.ink,bold:r===0,verticalAlignment:'middle',insets:10};}));
    return a;
  };
  const plot=(s,categories,series,{x=62,y=190,w=1138,h=410,max,step,percent=false,stacked=false,labels=true,format,valueTitle}={})=>s.charts.add('bar',{
    position:{left:x,top:y,width:w,height:h},categories,
    // Excel snapshots support at most 15 significant digits. Twelve suffice
    // beyond the displayed precision. Frozen source measurements stay unchanged.
    series:series.map((v,i)=>({name:v.name+'\u2002',values:v.values.map(value=>Number(value.toPrecision(12))),fill:v.fill??[C.teal,C.blue,C.amber][i],dataLabels:{deleted:v.labelValues===false,showValue:v.labelValues??labels,position:stacked?'center':'outEnd',numberFormatCode:format??(percent?'0.0%':'0;-0;;'),textStyle:{fontSize:22,bold:true,fill:stacked?C.white:C.ink}}})),
    barOptions:{direction:'column',grouping:stacked?'stacked':'clustered',gapWidth:90},hasLegend:true,
    legend:{position:'bottom',textStyle:{fontSize:22,fill:C.ink}},
    xAxis:{textStyle:{fontSize:22,fill:C.ink},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{min:0,max,majorUnit:step,numberFormatCode:percent?'0%':'0',textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:valueTitle?{text:valueTitle,textStyle:{fontSize:21,fill:C.muted}}:undefined},
    chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0}
  });
  const screenshot=async(s,name,alt)=>{
    const bytes=await fs.readFile(path.join(ROOT,'output/presentation/screenshots',name));
    s.images.add({dataUrl:'data:image/jpeg;base64,'+bytes.toString('base64'),contentType:'image/jpeg',alt,fit:'contain',position:{left:411,top:175,width:807,height:454}});
  };
  const seq=(s,x1,x2,y,label)=>{
    const a=s.shapes.add({geometry:'rect',name:'sequence start '+label,position:{left:x1,top:y,width:1,height:1},fill:'none',line:{fill:'none',width:0}});
    const b=s.shapes.add({geometry:'rect',name:'sequence end '+label,position:{left:x2,top:y,width:1,height:1},fill:'none',line:{fill:'none',width:0}});
    s.shapes.connect(a,b,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:C.teal,width:2},tail:{type:'arrow',width:'sm',length:'med'}});
    tx(s,label,Math.min(x1,x2)+8,y-35,Math.abs(x2-x1)-16,30,20,C.ink,false,'Calibri','center');
  };

  let s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
  tx(s,'Verity',70,120,1120,125,96,C.white,true);
  tx(s,'Governed knowledge editing for\nconnected factual questions',75,286,1110,135,43,C.white);
  tx(s,'Reason-KE++ with versioned memory and bounded retrieval',75,514,1110,66,29,'#C8DED7');

  s=slide(2,'Kaveri Payment Gateway, release 2. One ownership fact changes.');
  const chain=(y,labels,dark=false)=>{
    const xs=[66,369,672,975],widths=[240,240,240,236];
    const nodes=labels.map((t,i)=>box(s,t,xs[i],y,widths[i],104,{dark,size:25}));
    nodes.slice(0,-1).forEach((a,i)=>link(s,a,nodes[i+1]));return nodes;
  };
  head(s,'Initial approved path',66,187,500);
  chain(244,['Kaveri Gateway\nrelease 2','Bengaluru\nPlatform Team','Meera Iyer','Chennai']);
  head(s,'After ownership replacement',66,387,650);
  chain(444,['Kaveri Gateway\nrelease 2','Pune\nReliability Team','Arjun Deshmukh','Hyderabad'],true);
  foot(s,'Hosting remains Bengaluru. Pro and release 1 retain separate owners.\nActual implemented example with fictional services, people and assignments.');

  s=slide(3,'The source of truth is approved memory. The evaluation observes generated answers.');
  const contract=[['Correction transfer','Direct questions, paraphrases and connected paths endorse the applicable value.'],['Locality and scope','Other properties, similarly named services and historical releases retain their facts.'],['Authority and reversibility','Explicit approval, conflict rejection, event lineage and atomic rollback.'],['Accountable evaluation','Neural answers, structured operations, abstentions and errors remain distinguishable.']];
  contract.forEach(([a,b],i)=>{const y=200+i*100;tx(s,a,68,y,335,58,29,C.teal,true);tx(s,b,432,y,770,75,26);});
  foot(s,'Bounded task: short factual QA over controlled operations handbooks, with paths up to three relations.');

  s=slide(4,'The editing route and the retrieval layer answer different design questions.');
  table(s,[['Approach','What changes','Role in this project'],['Conventional RAG','Retrieved context','Matched Qwen instruction baseline'],['Reason-KE++ + memory','Approved context, frozen weights','Deployed reasoning-aware generator'],['Parametric editing','Selected neural parameters','Alternative route, not evaluated here']],[335,330,480],{y:197,h:300,size:25});
  head(s,'Recorded development policy choice',68,535,780);
  tx(s,'Reason-KE++ retrieved 24/24 vs full memory 21/24. Qwen retrieved also 24/24.',68,590,1125,54,26);
  foot(s,'Three development lineages. This selected context delivery before the final test, not model superiority.');

  s=slide(5,'Cold-start supervised learning establishes structure. PPO optimizes the scored response.');
  const data=box(s,'Edited fact +\nportability question',66,212,260,110);
  const teacher=box(s,'Teacher-generated\nstructured examples',428,212,300,110);
  const sft=box(s,'Supervised fine-tuning\nstructured response likelihood',834,212,378,110);
  link(s,data,teacher);link(s,teacher,sft);
  const rollout=box(s,'Policy rollouts\nacknowledge / decompose / action',834,443,378,125,{dark:true});
  const reward=box(s,'Stage-aware reward\nformat + process + outcome',428,443,300,125);
  const ppo=box(s,'PPO policy update\ncritic + KL control',66,443,260,125);
  link(s,sft,rollout,'bottom','top');link(s,rollout,reward,'left','right');link(s,reward,ppo,'left','right');
  const loopA=s.shapes.add({geometry:'rect',name:'PPO feedback start',position:{left:196,top:612,width:1,height:1},fill:'none',line:{fill:'none',width:0}});
  const loopB=s.shapes.add({geometry:'rect',name:'PPO feedback end',position:{left:1023,top:612,width:1,height:1},fill:'none',line:{fill:'none',width:0}});
  s.shapes.connect(ppo,loopA,{kind:'straight',fromSide:'bottom',toSide:'top',line:{fill:C.teal,width:2}});
  s.shapes.connect(loopA,loopB,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:C.teal,width:2}});
  s.shapes.connect(loopB,rollout,{kind:'straight',fromSide:'top',toSide:'bottom',line:{fill:C.teal,width:2},tail:{type:'arrow',width:'med',length:'med'}});
  tx(s,'updated policy',485,579,300,28,21,C.teal,false,'Calibri','center');
  foot(s,'Training data design: rewritten facts and labelled reasoning steps. Verity supplies its own separate evaluation worlds.');

  s=slide(6,'The output format makes intermediate steps machine-readable for training rewards.');
  tx(s,'<think>\n  <acknowledge> ... </acknowledge>\n  <decompose> ... </decompose>\n  <action>\n    [Sub question i] ... \\boxed{aᵢ}\n  </action>\n</think>\n<answer>final answer</answer>',68,198,590,335,23,C.ink,false,'Consolas');
  head(s,'Released reward implementation',735,194,480);
  tx(s,'H  Correct hop count\nD  Aligned subquestion similarity\nA  Correct boxed subanswer fraction\nF  Final-answer token F1',735,251,480,170,26);
  tx(s,'R = 0.2H + 0.4D + 0.4A + F',68,557,1125,54,34,C.teal,true);
  foot(s,'Invalid structure receives −1. Training rewards score generated stages. Chat displays text and its supplied sources.');

  s=slide(7,'Approved state and lexical context selection feed a separately hosted frozen model.');
  head(s,'Deterministic state',68,174,310);head(s,'Deterministic context',443,174,355);head(s,'Neural inference',884,174,335);
  const api=box(s,'Browser + API\nvalidation and operation lock',68,241,305,112);
  const sql=box(s,'SQLite approved memory\nfacts, events and revision',68,458,305,112);
  const selector=box(s,'Bounded lexical selector\nat most four approved facts',443,241,355,112);
  const hist=box(s,'History and prompt assembly\ncompatible complete pairs',443,458,355,112);
  const worker=box(s,'Private Reason-KE++ worker\nauthenticated inference',884,241,328,112,{dark:true});
  const saved=box(s,'API extraction + persistence\nanswer, raw trace and evidence',884,458,328,112);
  link(s,api,sql,'bottom','top');link(s,sql,selector,'right','left');link(s,sql,hist);link(s,selector,worker);link(s,hist,worker,'right','left');link(s,worker,saved,'bottom','top');
  foot(s,'The store defines approved state. Retrieval selects context. The neural worker generates the answer.');

  s=slide(8,'Example request: the contact-office question at knowledge revision r11.');
  const actors=[['Browser',100],['Python API',350],['SQLite',600],['Selector',850],['Model worker',1100]];
  actors.forEach(([name,x])=>{box(s,name,x-90,173,180,56,{size:25});s.shapes.add({geometry:'line',name:name+' lifeline',position:{left:x,top:235,width:0,height:385},line:{fill:C.grid,width:1.5,style:'dashed'}});});
  seq(s,100,350,275,'question');seq(s,350,600,322,'capture r11 + facts');seq(s,350,850,369,'query + approved snapshot');seq(s,850,350,416,'four selected fact IDs');seq(s,350,1100,463,'selected facts + revision-compatible history');seq(s,1100,350,510,'raw generation + final answer');seq(s,350,600,557,'atomic turn + evidence');seq(s,350,100,604,'answer + source display');
  foot(s,'The application lock spans generation. A short SQLite transaction persists the successful turn afterward.');

  s=slide(9,'Canonical key = (entity_id, relation_id, scope_id). Separate keys preserve scope.');
  table(s,[['Stored field','Concrete ownership record'],['entity_id','india-demo-001-service'],['relation_id / scope_id','owner / release-2'],['answer / source','Pune Reliability Team / handbook'],['lineage / version','Same ownership lineage, v2'],['supersedes / approval','v1 predecessor / explicit approval']],[265,470],{x:66,y:197,w:735,h:390,size:24});
  const draft=box(s,'Draft\nbrowser state only',882,197,330,92);
  const active=box(s,'Approved active v1\nunique represented key',882,354,330,92);
  const newer=box(s,'Replacement active v2\nv1 becomes superseded',882,511,330,92);
  link(s,draft,active,'bottom','top');link(s,active,newer,'bottom','top');
  foot(s,'Aliases are metadata. A partial unique index permits at most one active version per canonical key.');

  s=slide(10,'State, revision and both audit events commit together, or all writes roll back.');
  const pre=box(s,'Validate active replacement\nand same-key predecessor',66,197,345,100);
  const begin=box(s,'BEGIN IMMEDIATE\nreserved writer lock',470,197,305,100,{dark:true});
  const finish=box(s,'Withdraw v2, restore v1\none new revision, two events',834,197,378,100);
  link(s,pre,begin);link(s,begin,finish);
  tx(s,'try:\n  advance_revision_once()\n  withdraw(replacement)\n  restore(predecessor)\n  write_both_audit_events()\n  COMMIT\nexcept: ROLLBACK',68,350,600,255,24,C.ink,false,'Consolas');
  head(s,'Failure and history rules',765,348,445);
  tx(s,'Competing approval returns HTTP 409.\nInjected late-event failure restores state.\nHistorical replay leaves live state intact.\nKnowledge changes invalidate context caches.',765,410,445,200,25);
  foot(s,'These transaction guarantees concern represented identities and operational consistency.');

  s=slide(11,'BM25-style ranking uses approved question + scope + answer text, with fact-ID tie breaks.');
  tx(s,'S = top2(rank(question, memory))\nfrontier = answers(S)\nrepeat at most 3 rounds:\n  stop if |S| = 4 or no unseen facts\n  f = top1(rank(frontier, unseen))\n  S = S + [f]\n  frontier = answer(f)\nreturn S',68,207,715,340,24,C.ink,false,'Consolas');
  head(s,'Concrete expansion',863,203,350);
  tx(s,'Question seeds\nHosting + ownership\n\nPune Reliability Team\nOn-call contact record\n\nArjun Deshmukh\nOffice-city record',863,261,350,310,26);
  foot(s,'Four-fact context cap. No expected answer, gold support ID or question-depth metadata enters retrieval.');

  s=slide(12,'Development support completeness, 32 questions per depth group. No model generation.');
  const base=dev.baseline_two_stage_k4.by_depth;
  const chosen=dev.candidate_family.find(v=>v.cap===4&&v.seed_count===2);
  plot(s,['Locality','1 hop','2 hops','3 hops'],[{name:'Two-stage',values:[0,1,2,3].map(k=>base[String(k)].complete),fill:C.blue},{name:'Iterative',values:[0,1,2,3].map(k=>chosen.by_depth[String(k)].complete),fill:C.teal}],{x:62,y:196,w:790,h:410,max:36,step:8,valueTitle:'Complete paths'});
  head(s,'Complete paths',893,203,320);
  tx(s,'96/128 original\n128/128 iterative',893,260,320,115,31,C.ink,true);
  tx(s,'Six declared candidates.\nTwo display states checked.\nSelected: two seeds, four facts.',893,424,320,160,26,C.muted);
  foot(s,'Selection precedes neural profiling and final freeze. Development retrieval recall is not final answer accuracy.');

  s=slide(13,'The latest user turn contains approved facts and the current question.');
  tx(s,'[Updated Information]\nJSON: id, question, answer, scope, source\n\n[Query]\nIn which city is the owner team’s\non-call contact’s office?',68,208,650,220,25,C.ink,false,'Consolas');
  const visible=box(s,'Stored conversation\nall prior revisions remain visible',825,203,387,92);
  const allowed=box(s,'Compatible complete pairs\ncurrent revision and model only',825,372,387,92);
  link(s,visible,allowed,'bottom','top');
  head(s,'Matched historical comparison',68,480,690);
  tx(s,'Replay approved memory at the requested revision.\nGenerate both sides with the same pre-question history.\nExclude correction-conditioned pairs from both sides.',68,540,1125,101,27);
  foot(s,'At most eight history messages and 6,000 characters. A comparison generates new answers and does not rewrite stored turns.');

  s=slide(14,'Exact checkpoint IDs and revisions stay pinned. Corrections change memory, not weights.');
  table(s,[['Worker setting','Measured deployment'],['Parameters per checkpoint','7,615,616,512'],['Weight storage / compute','NF4 double quantization / FP16'],['Attention / decoding','SDPA / greedy, one beam'],['Input / output ceiling','4,096 / 512 tokens'],['Device','T4, one loaded model at a time'],['Load peak allocated CUDA','6.59 GiB']],[300,390],{x:66,y:192,w:690,h:418,size:23});
  head(s,'Answer extraction',816,197,398);
  tx(s,'<think> ... </think>\n<answer>Hyderabad</answer>',816,262,398,98,24,C.ink,false,'Consolas');
  tx(s,'One nonempty answer block:\nextract the final span.\n\nOtherwise disclose raw fallback.\nPersist raw output, sources,\nrevision, tokens and latency.',816,393,398,212,25);
  foot(s,'Measured on T4. CUDA allocation excludes other device usage. Extraction preserves raw output and format flags.');

  s=slide(15,'Split edit lineages before augmenting related questions.');
  table(s,[['Dataset property','Operations study','Depth challenge'],['Authored fact worlds','48','48'],['Development / final lineages','16 / 32','16 / 32'],['Development / final questions','128 / 256','128 / 256'],['Facts per world','9','12'],['Maximum tested path','2 hops','3 hops'],['Final wording bank','Shared template families','Separate dev/final banks']],[485,330,330],{y:193,h:403,size:24});
  foot(s,'Automated graph/label review, without independent human semantic review. Simple values and relation vocabulary can repeat.');

  s=slide(16,'Updated baselines receive the same approved information. Retrieved fact IDs match across generators.');
  table(s,[['Main condition','Generator','Supplied context','Assigned'],['R0 diagnostic','Reason-KE++','Initial full memory','256'],['R-RAG','Reason-KE++','Updated retrieved facts','256'],['R-ALL','Reason-KE++','Full updated memory','256'],['G-RAG','Qwen','Same retrieved fact IDs','256'],['G-ALL','Qwen','Full updated memory','256']],[265,255,450,175],{y:191,h:343,size:24});
  tx(s,'Original: 5 × 256 + 2 × 16 oracle = 1,312 generations',68,552,1125,40,29,C.teal,true);
  tx(s,'Separate depth: 2 × (256 main + 16 oracle) = 544 generations',68,600,1125,39,28);
  foot(s,'Oracle supplies labelled supporting facts to a preselected smaller subset. It does not replace main retrieval accuracy.');

  s=slide(17,'The primary unit is an assigned question. The uncertainty cluster is an edit lineage.');
  head(s,'Whole endorsed-answer scoring',68,199,530);
  tx(s,'accuracy = correct / all assigned\ncoverage = answered / all assigned\n\nUNKNOWN, errors and unexecuted cases\nreceive zero primary accuracy credit.',68,261,620,223,27);
  head(s,'Paired lineage bootstrap',784,199,430);
  tx(s,'Resample 32 whole lineages.\nKeep related variants together.\nCompare matched conditions.\n10,000 draws, fixed seed 42.',784,261,430,223,27);
  tx(s,'Preservation: initially correct locality subset.\nFactual locality accuracy: every locality question.',68,547,1125,83,29,C.teal,true);
  foot(s,'Fixed aliases score the requested subject, relation and final span. Extra target mentions do not earn automatic credit.');

  s=slide(18,'All five main conditions contain 256 assigned questions.');
  const codes=['R0','R-RAG','R-ALL','G-RAG','G-ALL'];
  plot(s,['Initial R','Retrieved R','Full R','Retrieved Q','Full Q'],[{name:'Correct',values:codes.map(c=>ot(c).counts.correct),fill:C.teal},{name:'Strict mismatch',values:codes.map(c=>ot(c).counts.wrong),fill:C.red},{name:'UNKNOWN',values:codes.map(c=>ot(c).counts.abstain),fill:C.amber,labelValues:false}],{x:62,y:186,w:1138,h:423,max:280,step:64,stacked:true,valueTitle:'Assigned outcomes'});
  foot(s,'Retrieved gain: +5.86 pp, 95% lineage interval +3.91 to +8.20.\nR = Reason-KE++. Q = Qwen. UNKNOWN: retrieved Q 15, full Q 2. Errors / unexecuted: 0.');

  s=slide(19,'64 assigned main questions per group and checkpoint.');
  plot(s,['Locality','1 hop','2 hops','3 hops'],[{name:'Reason-KE++',values:[0,1,2,3].map(k=>depth.conditions['R-RAG'].by_depth[String(k)].counts.correct)},{name:'Qwen',values:[0,1,2,3].map(k=>depth.conditions['G-RAG'].by_depth[String(k)].counts.correct)}],{x:62,y:191,w:790,h:420,max:72,step:16,valueTitle:'Correct answers'});
  head(s,'Main final answers',892,195,320);
  tx(s,'256/256 vs 248/256',892,250,320,75,31,C.ink,true);
  tx(s,'+3.125 pp\n95% interval +1.56 to +5.08',892,372,320,104,26);
  tx(s,'Three-hop difference: 1/64.\nIts interval includes zero.',892,528,320,83,25,C.muted);
  foot(s,'All eight Qwen nonsuccesses are UNKNOWN. Separate preselected oracle diagnostics: 16/16 per model.');

  s=slide(20,'All 25 strict wrong answers had required support and matched another supplied fact’s value.');
  const cause=audit.summary['operations:R-ALL'].exact_off_target_fact_roles;
  table(s,[['Endorsed off-target fact','Cases'],['Pro-service owner',String(cause.pro_service_owner)],['Release-2 owner',String(cause.release_2_owner)],['Unrelated office city',String(cause.unrelated_office_city)],['Previous owner contact',String(cause.previous_owner_contact)],['Hosting region',String(cause.hosting_region)],['Contact instead of owner',String(cause.current_owner_contact)]],[410,115],{x:66,y:192,w:525,h:415,size:24});
  head(s,'Garnet Relay, release 2',690,192,520);
  tx(s,'Requested fact f1\nSilver unit 77',690,253,520,100,29,C.teal,true);
  tx(s,'Pro-service fact f5\nSaffron unit 137',690,381,520,100,29,C.red,true);
  tx(s,'The explanation names f5 as if it\nanswered the requested service.',690,522,520,83,26);
  foot(s,'Post-hoc output association diagnosis, not proof of an internal attention mechanism. No inference or scoring changes.');

  s=slide(21,'Failures can occur after complete supporting facts reach the prompt.');
  table(s,[['Observed cohort','Correct','UNKNOWN','Population'],['Original two-hop paraphrases','17','15','32'],['Depth direct two-hop template 1','2','7','9'],['Other depth direct two-hop templates','23','0','23'],['Depth three-hop paraphrases','31','1','32']],[595,175,190,185],{y:191,h:322,size:24});
  head(s,'Full-memory Qwen has a different error profile',68,525,1125);
  tx(s,'10 off-target values + 2 correct chains with extra text + 1 contaminated chain.\nAll 13 remain strict mismatches. Two additional UNKNOWN responses.',68,578,1125,63,25);
  foot(s,'Frozen, nonrandomized wording groups. All 63 updated-method nonsuccesses received their required supporting facts.');

  s=slide(22,'Reason-KE++ original-study locality: preservation and factual accuracy use different denominators.');
  plot(s,['Baseline-correct\nretention (n=124)','All locality\ncorrect (n=128)'],[{name:'Retrieved context',values:[124/124,128/128]},{name:'Full memory',values:[117/124,119/128]}],{x:62,y:190,w:760,h:420,max:1.08,step:.25,percent:true,valueTitle:'Fraction correct'});
  head(s,'Supplied required support',862,195,350);
  tx(s,'Original: 320/320 facts\n64/64 complete two-hop paths\n\nDepth: 448/448 facts\n256/256 complete main paths',862,260,350,256,26);
  tx(s,'Shared selector per study.\nNo independent duplicate\nretrieval sample per model.',862,546,350,85,23,C.muted);
  foot(s,'Depth locality: 64/64 correct per model. It has no initial-base condition for an independent preservation estimate.');

  s=slide(23,'Warm pipeline timing, n=256 generated requests per condition, including abstentions.');
  const lat=[ot('R-RAG').generated_latency,ot('G-RAG').generated_latency,dt('R-RAG').generated_latency,dt('G-RAG').generated_latency];
  plot(s,['Ops p50','Ops p95','Depth p50','Depth p95'],[{name:'Reason-KE++',values:[lat[0].p50_seconds,lat[0].p95_seconds,lat[2].p50_seconds,lat[2].p95_seconds]},{name:'Qwen',values:[lat[1].p50_seconds,lat[1].p95_seconds,lat[3].p50_seconds,lat[3].p95_seconds]}],{x:62,y:191,w:790,h:420,max:22,step:5,format:'0.00',valueTitle:'Seconds'});
  head(s,'Retrieved answer coverage',892,195,320);
  tx(s,'Original\n100% vs 94.14%\n\nDepth\n100% vs 96.88%',892,255,320,245,29);
  tx(s,'Median output tokens\nOps: 204 vs 5.5\nDepth: 212 vs 6',892,538,320,101,24,C.muted);
  foot(s,'Includes selection, local IPC and generation. Excludes cold load and browser/network. R = Reason-KE++. Q = Qwen.');

  s=slide(24,'A draft changes nothing until explicit approval.');
  head(s,'Authority is visible',68,197,315);
  tx(s,'Active predecessor\n\nReplacement answer\n\nExplicit scope\n\nRecorded source',68,269,315,288,28);
  await screenshot(s,'verity-knowledge-editor.jpg','Genuine Verity editor showing an active Bengaluru record beside an unapproved Pune replacement draft');
  foot(s,'Verity correction editor. Browser-local replacement draft over the restored r11 workspace.');

  s=slide(25,'The audit timeline preserves what changed at each revision.');
  tx(s,'r9  Initial handbook\nr10  Replace ownership\nr11  Atomic rollback',68,203,315,165,29,C.teal,true);
  tx(s,'14/14 neural answers\n12 consistency checks\n\nOne fictional display lineage,\nseparate from held-out tests.',68,414,315,202,27);
  await screenshot(s,'technical-v2/verity-history.jpg','Genuine Verity history showing r10 publication/supersession and r11 withdrawal/restoration, from the audited saved receipt');
  foot(s,'Recorded application acceptance, one fictional display lineage. Displayed history comes from the audited receipt.');

  s=slide(26,'The answer’s revision and the current memory revision can correctly differ.');
  head(s,'Stored response',68,198,315);
  tx(s,'Generated at r10\nHyderabad\n\nCurrent memory: r11\nRestored Chennai path',68,264,315,216,28);
  tx(s,'New questions use the\ncurrent snapshot.\nStored messages retain\ntheir original evidence.',68,515,315,112,25,C.muted);
  await screenshot(s,'technical-v2/verity-recorded-r10.jpg','Actual recorded Hyderabad model answer at r10 displayed while restored live memory is r11; worker unavailable');
  foot(s,'Recorded r10 model response displayed against restored r11 memory. Fresh generation requires the model worker.');

  s=slide(27,'Aggregate findings remain connected to individual outputs and supplied facts.');
  head(s,'Inspectable failure',68,198,315);
  tx(s,'Requested release 1\nExpected Willow unit 192\n\nReturned Amber unit 72\nThe release-2 owner',68,264,315,221,27);
  tx(s,'Filters change the view.\nEvery assigned outcome\nkeeps its denominator.',68,532,315,98,25,C.muted);
  await screenshot(s,'technical-v2/verity-results-explorer.jpg','Genuine Verity results explorer filters the 25 saved R-ALL wrong answers and expands a wrong-release endorsement');
  foot(s,'Saved experiment records. The explorer supports raw-output inspection and JSON, CSV and scientific-figure exports.');

  s=slide(28,'Concrete software and analysis artifacts connect the product to defensible measurements.');
  table(s,[['Contribution','Implemented mechanism','Reviewable artifact'],['Governed memory','Identity, approval, events, rollback','store.py + workspace.py'],['Context adaptation','Bounded frontier + history rules','retrieval.py + server.py'],['Independent evaluation','Worlds, splits, fixed scoring','datasets + protocols + metrics'],['Evidence and diagnosis','Raw ledgers, replay, 63-case audit','receipts + audit scripts'],['Usable deployment','Chat, editor, explorer, container','static/ + notebooks + CI']],[285,485,375],{y:190,h:399,size:23});
  foot(s,'Reproducibility uses pinned model/data/config identities, atomic per-request records and resume of unexecuted requests only.');

  s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
  tx(s,'A reversible editing platform, evaluated',66,46,1150,70,44,C.white,true);
  tx(s,'Approved corrections transfer across connected facts\nand retain an inspectable, reversible history.',68,178,1120,105,37,C.white);
  tx(s,'Original retrieved: 256/256 vs 241/256\nSeparate depth: 256/256 vs 248/256',68,337,1120,105,33,'#C8DED7');
  tx(s,'Depth median latency: 15.94 s vs 1.12 s.\nStronger observed coverage has a substantial generation cost.',68,494,1120,91,29,C.white);
  tx(s,'Evidence is bounded by small fact worlds, repeated templates and automated review.\nLarger memory, ambiguous scope and fully faithful explanations remain unestablished.',68,623,1120,62,23,'#C8DED7');

  s=slide(30,'The clipped surrogate constrains policy updates. Training reference settings follow below.');
  tx(s,'rₜ(θ) = πθ(oₜ | q, o<ₜ) / πold(oₜ | q, o<ₜ)\n\nJclip = E[min(rₜ Aₜ, clip(rₜ, 1−ε, 1+ε) Aₜ)]',68,201,1125,127,29,C.ink,false,'Calibri');
  table(s,[['Reference setting','SFT','PPO'],['Learning rate','1e−5','Actor 1e−6, critic 1e−5'],['Epochs','10','15'],['KL coefficient','—','0.001']],[430,285,430],{y:367,h:200,size:24});
  foot(s,'Paper: αRprocess + (1−α)Routcome, main α=0.5. Pinned reward code: Rprocess + Routcome.\nEqual component weighting with different reward scale relative to KL.');

  s=slide(31,'All strict primary scores stay unchanged after inspection.');
  const mismatchRows=['045','047','005'].map(id=>{
    const row=audit.records.find(r=>r.id===`G-ALL:ops-${id}-two_hop_paraphrase`);
    return [`ops-${id} two-hop paraphrase`,row.expected,row.endorsed];
  });
  table(s,[['G-ALL record','Expected person','Endorsed final span'],...mismatchRows],[365,265,515],{y:198,h:285,size:23});
  tx(s,'Two outputs contain a compatible correct owner/contact chain but add unwanted text.\nOne combines a Pro owner with the correct contact.\nTwo Reason-KE++ wrong finals also contain the expected value in a boxed intermediate step.',68,527,1125,110,26);
  foot(s,'Post-hoc description distinguishes answer-contract and fact-association failures. It does not turn failures into new successes.');

  s=slide(32,'Use the source artifact that supports the question being asked.');
  table(s,[['Claim or mechanism','Primary project evidence'],['Model identity and budgets','configs/reasonke-model.json + instruction-model.json'],['Original frozen comparison','reports/reasonke/operations-final/'],['Separate depth and uncertainty','reports/reasonke/depth-final/'],['Real correction/rollback acceptance','depth-final/india-acceptance.json + host audit'],['All updated-method nonsuccesses','output/presentation/analysis/failure-audit.json'],['Displayed interface provenance','screenshots/technical-v2/provenance.json']],[490,655],{y:192,h:415,size:24});
  foot(s,'Raw outputs, protocol hashes, scoring code and labelled populations remain separate from presentation graphics.');

  if(p.slides.items.length!==content.total_slides)throw Error('Content/deck slide-count mismatch');
  for(const [i,s]of p.slides.items.entries()){
    const c=content.slides[i];
    const sources=c.sources.map(url=>url.startsWith('https:')?url:'https://github.com/ShivamRaghav7/verity/blob/codex/reasonke-integration/'+url);
    s.speakerNotes.text=`Member ${c.speaker}${c.appendix?' (optional backup)':''}\n\n${c.speech}\n\nTechnical detail\n${c.detail}\n\nVisual cue\n${c.cue}\n\nEvidence\n${sources.join('\n')}`;
  }
  return {slides:32,coreSlides:29,modelCalls:0,chartSlides:CHART_SLIDES,tableSlides:TABLE_SLIDES};
}
