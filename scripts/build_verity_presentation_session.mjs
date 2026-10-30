// Build only inside an Artifact Session callback. No inference or score changes.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const C={bg:'#FAFBFA',ink:'#152B2B',teal:'#26766D',muted:'#526464',blue:'#627FAD',grid:'#D8E1DE'};
const URL='https://github.com/ShivamRaghav7/verity/tree/ae7bb2010a82fd1d744d008eb7c6f870198aece5/';

export async function build(p){
  if(p.slides.items.length)throw Error('Empty presentation required');
  p.theme.defaultFont='Calibri';
  const tx=(s,t,x,y,w,h,size=27,color=C.ink,bold=false,font='Calibri')=>{
    const a=s.shapes.add({geometry:'textbox',name:t.slice(0,55),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    a.text=t;a.text.style={fontSize:size,typeface:font,color,bold,autoFit:'none',wrap:'square',insets:0};return a;
  };
  const sl=(title,n,sub='')=>{
    const s=p.slides.add({width:1280,height:720,background:{fill:C.bg}});
    tx(s,title,64,42,1150,65,44,C.ink,true);
    if(sub)tx(s,sub,64,117,1145,57,24,C.muted);
    tx(s,String(n),1180,669,40,25,17,C.muted);return s;
  };
  const caption=(s,t)=>tx(s,t,66,625,1130,52,21,C.muted);
  const refs=[];
  const cite=(s,paths)=>{refs.push(paths.map(x=>x.startsWith('https:')?x:URL+x).join('\n'));};
  const table=(s,values,widths,y=190,h=355)=>{
    const t=s.tables.add({rows:values.length,columns:values[0].length,left:66,top:y,width:1145,height:h,columnWidths:widths,values});
    t.borders.assign({fill:C.grid,width:1,style:'solid'});
    for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
      const a=t.getCell(r,c);a.fill=r===0?'#E5EEEB':C.bg;
      a.text.style={fontSize:25,typeface:'Calibri',color:C.ink,bold:r===0,verticalAlignment:'middle',insets:10};
    }
    return t;
  };
  const chart=(s,categories,values,max,pos)=>s.charts.add('bar',{
    position:pos,categories,series:[{name:'ReasonKE',values:values[0],fill:C.teal},{name:'Qwen',values:values[1],fill:C.blue}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:true,
    legend:{position:'bottom',textStyle:{fontSize:22,fill:C.ink}},
    xAxis:{textStyle:{fontSize:22,fill:C.ink},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{min:0,max,majorUnit:max===280?64:16,numberFormatCode:'0',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:'Correct answers',textStyle:{fontSize:20,fill:C.muted}}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:22,bold:true,fill:C.ink}},
    chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0}
  });

  let s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
  tx(s,'Verity',70,140,1110,120,96,'#FFFFFF',true);
  tx(s,'Knowledge Editing and Verification Platform',74,285,1090,75,39,'#FFFFFF');
  tx(s,'Versioned memory, bounded retrieval and a learned reasoner',74,432,1090,70,29,'#C8DED7');
  tx(s,'Technical project presentation',74,620,1000,34,22,'#C8DED7');
  cite(s,['docs/CURRENT_ARCHITECTURE.md']);

  s=sl('The edit contract and three hop dependence',2,'One approved ownership replacement changes a connected answer path');
  table(s,[['Relation','Initial answer path','Corrected answer path'],['Gateway owner','Bengaluru Platform Team','Pune Reliability Team'],['Owner contact','Meera Iyer','Arjun Deshmukh'],['Contact office','Chennai','Hyderabad']],[275,435,435],200,320);
  tx(s,'Locality controls',66,555,315,40,29,C.teal,true);
  tx(s,'Hosting remains Bengaluru. Pro and release 1 retain separate owners.',350,555,850,66,26);
  caption(s,'Fictional Kaveri Payment Gateway, release 2. Only the ownership record changes.');
  cite(s,['docs/INDIA_DEMO.md','reports/reasonke/depth-final/india-acceptance.json']);

  s=sl('Inference architecture and evidence boundary',3);
  const flow=[['Browser and Python API','Validate the request and serialize memory and generation operations.'],['Revision aware SQLite','Read approved facts and current revision conversation pairs.'],['Bounded context selector','Choose up to four fact records without target answers or gold IDs.'],['Private neural worker','Frozen ReasonKE checkpoint generates the final answer and raw trace.'],['Persisted answer evidence','Store source IDs, memory revision, model revision, tokens and latency.']];
  flow.forEach(([a,b],i)=>{let y=153+i*86;tx(s,a,68,y,1080,39,29,C.teal,true);tx(s,b,68,y+39,1100,43,25);});
  caption(s,'A source establishes supplied context. It does not certify every generated sentence.');
  cite(s,['docs/CURRENT_ARCHITECTURE.md','editguard_chat/reasonke_backend.py']);

  s=sl('Frozen checkpoints and inference configuration',4,'ReasonKE_plus and Qwen2.5 7B Instruct each contain 7,615,616,512 parameters');
  table(s,[['Setting','Fixed configuration'],['Weights','Pinned safetensors, frozen during editing'],['Precision','NF4 double quantization, FP16 compute, SDPA'],['Generation','Greedy decoding, one beam, no sampling'],['Token limits','4,096 input tokens and at most 512 new tokens'],['Hardware','T4, one loaded model at a time']],[300,845],190,375);
  caption(s,'Measured load peak allocated CUDA: 6.59 GiB. Upstream authors trained both checkpoints.');
  cite(s,['docs/REASONKE_MODEL_CARD.md','editguard_reasonke/runtime.py','https://huggingface.co/YukinoKaren/ReasonKE_plus','https://huggingface.co/Qwen/Qwen2.5-7B-Instruct']);

  s=sl('Lexical retrieval with bounded answer expansion',5,'BM25 style ranking over question, scope and answer text. Stable fact ID tie breaks.');
  tx(s,'Original two stage selector',68,193,535,45,30,C.teal,true);
  tx(s,'Rank the question and keep two seeds.\nAppend their answers to the question.\nRerank and fill a four fact context.',68,257,535,207,28);
  tx(s,'Iterative depth selector',687,193,525,45,30,C.teal,true);
  tx(s,'Rank the question and keep two seeds.\nUse selected answers as a frontier.\nAdd the top unseen fact per round.\nStop at four facts or three rounds.',687,257,525,224,28);
  tx(s,'Development complete paths',68,528,540,38,28,C.ink,true);
  tx(s,'96/128 original versus 128/128 iterative\nThree hop paths: 0/32 versus 32/32',68,574,1100,78,26,C.muted);
  cite(s,['editguard_benchmark/retrieval.py','editguard_depth/retrieval.py','docs/DEPTH_RUN_PLAN.md']);

  s=sl('Transactional editing and rollback',6,'Canonical fact key: entity_id, relation_id and scope_id');
  tx(s,'State invariants',68,189,500,45,31,C.teal,true);
  tx(s,'At most one active version per key.\nA replacement names its predecessor.\nCompeting active approval returns 409.\nEvents retain lineage and provenance.',68,255,515,240,28);
  tx(s,'Rollback transaction',685,189,525,45,31,C.teal,true);
  tx(s,'BEGIN IMMEDIATE\nwithdraw(replacement)\nrestore(predecessor)\nrecord_both_events(revision)\nCOMMIT',685,255,525,240,26,C.ink,false,'Consolas');
  caption(s,'Rollback pseudocode. Two audit events share one revision. Historical reads leave live memory unchanged.');
  cite(s,['editguard_chat/store.py','docs/CHAT_API.md','tests/test_chat_store.py']);

  s=sl('Dataset construction and leakage controls',7,'An independently authored controlled demonstration dataset for each study');
  table(s,[['Quantity','Original operations','Separate depth'],['Authored lineages / questions','48 / 384','48 / 384'],['Development / final lineages','16 / 32','16 / 32'],['Main final questions / condition','256','256'],['Facts per world','9','12'],['Maximum evaluated path depth','2 hops','3 hops']],[465,340,340],190,370);
  caption(s,'Split lineages before augmentation. Automated labels. Templates repeat within splits.');
  cite(s,['docs/OPERATIONS_DATASET_CARD.md','docs/DEPTH_DATASET_CARD.md']);

  s=sl('Frozen experiment and scoring protocol',8);
  tx(s,'Original operations',68,167,535,45,31,C.teal,true);
  tx(s,'5 main conditions x 256 questions\n2 oracle diagnostics x 16 questions\n1,312 assigned generations',68,229,540,152,29);
  tx(s,'Separate depth',687,167,525,45,31,C.teal,true);
  tx(s,'2 models x (256 main + 16 oracle)\n544 assigned generations\nShared selected fact IDs per question',687,229,525,152,29);
  tx(s,'Primary accuracy = correct / all assigned',68,440,1125,49,33,C.ink,true);
  tx(s,'Score the endorsed subject, relation and short answer. UNKNOWN receives zero credit.\nSeparate retrieval completeness, gold support diagnostics and generation quality.',68,507,1125,103,26,C.muted);
  caption(s,'Updated baselines receive the same approved facts. Settings and scoring freeze before final inference.');
  cite(s,['reports/reasonke/operations-final/protocol.json','reports/reasonke/depth-final/protocol.json','editguard_benchmark/scoring.py']);

  s=sl('Original study and the full memory failure',9,'256 assigned questions per updated condition. Count every strict mismatch and abstention.');
  chart(s,['Updated retrieval','Full updated memory'],[[256,231],[241,241]],280,{left:60,top:184,width:805,height:405});
  tx(s,'Retrieved Qwen',902,219,310,43,29,C.teal,true);
  tx(s,'15 UNKNOWN\n0 wrong answers',902,276,310,100,28);
  tx(s,'Full memory ReasonKE',902,419,310,80,29,C.ink,true);
  tx(s,'25 strict wrong answers',902,511,310,65,27);
  caption(s,'ReasonKE retrieval: 124/124 baseline correct controls retained, 128/128 total controls correct.');
  cite(s,['reports/reasonke/operations-final/results.json','reports/reasonke/operations-final/FAILURE_ANALYSIS.md']);

  s=sl('Depth results, uncertainty and latency',10,'64 assigned questions per group and model in the separate depth study');
  chart(s,['Locality','1 hop','2 hops','3 hops'],[[64,64,64,64],[64,64,57,63]],72,{left:60,top:184,width:790,height:405});
  tx(s,'Main answers',888,191,320,42,28,C.teal,true);
  tx(s,'256/256 vs 248/256',888,245,320,49,31,C.ink,true);
  tx(s,'Warm p50 / p95',888,336,320,42,27,C.teal,true);
  tx(s,'ReasonKE 15.94 / 19.28 s\nQwen 1.12 / 1.38 s',888,393,320,105,25);
  tx(s,'Eight Qwen abstentions',888,538,320,50,25,C.muted);
  caption(s,'Main gain +3.125 pp, paired lineage 95% interval +1.56 to +5.08. Three hop interval includes zero.');
  cite(s,['reports/reasonke/depth-final/results.json','reports/reasonke/depth-final/FAILURE_ANALYSIS.md']);

  s=sl('Failure analysis and the meaning of 100%',11);
  tx(s,'Observed failure pattern',68,177,1100,43,32,C.teal,true);
  tx(s,'Seven Qwen two hop abstentions share one direct wording template.\nRequired supporting facts were present in all eight depth abstentions.',68,236,1125,106,28);
  tx(s,'Perfect scores have a bounded population',68,383,1125,43,32,C.ink,true);
  tx(s,'32 held out lineages per study, with eight correlated question variants.\nSmall fact worlds and repeated templates with complete retrieved support.\nAutomated labels and no independent human semantic review.',68,441,1125,142,27,C.muted);
  caption(s,'Exact final answers do not prove explanation faithfulness or unrestricted conversation accuracy.');
  cite(s,['reports/reasonke/depth-final/FAILURE_ANALYSIS.md','docs/EXPLANATION_AUDIT.md']);

  s=sl('Application acceptance and project contribution',12);
  tx(s,'Measured correction workflow',68,178,1110,45,33,C.teal,true);
  tx(s,'Initial memory r9. Ownership replacement r10. Atomic rollback r11.\n14/14 actual model answers and 12 consistency checks passed.',68,243,1125,108,29);
  tx(s,'Engineering contribution',68,394,1110,45,33,C.ink,true);
  tx(s,'Versioned memory and history, context selection adaptations, independent data,\nfinite matched evaluation, raw output audits and an inspectable application.',68,455,1125,110,28);
  caption(s,'One fictional display lineage, separate from held out accuracy. Training and the model method belong upstream.');
  cite(s,['reports/reasonke/depth-final/india-acceptance.json','reports/reasonke/depth-final/india-host-audit.json','docs/CONTRIBUTIONS_AND_EVIDENCE.md']);

  const script=await fs.readFile(path.join(ROOT,'docs/presentation/VERITY_SPEAKER_SCRIPT.md'),'utf8');
  const speech=[...script.matchAll(/^> (.+)$/gm)].map(m=>m[1]);
  if(speech.length!==12)throw Error('Expected twelve exact speech blocks');
  const speakers=[1,1,2,2,2,3,3,4,4,4,5,5];
  p.slides.items.forEach((slide,i)=>{slide.speakerNotes.text=`Member ${speakers[i]}\n\n${speech[i]}\n\nEvidence\n${refs[i]}`;});
  await applyWebsiteScreenshots(p);
  return {slides:12,modelCalls:0};
}

// A focused edit for the existing live deck. Other slides, charts and tables stay intact.
export async function applyWebsiteScreenshots(p){
  if(p.slides.items.length!==12)throw Error('Expected the twelve-slide technical deck');
  const set=(s,name,text,position,size,color=C.ink,bold=false,font='Calibri')=>{
    const matches=s.shapes.items.filter(x=>x.name.startsWith(name.slice(0,45)));
    if(matches.length!==1)throw Error(`Expected one existing shape: ${name}`);
    const a=matches[0];a.text=text;a.position=position;
    a.text.style={fontSize:size,typeface:font,color,bold,autoFit:'none',wrap:'square',insets:0};
  };
  const image=async(s,file,alt)=>{
    const bytes=await fs.readFile(path.join(ROOT,'output/presentation/screenshots',file));
    const options={dataUrl:'data:image/jpeg;base64,'+bytes.toString('base64'),contentType:'image/jpeg',alt,fit:'contain',position:{left:421,top:168,width:791,height:445}};
    if(s.images.items.length>1)throw Error('Unexpected pre-existing slide images');
    if(s.images.items.length)s.images.items[0].replace(options);
    else s.images.add(options);
  };
  let s=p.slides.items[5];
  set(s,'State invariants','State invariants',{left:68,top:188,width:325,height:42},29,C.teal,true);
  set(s,'At most one active version per key.\nA replacement names ',
    'One active version per key.\nNamed predecessor.\nHTTP 409 for conflicts.\nLineage and provenance.',
    {left:68,top:247,width:325,height:155},25);
  set(s,'Rollback transaction','Atomic rollback',{left:68,top:425,width:325,height:40},29,C.teal,true);
  set(s,'BEGIN IMMEDIATE\nwithdraw(replacement)\nrestore(predecesso',
    'BEGIN IMMEDIATE\nwithdraw(new)\nrestore(previous)\nwrite_events(revision)\nCOMMIT',
    {left:68,top:478,width:325,height:133},21,C.ink,false,'Consolas');
  set(s,'Rollback pseudocode. Two audit events share one revision.',
    'Current Verity interface with a replacement draft. Approval remains explicit.\nRollback writes two audit events at one revision. Historical reads leave live memory unchanged.',
    {left:66,top:632,width:1100,height:56},21,C.muted);
  await image(s,'verity-knowledge-editor.jpg','Verity correction library and unapproved Pune ownership replacement draft at restored revision 11');

  s=p.slides.items[11];
  set(s,'Measured correction workflow','Measured workflow',{left:68,top:177,width:325,height:48},30,C.teal,true);
  set(s,'Initial memory r9. Ownership replacement r10. Atomic roll',
    'r9  Initial memory\nr10  Ownership edit\nr11  Atomic rollback\n\n14/14 neural answers\n12 consistency checks',
    {left:68,top:245,width:325,height:234},27);
  set(s,'Engineering contribution','Our contribution',{left:68,top:491,width:325,height:43},29,C.ink,true);
  set(s,'Versioned memory and history, context selection adaptatio',
    'Governed memory, retrieval,\nindependent data and\nreproducible evaluation.',
    {left:68,top:545,width:325,height:89},24);
  set(s,'One fictional display lineage, separate from held out acc',
    'Recorded model response in the current UI, not fresh inference. One fictional display lineage.\nSeparate from held-out accuracy. Upstream authors trained the model.',
    {left:66,top:640,width:1100,height:53},21,C.muted);
  await image(s,'verity-recorded-chat.jpg','Verity displays the actual recorded Chennai three-hop model answer after revision 11 rollback; live worker unavailable');

  const script=await fs.readFile(path.join(ROOT,'docs/presentation/VERITY_SPEAKER_SCRIPT.md'),'utf8');
  const speech=[...script.matchAll(/^> (.+)$/gm)].map(m=>m[1]);
  if(speech.length!==12)throw Error('Expected twelve exact speech blocks');
  const provenance='Screenshots: actual local Verity website, captured 2026-10-30. The screenshot workspace restores saved India acceptance responses, not new inference. The worker status remains unavailable.\nhttps://github.com/ShivamRaghav7/verity/blob/codex/reasonke-integration/output/presentation/screenshots/provenance.json';
  p.slides.items[5].speakerNotes.text=`Member 3\n\n${speech[5]}\n\nEvidence\n${URL}editguard_chat/store.py\n${URL}docs/CHAT_API.md\n${provenance}`;
  p.slides.items[11].speakerNotes.text=`Member 5\n\n${speech[11]}\n\nEvidence\n${URL}reports/reasonke/depth-final/india-acceptance.json\n${URL}docs/CONTRIBUTIONS_AND_EVIDENCE.md\n${provenance}`;
  return {slidesChanged:[6,12],websiteScreenshots:2,modelCalls:0};
}
