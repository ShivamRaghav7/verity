// Apply inside an Artifact Session callback so all edits reach the live viewer.
// Numerical charts use the published fixed-study results, never a model call.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const C = {bg:'#FAFBFA', ink:'#152B2B', teal:'#26766D', muted:'#526464', blue:'#627FAD', grid:'#D8E1DE'};
export async function build(p) {
  if (p.slides.items.length) throw new Error('Use an empty session for the initial build');
  p.theme.defaultFont = 'Calibri';
  const source = 'https://github.com/ShivamRaghav7/verity/tree/ae7bb2010a82fd1d744d008eb7c6f870198aece5/';
  function txt(s,t,x,y,w,h,size=27,color=C.ink,bold=false){
    const a=s.shapes.add({geometry:'textbox',name:t.slice(0,45),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    a.text=t;a.text.style={fontSize:size,typeface:'Calibri',color,bold,autoFit:'none',wrap:'square',insets:0};return a;
  }
  function slide(title,n,sub=''){
    const s=p.slides.add({width:1280,height:720,background:{fill:C.bg}});
    txt(s,title,64,45,1152,64,44,C.ink,true);
    if(sub)txt(s,sub,64,118,1140,60,25,C.muted);
    txt(s,String(n).padStart(2,'0'),1184,665,35,25,17,C.muted);
    return s;
  }
  const notes=(s,t,refs=[])=>s.speakerNotes.text=t+'\n\nEvidence\n'+refs.map(r=>source+r).join('\n');
  const img=async(s,relative,pos,alt)=>s.images.add({blob:new Uint8Array(await fs.readFile(path.join(ROOT,relative))),contentType:'image/png',position:pos,fit:'contain',alt});
  function chart(s,categories,values,max,position){
    return s.charts.add('bar',{
      position,categories,
      series:[{name:'ReasonKE',values:values[0],fill:C.teal},{name:'Qwen',values:values[1],fill:C.blue}],
      barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
      hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:23,fill:C.ink}},
      xAxis:{textStyle:{fontSize:22,fill:C.ink},line:{fill:C.grid,width:1},majorGridlines:null},
      yAxis:{min:0,max,majorUnit:max===280?64:16,numberFormatCode:'0',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:C.grid,width:1},title:{text:'Correct answers',textStyle:{fontSize:20,fill:C.muted}}},
      dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:22,bold:true,fill:C.ink}},
      chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0}
    });
  }
  let s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
  txt(s,'EditGuard',70,145,1090,112,92,'#FFFFFF',true);
  txt(s,'Knowledge Editing and Verification Platform',74,280,1080,80,39,'#FFFFFF');
  txt(s,'Approved corrections for connected factual questions',74,423,1000,65,29,'#C8DED7');
  txt(s,'Project presentation',74,618,700,33,22,'#C8DED7');
  notes(s,'Speaker 1. Introduce the problem and goal. A correction should change affected answers while preserving unrelated facts, with visible history and rollback.');

  s=slide('One change affects connected answers',2,'Fictional India operations example: Kaveri Payment Gateway, release 2');
  txt(s,'BEFORE APPROVAL',70,217,510,40,22,C.muted,true);
  txt(s,'AFTER APPROVAL',687,217,510,40,22,C.muted,true);
  txt(s,'Bengaluru Platform Team',70,282,540,55,33,C.ink,true);
  txt(s,'Pune Reliability Team',687,282,540,55,33,C.teal,true);
  txt(s,'Meera Iyer',70,370,510,50,32);
  txt(s,'Arjun Deshmukh',687,370,510,50,32,C.teal);
  txt(s,'Office in Chennai',70,457,510,50,32);
  txt(s,'Office in Hyderabad',687,457,540,50,32,C.teal);
  txt(s,'Hosting stays in Bengaluru. Older releases keep their own facts.',70,584,1120,70,25,C.muted);
  notes(s,'Speaker 1. The rows are owning team, contact and office city. Only the ownership fact changes. Two independent supporting relations let the model derive the new contact and city. All organisations and people in this handbook are fictional.', ['docs/INDIA_DEMO.md','reports/reasonke/depth-final/india-acceptance.json']);

  s=slide('The model and memory have separate roles',3);
  const rows=[['01','Approved memory','SQLite keeps scoped fact versions, sources and history.'],['02','Context selection','Lexical retrieval expands from the question to connected facts.'],['03','Learned answer generation','The pinned ReasonKE_plus checkpoint generates a short answer.'],['04','Visible evidence','The interface shows supplied facts and the memory revision.']];
  rows.forEach((r,i)=>{const y=164+i*107;txt(s,r[0],68,y,72,48,30,C.teal,true);txt(s,r[1],170,y,970,42,30,C.ink,true);txt(s,r[2],170,y+45,1000,46,25,C.muted);});
  txt(s,'ReasonKE training belongs to the upstream authors. A new edit changes memory.',68,620,1115,40,23,C.muted);
  notes(s,'Speaker 2. Final system: author-published ReasonKE_plus, frozen weights, deterministic bounded lexical selector, approved external memory, private worker. No separate trained scope classifier in this final deployment. No per-edit fine-tuning. Retrieved provenance does not verify every generated sentence. Upstream checkpoint: https://huggingface.co/YukinoKaren/ReasonKE_plus revision ceaedc2ae7c8fed6257953587d991ce550db6106. Research attribution: https://github.com/YukinoshitaKaren/Reason-KE', ['docs/CURRENT_ARCHITECTURE.md','docs/REASONKE_MODEL_CARD.md']);

  s=slide('Every correction has a controlled lifecycle',4);
  const steps=[['Approve','An operator supplies the fact, scope and source.'],['Resolve conflicts','A competing active version requires explicit resolution.'],['Compare history','Generate answers using historical and current memory.'],['Roll back','Restore the predecessor in one transaction.']];
  steps.forEach((r,i)=>{const y=159+i*116;txt(s,r[0],68,y,465,43,31,C.teal,true);txt(s,r[1],68,y+43,470,65,24,C.muted);});
  await img(s,'output/operations-2026-10-27/depth-dashboard-complete.png',{left:600,top:175,width:615,height:386},'Actual EditGuard research results interface');
  txt(s,'Audit trail and evidence dashboard',600,584,605,55,24,C.muted);
  notes(s,'Speaker 2. Approval is explicit. Canonical entity, relation and scope identify the fact. HTTP 409 rejects an unresolved competing approval. Replacement and rollback are atomic. A customer message never silently becomes approved knowledge. Screenshot is the actual saved application interface.', ['docs/CHAT_API.md','docs/CURRENT_ARCHITECTURE.md','editguard_chat/store.py']);

  s=slide('Controlled data makes the edit testable',5,'Two independent fictional studies, reported separately');
  const cols=[[70,'48','authored graph lineages per study'],[466,'32','held out lineages per study'],[862,'256','main questions per condition']];
  cols.forEach(([x,n,t])=>{txt(s,n,x,207,340,90,78,C.teal,true);txt(s,t,x,306,310,78,26);});
  txt(s,'Original study',70,441,500,45,31,C.ink,true);
  txt(s,'9 facts per world. Direct, paraphrased and two hop questions, plus locality controls.',70,491,512,125,26,C.muted);
  txt(s,'Separate depth study',687,441,520,45,31,C.ink,true);
  txt(s,'12 facts per world. Adds three hop paths and separate wording banks.',687,491,520,125,26,C.muted);
  txt(s,'Split lineages before augmentation. Labels have automated audits, without independent human review.',70,631,1110,45,21,C.muted);
  notes(s,'Speaker 3. Each study has 16 development and 32 final lineages. Eight correlated question variants per final lineage yield 256 main questions per condition. Templates repeat within splits. The original shares wording across splits; the depth extension uses separate wording banks, but both remain controlled and limited. These datasets are independent of upstream benchmark data.', ['docs/OPERATIONS_DATASET_CARD.md','docs/DEPTH_DATASET_CARD.md','reports/reasonke/depth-final/README.md']);

  s=slide('Matched comparisons and frozen scoring',6);
  txt(s,'Same approved information',70,176,1100,45,32,C.teal,true);
  txt(s,'Both updated models receive the same facts and context selection policy.',70,224,1100,55,28);
  txt(s,'Same output contract',70,315,1100,45,32,C.teal,true);
  txt(s,'Greedy generation. Fixed budgets. Abstentions receive zero primary credit.',70,364,1100,55,28);
  txt(s,'Separate failure sources',70,454,1100,45,32,C.teal,true);
  txt(s,'Measure retrieval, gold support diagnostics and generated answers separately.',70,502,1100,60,28);
  txt(s,'1,312 original requests and 544 depth requests completed under separate frozen protocols.',70,617,1100,50,24,C.muted);
  notes(s,'Speaker 3. Original: five main conditions times 256 plus two oracle diagnostics times 16 equals 1,312. Depth: two models times 256 main plus 16 oracle equals 544. R0 uses stale initial facts only as a diagnostic, not as the updated RAG baseline. T4, pinned checkpoints, NF4 quantization and matched answer settings. Gold-support diagnostic IDs are separate from main retrieval, never pooled. No final test tuning.', ['reports/reasonke/operations-final/protocol.json','reports/reasonke/depth-final/protocol.json','reports/reasonke/operations-final/results.json','reports/reasonke/depth-final/results.json']);

  s=slide('Selected context improves this ReasonKE result',7,'Original operations study, 256 assigned questions in each condition');
  chart(s,['Updated retrieval','Full updated memory'],[[256,231],[241,241]],280,{left:60,top:184,width:805,height:406});
  txt(s,'15 Qwen abstentions',903,223,300,83,32,C.teal,true);
  txt(s,'with updated retrieval',903,306,300,70,25,C.muted);
  txt(s,'25 ReasonKE errors',903,421,300,83,32,C.ink,true);
  txt(s,'with full updated memory',903,504,300,75,25,C.muted);
  txt(s,'Every updated condition receives the correction. The model comparison reverses under full memory.',70,628,1125,48,22,C.muted);
  notes(s,'Speaker 4. Original main results: ReasonKE retrieval 256 correct/256, Qwen retrieval 241/256 with15 UNKNOWN. ReasonKE full memory231/256 with25 strict wrong. Qwen full241/256 with13strictwrong+2abstentions. Some strict wrong outputs include formatting errors, so do not label every Qwen mismatch factually false. The original initial diagnostic136/256 is intentionally omitted from this updated-model chart to keep information access fair. ReasonKE retrieval minus Qwen retrieval: +5.859375pp,95%lineagebootstrap[3.90625,8.203125].',[ 'reports/reasonke/operations-final/results.json','reports/reasonke/operations-final/FAILURE_ANALYSIS.md']);

  s=slide('Depth accuracy has a substantial speed tradeoff',8,'Separate depth study, 64 assigned questions per group and model');
  chart(s,['Locality','1 hop','2 hops','3 hops'],[[64,64,64,64],[64,64,57,63]],72,{left:60,top:186,width:792,height:400});
  txt(s,'256 / 256',891,208,320,55,39,C.teal,true);
  txt(s,'ReasonKE main answers',891,269,315,55,25,C.muted);
  txt(s,'248 / 256',891,354,320,55,39,C.ink,true);
  txt(s,'Qwen, eight abstentions',891,414,315,65,25,C.muted);
  txt(s,'Warm median latency',891,517,320,40,24,C.muted);
  txt(s,'15.94 s vs 1.12 s',891,558,330,47,31,C.ink,true);
  txt(s,'Main difference +3.125 pp, 95% lineage interval +1.56 to +5.08 pp. Three hop interval includes zero.',70,630,1115,46,21,C.muted);
  notes(s,'Speaker 4. Both main methods received complete required support. ReasonKE256/256,Qwen248/256with8UNKNOWN. Threehop64/64and63/64. Both separate preselected oracle diagnostics16/16, but none of the8mainQwenfailure IDs overlapthat diagnostic, so do notclaimoraclefixedthosefailures. Mainpaired95%lineagebootstrap1.5625to5.078125pp. Threehopdifference1.5625ppCI0to4.6875includeszero. Warm p50/p95 ReasonKE15.9358/19.2828s,Qwen1.1190/1.3847s,256requestspermodel. No monotonicdepthdeclineclaim.', ['reports/reasonke/depth-final/results.json','reports/reasonke/depth-final/FAILURE_ANALYSIS.md']);

  s=slide('The website makes changes inspectable',9);
  await img(s,'output/operations-2026-10-27/india-three-hop-complete.png',{left:64,top:147,width:485,height:507},'Actual saved three hop conversation showing Chennai before the edit');
  const demo=[['1','Ask the original office question'],['2','Approve the ownership replacement'],['3','Check paraphrases and unaffected facts'],['4','Compare history and roll back']];
  demo.forEach(([n,t],i)=>{txt(s,n,605,181+i*91,45,43,31,C.teal,true);txt(s,t,670,181+i*91,530,73,29);});
  txt(s,'14/14 model answers and 12 consistency checks passed in the finite website workload.',605,567,595,93,24,C.muted);
  notes(s,'Speaker 5. The screenshot is a saved real UI response before the correction: Chennai. A fresh generation after the approved replacement should answerHyderabad, and rollbackshouldrestoreChennai. The separate recorded acceptanceworkloadpassed14/14 neuralanswersand12consistencychecks. It representsonefictionaldisplaylineage,not14newheldoutfacts. The displayed screenshotitselfwasanadditionalunscoredmanualdemoanddoesnotexpandanydenominator. Optional live demo is outside the main talk timing. If the GPU is offline, identify the screen as saved evidence rather than pretending to generate live.', ['docs/INDIA_DEMO.md','reports/reasonke/depth-final/india-acceptance.json','reports/reasonke/depth-final/india-host-audit.json']);

  s=slide('What the evidence supports',10);
  txt(s,'A working correction and rollback platform',70,179,1110,60,37,C.teal,true);
  txt(s,'Approved memory supports connected answers with visible history.',70,248,1110,68,29);
  txt(s,'100% describes these controlled main tests',70,369,1110,60,37,C.ink,true);
  txt(s,'Small fact worlds, repeated templates and complete retrieved support limit generalisation.',70,438,1110,90,29,C.muted);
  txt(s,'Our contribution',70,572,340,40,26,C.teal,true);
  txt(s,'Governance, context selection, independent data, reproducible evaluation and the application.',70,614,1100,55,25,C.ink);
  notes(s,'Speaker 5. Conclude with bounded findings. The checkpoint comes from the upstream authors. We contribute the integrated governed product and its evidence. Real-world validation still needs independently reviewed operationalquestions, larger memories, ambiguityandmissingfacts. Correctfinalanswersdonotcertify every explanation. Do not promise100%arbitraryquestionaccuracy.', ['docs/CONTRIBUTIONS_AND_EVIDENCE.md','reports/reasonke/depth-final/FAILURE_ANALYSIS.md']);
  const script=await fs.readFile(path.join(ROOT,'docs/presentation/SPEAKER_SCRIPT.md'),'utf8');
  const speeches=[...script.matchAll(/^> (.+)$/gm)].map(m=>m[1]);
  if(speeches.length!==10)throw new Error('Expected ten exact speech blocks');
  p.slides.items.forEach((slide,i)=>{
    const refs=slide.speakerNotes.text.split('\n\nEvidence\n')[1]||'';
    slide.speakerNotes.text='Member '+(Math.floor(i/2)+1)+'\n\n'+speeches[i]+'\n\nEvidence\n'+refs;
  });
  p.slides.items[2].speakerNotes.append('\nhttps://github.com/YukinoshitaKaren/Reason-KE\nhttps://huggingface.co/YukinoKaren/ReasonKE_plus');
  return {slides:p.slides.items.length,modelCalls:0};
}
