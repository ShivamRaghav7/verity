// Verity mid-project deck. Native diagrams, charts and table. No inference.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const C={bg:'#F8F6F1',ink:'#123039',teal:'#247A70',coral:'#C97552',muted:'#576969',line:'#D7DFDA',light:'#E7EEEA',white:'#FFFFFF',gold:'#AB8353'};
export const CHART_SLIDES=[10,11];
export const TABLE_SLIDES=[9];

export async function build(p,{allowMissingImages=false}={}){
 const read=async name=>JSON.parse(await fs.readFile(path.join(ROOT,name),'utf8'));
 const content=await read('docs/presentation/verity-midproject-content.json');
 const original=await read('reports/reasonke/operations-final/results.json');
 const depth=await read('reports/reasonke/depth-final/results.json');
 const o=code=>original.conditions[code].overall;
 const d=code=>depth.conditions[code].overall;
 if(p.slides.items.length)throw Error('Use an empty presentation; the old package is preserved.');
 p.theme.defaultFont='Calibri';
 const text=(s,t,x,y,w,h,size=29,color=C.ink,bold=false,font='Calibri',align='left')=>{
  const a=s.shapes.add({geometry:'textbox',name:t.slice(0,80),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  a.text=t;a.text.style={fontSize:size,typeface:font,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:0};return a;
 };
 const slide=(n,sub='',dark=false)=>{
  const s=p.slides.add({width:1280,height:720,background:{fill:dark?C.ink:C.bg}});
  text(s,content.slides[n-1].title,64,37,1152,n===13?110:80,42,dark?C.white:C.ink,true);
  if(sub)text(s,sub,66,n===13?146:119,1145,55,25,dark?'#C9DDD7':C.muted);
  text(s,'Verity',64,675,210,24,18,dark?'#C9DDD7':C.muted);
  text(s,String(n).padStart(2,'0'),1183,675,38,24,18,dark?'#C9DDD7':C.muted,false,'Calibri','right');
  return s;
 };
 const foot=(s,t,dark=false)=>text(s,t,64,627,1148,42,21,dark?'#C9DDD7':C.muted);
 const box=(s,t,x,y,w,h,{dark=false,accent=false,size=28}={})=>{
  const a=s.shapes.add({geometry:'rect',name:t.slice(0,80),position:{left:x,top:y,width:w,height:h},fill:dark?C.ink:accent?C.teal:C.light,line:{fill:dark?C.ink:accent?C.teal:C.line,width:1}});
  a.text=t;a.text.style={fontSize:size,typeface:'Calibri',color:dark||accent?C.white:C.ink,alignment:'center',verticalAlignment:'middle',wrap:'square',insets:13};return a;
 };
 const link=(s,a,b,from='right',to='left')=>s.shapes.connect(a,b,{kind:from==='right'&&to==='left'?'straight':'elbow',fromSide:from,toSide:to,line:{fill:C.teal,width:2.2},tail:{type:'arrow',width:'med',length:'med'}});
 const image=async(s,name,alt,x,y,w,h)=>{
  let bytes;
  try{bytes=await fs.readFile(path.join(ROOT,'output/presentation/midproject',name));}
  catch(e){if(!allowMissingImages)throw e;text(s,'Genuine website capture',x,y,w,h,32,C.muted);return;}
  const contentType=name.endsWith('.png')?'image/png':'image/jpeg';
  s.images.add({dataUrl:`data:${contentType};base64,`+bytes.toString('base64'),contentType,alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
 };
 const chart=(s,categories,series,{x=64,y=205,w=760,h=395,max=280,step=64,valueTitle='Correct answers',format='0',legend=true}={})=>s.charts.add('bar',{
  position:{left:x,top:y,width:w,height:h},categories,
  series:series.map((v,i)=>({name:v.name+'\u2002',values:v.values.map(n=>Number(n.toPrecision(12))),fill:v.fill??[C.teal,C.coral][i],points:v.points,
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:25,bold:true,fill:C.ink}}})),
  barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:legend,
  legend:{position:'bottom',textStyle:{fontSize:23,fill:C.ink}},
  xAxis:{textStyle:{fontSize:24,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},
  yAxis:{min:0,max,majorUnit:step,numberFormatCode:'0',textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:C.line,width:1},title:{text:valueTitle,textStyle:{fontSize:22,fill:C.muted}}},
  chartFill:C.bg,chartLine:{fill:'none',width:0},plotAreaFill:C.bg,plotAreaLine:{fill:'none',width:0}
 });

 let s=p.slides.add({width:1280,height:720,background:{fill:C.ink}});
 text(s,'Verity',70,118,1140,132,106,C.white,true);
 text(s,'Governed knowledge updates\nfor AI assistants',76,295,1120,144,46,C.white);
 text(s,'Mid-project review',76,541,650,46,29,'#CBDDD7');
 text(s,'31 October 2026',76,598,650,35,24,'#CBDDD7');

 s=slide(2,'A historical reference snapshot that an assistant must keep current.');
 text(s,'20 DECEMBER 2021',65,209,525,42,25,C.muted,true);
 text(s,'Government of India',65,258,530,61,39,C.ink,true);
 text(s,'27 JANUARY 2022',65,367,525,42,25,C.teal,true);
 text(s,'Talace Private Limited',65,416,530,61,39,C.teal,true);
 text(s,'Parent: Tata Sons',65,491,530,53,30,C.ink);
 await image(s,'assets/air-india-handover-pib.jpg','Official photograph in the PIB Air India handover announcement of 27 January 2022',640,204,572,382);
 foot(s,'Government and Tata primary sources. Ownership is scoped to the 2022 handover.');

 s=slide(3,'One approved memory, a bounded context, an inspectable response.');
 const steps=['Question','API\nCapture revision','Retrieve\nApproved facts','Fixed generator','Answer\nSource evidence'].map((t,i)=>box(s,t,64+i*236,230,200,138,{accent:i===3,size:27}));
 for(let i=0;i<steps.length-1;i++)link(s,steps[i],steps[i+1]);
 const admin=box(s,'Operator\napproval',64,477,200,104,{size:27});
 const memory=box(s,'SQLite\nVersions and audit events',300,477,436,104,{size:28});
 link(s,admin,memory);link(s,memory,steps[2],'top','bottom');
 text(s,'ReasonKE_plus / Qwen2.5-7B\nPinned 4-bit inference on T4',775,443,438,100,26,C.muted);
 foot(s,'Each saved response retains the supplied fact IDs, model identity and memory revision.');

 s=slide(4,'Follow approved answer text to the next relevant record.');
 const nodes=['Air India','Talace','Tata Sons','Mumbai'].map((t,i)=>box(s,t,64+i*296,231,260,115,{accent:i===3,size:34}));
 for(let i=0;i<3;i++)link(s,nodes[i],nodes[i+1]);
 ['selected owner','parent company','official contact city'].forEach((v,i)=>text(s,v,64+i*296,365,260,42,24,C.muted,false,'Calibri','center'));
 text(s,'Rank the question.\nExpand through selected answers.\nKeep a four-fact context budget.',66,455,800,144,31,C.ink);
 text(s,'2 seeds\n4 facts',962,453,250,143,42,C.teal,true);
 foot(s,'The original study uses two-stage retrieval. The depth challenge uses iterative expansion.');

 s=slide(5,'Approval and recovery are part of the product.');
 const life=['Draft\nSource and scope','Approve\nActive record','Supersede\nExplicit predecessor','Rollback\nRestore prior state'].map((t,i)=>box(s,t,64+i*296,238,260,147,{accent:i===1,size:29}));
 for(let i=0;i<3;i++)link(s,life[i],life[i+1]);
 text(s,'One active version per represented fact key',65,460,1140,55,34,C.ink,true);
 text(s,'State, revision and audit events commit together.',65,534,1140,55,31,C.teal);
 foot(s,'Competing approval requires explicit resolution. Earlier versions and messages remain inspectable.');

 s=slide(6,'Air India handover example, with the dated official source.');
 await image(s,'screenshots/air-india-editor.jpg','Genuine Verity correction editor with the official Air India handover source',64,182,890,440);
 text(s,'Dated source\n\nExplicit scope\n\nApprove replacement',988,222,224,312,28,C.ink,true);
 foot(s,'Recorded website walkthrough. The selected local snapshot advances from 2021 to the 2022 handover.');

 s=slide(7,'The same sourced example, through the actual chat workflow.');
 await image(s,'screenshots/air-india-chat.jpg','Genuine model-generated connected answer using sourced Air India facts',64,182,420,440);
 text(s,'Parent of the selected owner',535,209,677,55,31,C.ink,true);
 text(s,'Air India → Talace → Tata Sons',535,291,677,70,32,C.teal);
 text(s,'Tata Sons Private Limited',535,390,677,74,38,C.ink,true);
 text(s,'Actual generated answer · revision 8\nSupplied source records remain inspectable.',535,500,677,91,27,C.muted);
 foot(s,'Recorded website response. Supplied evidence and the answer’s memory revision remain visible.');

 s=slide(8,'A source-backed update remains reversible and traceable.');
 await image(s,'screenshots/air-india-history.jpg','Genuine Verity version and rollback history for the sourced Air India snapshot',64,182,420,440);
 text(s,'r6  Approve the handover',535,216,677,70,34,C.ink,true);
 text(s,'r7  Restore the earlier snapshot',535,323,677,70,34,C.teal,true);
 text(s,'r8  Reapply the handover',535,430,677,70,34,C.ink,true);
 text(s,'Competing approval rejected: HTTP 409\nEach stored answer keeps its original revision.',535,535,677,65,26,C.muted);
 foot(s,'Rollback restores an earlier local reference snapshot. The historical event itself is unchanged.');

 s=slide(9,'Controlled fictional operations handbooks, with explicit ground truth.');
 const values=[['Study','Facts / world','Question depth','Main questions / condition'],['Original operations','9','Up to 2 hops','256'],['Depth challenge','12','Up to 3 hops','256']];
 const t=s.tables.add({rows:3,columns:4,left:64,top:222,width:1152,height:218,columnWidths:[340,204,255,353],values});
 t.borders.assign({fill:C.line,width:1,style:'solid'});
 values.forEach((r,i)=>r.forEach((_,j)=>{const cell=t.getCell(i,j);cell.fill=i===0?C.light:C.bg;cell.text.style={fontSize:27,typeface:'Calibri',color:C.ink,bold:i===0,verticalAlignment:'middle',insets:12};}));
 text(s,'48 worlds per study: 16 development, 32 final lineages',65,478,1145,58,32,C.ink,true);
 text(s,'Matched facts and budgets. Every assigned outcome stays in the score.',65,553,1145,52,28,C.teal);
 foot(s,'Lineages split before variants. Labels were automatically audited. The sourced demo is a separate population.');

 s=slide(10,'Retrieved current facts. Correct endorsed final answers per 256 assigned questions.');
 chart(s,['Original study','Depth challenge'],[{name:'ReasonKE_plus',values:[o('R-RAG').counts.correct,d('R-RAG').counts.correct]},{name:'Qwen2.5-7B',values:[o('G-RAG').counts.correct,d('G-RAG').counts.correct]}]);
 text(s,'Full-memory original',873,227,338,45,29,C.ink,true);
 text(s,'ReasonKE_plus: 231/256\nQwen: 241/256',873,293,338,103,28,C.ink);
 text(s,'25 wrong ReasonKE_plus\nfinals matched other\nsupplied fact values.',873,444,338,123,27,C.coral,true);
 foot(s,'Retrieved Qwen abstentions: 15 original, 8 depth. Small worlds and controlled wording bound the perfect scores.');

 s=slide(11,'Depth challenge, warm pipeline timing. Same question population and inference budgets.');
 chart(s,['ReasonKE_plus\n+ iterative retrieval','Qwen2.5-7B\n+ iterative retrieval'],[{name:'Median latency',values:[d('R-RAG').generated_latency.p50_seconds,d('G-RAG').generated_latency.p50_seconds],points:[{idx:0,fill:C.teal},{idx:1,fill:C.coral}]}],{max:22,step:5,valueTitle:'Seconds',format:'0.00',legend:false});
 text(s,'8 more correct answers',873,232,338,92,36,C.teal,true);
 text(s,'256 versus 248\nof 256 assigned',873,337,338,95,29,C.ink);
 text(s,'p95: 19.28 s / 1.38 s\nReasonKE_plus / Qwen',873,482,338,100,26,C.muted);
 foot(s,'n=256 per condition, including abstentions. Selection + local IPC + generation. Excludes cold load/browser network.');

 s=slide(12,'A working maintenance platform and a reproducible experimental baseline.');
 const contributions=[['Approved memory','Canonical identity, source, approval, versions and atomic rollback'],['Context handling','Bounded retrieval, connected expansion and revision-aware history'],['Evidence and evaluation','Fixed protocols, recoverable ledgers, scoring and failure diagnosis'],['Usable delivery','Chat, source/history views, results explorer, container and notebooks']];
 contributions.forEach((r,i)=>{text(s,r[0],66,219+i*88,335,66,29,C.teal,true);text(s,r[1],427,219+i*88,784,66,28,C.ink);});
 foot(s,'Original retrieved ReasonKE_plus retained 124/124 baseline-correct locality answers.');

 s=slide(13,'A focused, matched adaptation study.',true);
 const future=['Instruction baseline','SFT','SFT + PPO'].map((v,i)=>box(s,v,64+i*396,239,360,108,{accent:i===2,size:33}));
 link(s,future[0],future[1]);link(s,future[1],future[2]);
 text(s,'Reviewed edit data and a profiled pilot',66,399,1145,55,31,C.white,true);
 text(s,'Unseen entities and wording. Larger memories and deeper questions.',66,480,1145,65,29,'#D9E8E2');
 text(s,'Same retrieval and budgets. Accuracy, locality, coverage, latency and resources.',66,554,1145,58,27,'#D9E8E2');
 foot(s,'Planned research question: does adaptation improve transfer and robustness?',true);

 let elapsed=0;
 for(let i=0;i<p.slides.items.length;i++){
  const row=content.slides[i];const sources=row.sources.map(v=>v.startsWith('https:')?v:'https://github.com/ShivamRaghav7/verity/blob/codex/reasonke-integration/'+v);
  const stamp=n=>`${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;
  p.slides.items[i].speakerNotes.text=`Member ${row.speaker}. ${stamp(elapsed)}–${stamp(elapsed+row.seconds)} (${row.seconds} seconds).\n\n${row.speech}\n\nVisual cue\n${row.cue}\n\nRehearsal detail\n${row.detail}\n\nEvidence\n${sources.join('\n')}`;
  elapsed+=row.seconds;
 }
 if(p.slides.items.length!==13||elapsed!==600)throw Error('Slide count/timing mismatch');
 return {slides:13,seconds:600,chartSlides:CHART_SLIDES,tableSlides:TABLE_SLIDES,newModelCalls:0};
}
