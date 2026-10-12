"""Bounded synthetic checks of the portable human-review page's JavaScript."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from scripts.prepare_data import content_hash
from scripts.review_dataset import HTML_TEMPLATE


NODE = shutil.which("node")
SCRIPT = re.search(r"</script><script>(.*?)</script>", HTML_TEMPLATE, re.S).group(1)
HARNESS = r'''
const vm=require("node:vm"),assert=require("node:assert/strict");
const request=JSON.parse(require("node:fs").readFileSync(0,"utf8"));
const elements=new Map(),downloads=[],blobs=[];
class Element{
 constructor(tag="div"){this.tag=tag;this.children=[];this.listeners={};this.value="";this.disabled=false;this.textContent="";this.files=[]}
 append(...items){for(const item of items){this.children.push(item);if(item.id)elements.set(item.id,item)}}
 addEventListener(name,callback){this.listeners[name]=callback}
 click(){if(this.tag==="a"){downloads.push({filename:this.download,text:blobs[blobs.length-1].parts.join("")});return}if(!this.disabled&&this.listeners.click)this.listeners.click({target:this})}
 focus(){this.focused=true}scrollIntoView(){this.visited=true}
}
for(const id of ["review-data","reviewer","summary","status","cards","all-rows","progress-counts","progress","load-decisions","next-pending","export"]){const element=new Element();element.id=id;elements.set(id,element)}
elements.get("review-data").textContent=JSON.stringify(request.payload);
elements.get("all-rows").append(new Element("option"));
const sandbox={document:{getElementById:id=>elements.get(id),createElement:tag=>new Element(tag)},
 window:{confirm:()=>{sandbox.confirmations++;return sandbox.allowReplacement}},confirmations:0,allowReplacement:false,
 Blob:class{constructor(parts){this.parts=parts;blobs.push(this)}},URL:{createObjectURL:()=>"synthetic:blob",revokeObjectURL:()=>{}},TextEncoder,console,Date};
vm.createContext(sandbox);vm.runInContext(request.script,sandbox,{timeout:1000});
const evaluate=source=>vm.runInContext(source,sandbox,{timeout:1000});
const invokeLoad=exported=>evaluate("loadDecisionText("+JSON.stringify(typeof exported==="string"?exported:JSON.stringify(exported))+")");
const snapshot=()=>JSON.parse(evaluate("JSON.stringify(decisions)"));
const rowControl=id=>evaluate("controls.get("+JSON.stringify(id)+")");
const incoming=rows=>({dataset_id:request.payload.dataset_id,origin_manifest_sha256:request.payload.manifest_sha256,rows});
new Function("assert","elements","downloads","evaluate","invokeLoad","snapshot","rowControl","incoming","request","sandbox",request.case)(assert,elements,downloads,evaluate,invokeLoad,snapshot,rowControl,incoming,request,sandbox);
console.log(JSON.stringify({passed:true}));
'''


class ReviewPageStaticTests(unittest.TestCase):
    def test_controls_are_local_and_do_not_create_decisions_automatically(self):
        for control in ("load-decisions", "progress-counts", "all-rows", "next-pending"):
            self.assertIn('id="' + control + '"', HTML_TEMPLATE)
        for forbidden in ("fetch(", "XMLHttpRequest", "localStorage", "sessionStorage", "innerHTML"):
            self.assertNotIn(forbidden, SCRIPT)
        self.assertIn('Object.create(null)', SCRIPT)
        self.assertIn('["","Pending"]', SCRIPT)


@unittest.skipUnless(NODE, "Node.js is needed for portable review-page JavaScript fixtures.")
class ReviewPageJavascriptTests(unittest.TestCase):
    def setUp(self):
        rows = [{"id": f"synthetic-row-{index}", "question": "Synthetic review-page fixture only."}
                for index in range(25)]
        self.payload = {"dataset_id": "synthetic-review-page", "manifest_sha256": "a" * 64,
                        "task": "scope", "split": "validation",
                        "rows": [{"row": row, "row_sha256": content_hash(row), "integrity_flags": []}
                                 for row in rows]}
        self.decision = {"decision": "accept", "reviewer": "Synthetic fixture reviewer only",
                         "reviewed_at": "2026-10-11T16:00:00+05:30",
                         "row_sha256": self.payload["rows"][0]["row_sha256"]}

    def run_case(self, source, **extra):
        request = {"script": SCRIPT, "payload": self.payload, "case": source,
                   "decision": self.decision, **extra}
        result = subprocess.run([NODE, "-e", HARNESS], input=json.dumps(request), capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])

    def test_script_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.js"
            path.write_text(SCRIPT, encoding="utf-8")
            result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_loaded_records_keep_identity_timestamp_controls_and_pending_counts(self):
        self.run_case('''
assert.deepEqual(snapshot(),{});
assert.equal(invokeLoad(incoming({"synthetic-row-0":request.decision})),1);
assert.deepEqual(snapshot()["synthetic-row-0"],request.decision);
assert.equal(rowControl("synthetic-row-0").select.value,"accept");
assert.equal(rowControl("synthetic-row-0").save.disabled,true);
assert.ok(rowControl("synthetic-row-0").info.textContent.includes(request.decision.reviewed_at));
assert.ok(elements.get("progress-counts").textContent.includes("1 accepted · 0 rejected · 24 pending"));
assert.equal(invokeLoad(incoming({"synthetic-row-0":Object.fromEntries(Object.entries(request.decision).reverse())})),0);
assert.deepEqual(snapshot()["synthetic-row-0"],request.decision);
''')

    def test_all_loaded_records_are_validated_before_any_state_change(self):
        self.run_case('''
const second={...request.decision,row_sha256:request.payload.rows[1].row_sha256,reviewer:""};
assert.throws(()=>invokeLoad(incoming({"synthetic-row-0":request.decision,"synthetic-row-1":second})),/reviewer/);
assert.deepEqual(snapshot(),{});
assert.equal(elements.get("progress").value,0);
''')

    def test_malformed_identity_hash_name_date_decision_and_reject_reason_are_refused(self):
        self.run_case('''
for(const change of [{decision:"pending"},{reviewer:" "},{row_sha256:"b".repeat(64)},
 {reviewed_at:"2026-10-11T16:00:00"},{reviewed_at:"2026-02-30T16:00:00Z"},
 {reviewed_at:"2026-10-11T25:00:00Z"},{reviewed_at:"2026-10-11T16:00:00+24:00"},
 {decision:"reject"},{decision:"reject",reason:" "},{reason:1},{unknown_field:"extra"},
 {reviewer_type:"automated_development"}]){
 assert.throws(()=>invokeLoad(incoming({"synthetic-row-0":{...request.decision,...change}})));
 assert.deepEqual(snapshot(),{});
}
for(const change of [{dataset_id:"another-dataset"},{origin_manifest_sha256:"b".repeat(64)},
 {origin_manifest_sha256:undefined},{rows:[]},{rows:{}},{rows:{"unknown-row":request.decision}},
 {review_policy:"automated_development"},{human_reviewed_data:false}])
 assert.throws(()=>invokeLoad({...incoming({"synthetic-row-0":request.decision}),...change}));
assert.throws(()=>invokeLoad(" ".repeat(2*1024*1024+1)),/limit/);
''')

    def test_duplicate_keys_and_conflicting_records_are_refused_atomically(self):
        self.run_case('''
const original=incoming({"synthetic-row-0":request.decision});invokeLoad(original);
const changed={...request.decision,decision:"reject",reason:"Synthetic conflict only."};
const fresh={...request.decision,row_sha256:request.payload.rows[1].row_sha256};
assert.throws(()=>invokeLoad(incoming({"synthetic-row-1":fresh,"synthetic-row-0":changed})),/Conflicting/);
assert.deepEqual(snapshot(),original.rows);
const duplicate=JSON.stringify(original).replace('"rows":{','"rows":{"synthetic-row-0":'+JSON.stringify(request.decision)+',');
assert.throws(()=>invokeLoad(duplicate),/Duplicate JSON key/);
assert.throws(()=>invokeLoad('{"dataset_id":"x","dataset_id":"y"}'),/Duplicate JSON key/);
assert.deepEqual(snapshot(),original.rows);
''')

    def test_navigation_covers_all_rows_and_next_pending_wraps_without_deciding(self):
        self.run_case('''
assert.equal(elements.get("all-rows").children.length,26);
invokeLoad(incoming({"synthetic-row-0":request.decision}));
elements.get("next-pending").click();assert.equal(elements.get("all-rows").value,"1");
evaluate("visitRow(24)");assert.equal(elements.get("row-24").visited,true);
elements.get("next-pending").click();assert.equal(elements.get("all-rows").value,"1");
assert.equal(Object.keys(snapshot()).length,1);
''')

    def test_explicit_decisions_and_export_retain_rows_and_unique_task_split_filename(self):
        self.run_case('''
elements.get("export").click();assert.equal(downloads.length,0);
const control=rowControl("synthetic-row-0");control.select.value="reject";
elements.get("reviewer").value="Synthetic new reviewer only";
control.save.click();assert.deepEqual(snapshot(),{});
control.reason.value="Synthetic reason only.";control.save.click();
const recorded=snapshot()["synthetic-row-0"];assert.equal(recorded.decision,"reject");
control.save.click();assert.equal(snapshot()["synthetic-row-0"].reviewed_at,recorded.reviewed_at);
control.select.value="accept";control.save.click();assert.equal(sandbox.confirmations,1);
assert.equal(snapshot()["synthetic-row-0"].decision,"reject");
elements.get("export").click();assert.equal(downloads.length,1);
assert.equal(downloads[0].filename,"review-decisions-synthetic-review-page-scope-validation.json");
const exported=JSON.parse(downloads[0].text);assert.deepEqual(exported.rows,snapshot());
assert.equal(exported.origin_manifest_sha256,request.payload.manifest_sha256);
assert.equal(exported.review_policy,"human");assert.equal(exported.human_reviewed_data,true);
assert.equal(exported.rows["synthetic-row-0"].reviewer_type,"human");
''')


if __name__ == "__main__":
    unittest.main()
