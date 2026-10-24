# Published reference results and our measurements

The authors' [VLKEB paper, Table 2](https://arxiv.org/html/2403.07350v3#S4.T2)
reports the following **upstream BLIP2-OPT MEND results**, in percent:

| Reliability | Text generality | Image generality | Text locality | Image locality | Portability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 98.52 | 98.42 | 98.47 | 99.34 | 89.05 | 28.80 |

These are the authors' benchmark results, **not results measured by our project**.
Their paper evaluates single edits separately and also studies sequential edits.
The paper's dataset contains 5,000 training and 3,174 test editing cases. These
figures motivate a research baseline; they do not guarantee success on arbitrary
images, corrections or unrestricted generated answers.

The [released scoring implementation](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/trainer/losses.py)
scores target-token predictions with the full target included in model input.
Our generated-answer evaluation instead starts from the question and image and
scores the complete bounded continuation. A token score of 100% under teacher
forcing does not imply a correct, terminated generated answer. Locality measures
answer preservation, while our locality-category accuracy additionally checks
the factual target; those measures also must not be conflated.

Our released-editor adapter changes loading allocation, provides bounded greedy
generation, and explicitly records contextual target alignment. The FT baseline
also changes precision handling and supervises answer termination. We therefore
describe this as reuse of the published research core with documented
adaptations, rather than a verified reproduction of the authors' full results.

Use `scripts/render_vlkeb_results.py` to render saved evaluation, smoke or
diagnostic JSON into PNG/PDF figures and CSV/JSON data. Each output records source
file hashes. Missing evidence is an error; the script does not fill in expected
performance. Synthetic renderer checks use `--test-fixture` and visibly label
every figure. Published numbers remain in this attributed reference document
and are never inserted into our experiment outputs or presented as improvements
we measured.
