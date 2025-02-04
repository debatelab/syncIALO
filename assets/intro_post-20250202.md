<div align=center>

# syncIAL🍏

<img src="./syncialo_tree.png" width="50%">

### A Multi-Purpose Synthetic Debate and Argument Mapping Corpus

</div>


## tl;dr

* 🛢️ [Dataset at HF Hub](https://huggingface.co/datasets/DebateLabKIT/syncialo-raw)  
* 👩‍💻 [Python Code Repo](https://github.com/debatelab/syncIALO)  
* 🏋️‍♀️ [Distilled ML Dataset](https://huggingface.co/datasets/DebateLabKIT/deep-argmap-conversations)

## What exactly is syncIALO?

syncIALO is a collection of synthetic [argument mapping](https://en.wikipedia.org/wiki/Argument_map) datasets. Its first and primary corpus (uninspiringly called `synthetic_corpus-001`) contains 

* **>600k claims (aka arguments)**, which are organized in
* **>1000 argument maps**.

syncIALO argument maps are directed graphs: nodes represent claims and labeled edges indicate that one claim supports or attacks another one.

These argument maps can be easily loaded and processed with [networkx](https://networkx.org/documentation/stable/index.html).

```python
from huggingface_hub import hf_hub_download
import json
import networkx as nx
from pathlib import Path

path = Path(hf_hub_download(
    repo_id="DebateLabKIT/syncialo-raw",
    filename="data/synthetic_corpus-001/eval/debate-eval-0001/node_link_data-debate-eval-0001.json"))
argmap = nx.node_link_graph(json.loads(path.read_text()))

type(argmap)
# >>> networkx.classes.digraph.DiGraph

argmap.number_of_nodes()
# >>> 511

argmap.number_of_edges()
# >>> 510

next(iter(argmap.nodes.data()))[1]
# >>> {'claim': 'Governments should provide substantial financial
# >>> incentives to families with children to counteract declining
# >>> population growth and mitigate the long-term consequences on
# >>> societal stability and progress.',
# >>>  'label': 'Pay to Populate'}

```

Let me show you a randomly sampled subgraph from a syncIALO debate in the train split, rendered as, and with [Argdown](https://argdown.org):

<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/@argdown/web-components/dist/argdown-map.css"><script src="https://cdn.jsdelivr.net/npm/@webcomponents/webcomponentsjs/webcomponents-bundle.js" type="module"></script><script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@argdown/web-components/dist/argdown-map.js"></script><figure  role="group" class="argdown-figure"><argdown-map      initial-view="map"><div slot="source" class=""><pre class="language-argdown"><code class="language-argdown"><span class="hljs-statement-title">[Learning Over Leisure]:</span> Schools should restrict students&#x27; access to fan fiction and social media to protect the integrity of education. 
<span class="hljs-attack">    &lt;-</span> <span class="hljs-argument-title">&lt;Restriction Infringes on Freedom of Expression&gt;:</span> Restricting access to fan fiction and social media unconstitutionally limits students&#x27; right to freedom of expression and stifles their creativity.
<span class="hljs-support">        &lt;+</span> <span class="hljs-argument-title">&lt;Lifelong Learning&gt;:</span> By exercising their freedom of expression, students develop essential skills in critical thinking, problem-solving, and effective communication, preparing them for success in their future careers and personal lives.
<span class="hljs-attack">        &lt;-</span> <span class="hljs-argument-title">&lt;Echo Chamber Effect&gt;:</span> Exercising freedom of expression in an unstructured environment can create an echo chamber where students only communicate with like-minded individuals, failing to develop the skills to engage with diverse perspectives and opposing views.
<span class="hljs-attack">            &lt;-</span> <span class="hljs-argument-title">&lt;Silent Observer&gt;:</span> Developing skills to engage with diverse perspectives and opposing views is not essential for effective communication in situations where listening and observing, rather than actively engaging, is the most effective strategy.
<span class="hljs-attack">        &lt;-</span> <span class="hljs-argument-title">&lt;Fan Fiction Distortion&gt;:</span> Fan fiction and social media often distort students&#x27; creativity by promoting unoriginal and copyrighted content, rather than fostering genuine artistic expression.
<span class="hljs-attack">            &lt;-</span> <span class="hljs-argument-title">&lt;Artistic Evolution&gt;:</span> The value of artistic expression lies in its ability to evoke emotions and spark new ideas, regardless of whether it is original or builds upon existing works, making the distinction between original and unoriginal content irrelevant.
<span class="hljs-support">        &lt;+</span> <span class="hljs-argument-title">&lt;Innovation Incubator&gt;:</span> Unrestricted freedom of expression enables students to develop critical thinking, problem-solving, and communication skills, essential for academic and professional success.
<span class="hljs-support">    &lt;+</span> <span class="hljs-argument-title">&lt;Focus on Fundamentals-1&gt;:</span> Restricting access to fan fiction and social media in schools allows students to prioritize core academic subjects and develop a solid foundation in STEM fields, literature, and critical thinking.
<span class="hljs-support">    &lt;+</span> <span class="hljs-argument-title">&lt;Focus on Fundamentals-2&gt;:</span> By limiting access to non-academic online content, schools can redirect students&#x27; attention to foundational subjects, fostering a stronger understanding of complex concepts and better retention of critical information.
<span class="hljs-support">        &lt;+</span> <span class="hljs-argument-title">&lt;Knowledge Pyramid&gt;:</span> A strong grasp of foundational subjects allows students to recognize relationships between different ideas and concepts, creating a hierarchical structure of knowledge that enhances retention and recall of critical information.</code></pre></div><div slot="map">
<!-- Generated by graphviz version 2.47.0 (20210316.0004)
 -->
<!-- Title: Argument Map Pages: 1 -->
<svg width="720pt" height="348pt"
 viewBox="0.00 0.00 720.00 348.35" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(0.68 0.68) rotate(0) translate(4 512)">
<title>Argument Map</title>
<!-- n0 -->
<g id="node1" class="node">
<title>n0</title>
<path fill="white" stroke="#1b9e77" stroke-width="2" d="M829,-508C829,-508 657,-508 657,-508 651,-508 645,-502 645,-496 645,-496 645,-456 645,-456 645,-450 651,-444 657,-444 657,-444 829,-444 829,-444 835,-444 841,-450 841,-456 841,-456 841,-496 841,-496 841,-502 835,-508 829,-508"/>
<text text-anchor="start" x="693.54" y="-494" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Learning Over Leisure</text>
<text text-anchor="start" x="671.32" y="-477" font-family="arial" font-size="10.00" fill="#000000">Schools should restrict students&#39;</text>
<text text-anchor="start" x="658.81" y="-465" font-family="arial" font-size="10.00" fill="#000000"> access to fan fiction and social media</text>
<text text-anchor="start" x="662.14" y="-453" font-family="arial" font-size="10.00" fill="#000000"> to protect the integrity of education. </text>
</g>
<!-- n1 -->
<g id="node2" class="node">
<title>n1</title>
<path fill="#1b9e77" stroke="black" d="M615,-400C615,-400 443,-400 443,-400 437,-400 431,-394 431,-388 431,-388 431,-316 431,-316 431,-310 437,-304 443,-304 443,-304 615,-304 615,-304 621,-304 627,-310 627,-316 627,-316 627,-388 627,-388 627,-394 621,-400 615,-400"/>
<text text-anchor="start" x="450.93" y="-386" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Restriction Infringes on Freedom of</text>
<text text-anchor="start" x="502.88" y="-376" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000"> Expression</text>
<text text-anchor="start" x="450.37" y="-361" font-family="arial" font-size="10.00" fill="#000000">Restricting access to fan fiction and</text>
<text text-anchor="start" x="447.33" y="-349" font-family="arial" font-size="10.00" fill="#000000"> social media unconstitutionally limits</text>
<text text-anchor="start" x="465.65" y="-337" font-family="arial" font-size="10.00" fill="#000000"> students&#39; right to freedom of</text>
<text text-anchor="start" x="468.43" y="-325" font-family="arial" font-size="10.00" fill="#000000"> expression and stifles their</text>
<text text-anchor="start" x="506.5" y="-313" font-family="arial" font-size="10.00" fill="#000000"> creativity.</text>
</g>
<!-- n1&#45;&gt;n0 -->
<g id="edge7" class="edge">
<title>n1&#45;&gt;n0</title>
<path fill="none" stroke="#ff0000" d="M611.74,-400.17C634.28,-413.02 658.33,-426.73 679.61,-438.86"/>
<polygon fill="#ff0000" stroke="#ff0000" points="678.12,-442.04 688.55,-443.96 681.59,-435.96 678.12,-442.04"/>
</g>
<!-- n2 -->
<g id="node3" class="node">
<title>n2</title>
<path fill="#1b9e77" stroke="black" d="M184,-260C184,-260 12,-260 12,-260 6,-260 0,-254 0,-248 0,-248 0,-160 0,-160 0,-154 6,-148 12,-148 12,-148 184,-148 184,-148 190,-148 196,-154 196,-160 196,-160 196,-248 196,-248 196,-254 190,-260 184,-260"/>
<text text-anchor="start" x="59.65" y="-246" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Lifelong Learning</text>
<text text-anchor="start" x="31.88" y="-229" font-family="arial" font-size="10.00" fill="#000000">By exercising their freedom of</text>
<text text-anchor="start" x="10.75" y="-217" font-family="arial" font-size="10.00" fill="#000000"> expression, students develop essential</text>
<text text-anchor="start" x="44.95" y="-205" font-family="arial" font-size="10.00" fill="#000000"> skills in critical thinking,</text>
<text text-anchor="start" x="30.21" y="-193" font-family="arial" font-size="10.00" fill="#000000"> problem&#45;solving, and effective</text>
<text text-anchor="start" x="19.38" y="-181" font-family="arial" font-size="10.00" fill="#000000"> communication, preparing them for</text>
<text text-anchor="start" x="20.2" y="-169" font-family="arial" font-size="10.00" fill="#000000"> success in their future careers and</text>
<text text-anchor="start" x="64.66" y="-157" font-family="arial" font-size="10.00" fill="#000000"> personal lives.</text>
</g>
<!-- n2&#45;&gt;n1 -->
<g id="edge1" class="edge">
<title>n2&#45;&gt;n1</title>
<path fill="none" stroke="#00ff00" d="M196.19,-256.41C199.14,-257.65 202.08,-258.85 205,-260 275.2,-287.58 356.99,-310.48 420.83,-326.41"/>
<polygon fill="#00ff00" stroke="#00ff00" points="420.37,-329.91 430.92,-328.91 422.05,-323.11 420.37,-329.91"/>
</g>
<!-- n3 -->
<g id="node4" class="node">
<title>n3</title>
<path fill="#1b9e77" stroke="black" d="M401.5,-260C401.5,-260 226.5,-260 226.5,-260 220.5,-260 214.5,-254 214.5,-248 214.5,-248 214.5,-160 214.5,-160 214.5,-154 220.5,-148 226.5,-148 226.5,-148 401.5,-148 401.5,-148 407.5,-148 413.5,-154 413.5,-160 413.5,-160 413.5,-248 413.5,-248 413.5,-254 407.5,-260 401.5,-260"/>
<text text-anchor="start" x="266.99" y="-246" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Echo Chamber Effect</text>
<text text-anchor="start" x="228.37" y="-229" font-family="arial" font-size="10.00" fill="#000000">Exercising freedom of expression in an</text>
<text text-anchor="start" x="224.75" y="-217" font-family="arial" font-size="10.00" fill="#000000"> unstructured environment can create an</text>
<text text-anchor="start" x="235.59" y="-205" font-family="arial" font-size="10.00" fill="#000000"> echo chamber where students only</text>
<text text-anchor="start" x="246.16" y="-193" font-family="arial" font-size="10.00" fill="#000000"> communicate with like&#45;minded</text>
<text text-anchor="start" x="241.15" y="-181" font-family="arial" font-size="10.00" fill="#000000"> individuals, failing to develop the</text>
<text text-anchor="start" x="250.88" y="-169" font-family="arial" font-size="10.00" fill="#000000"> skills to engage with diverse</text>
<text text-anchor="start" x="238.37" y="-157" font-family="arial" font-size="10.00" fill="#000000"> perspectives and opposing views.</text>
</g>
<!-- n3&#45;&gt;n1 -->
<g id="edge3" class="edge">
<title>n3&#45;&gt;n1</title>
<path fill="none" stroke="#ff0000" d="M395.28,-260.2C413.47,-272.55 432.73,-285.63 450.79,-297.89"/>
<polygon fill="#ff0000" stroke="#ff0000" points="449.31,-301.12 459.55,-303.84 453.25,-295.33 449.31,-301.12"/>
</g>
<!-- n4 -->
<g id="node5" class="node">
<title>n4</title>
<path fill="#1b9e77" stroke="black" d="M400,-112C400,-112 228,-112 228,-112 222,-112 216,-106 216,-100 216,-100 216,-12 216,-12 216,-6 222,0 228,0 228,0 400,0 400,0 406,0 412,-6 412,-12 412,-12 412,-100 412,-100 412,-106 406,-112 400,-112"/>
<text text-anchor="start" x="279.55" y="-98" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Silent Observer</text>
<text text-anchor="start" x="242.87" y="-81" font-family="arial" font-size="10.00" fill="#000000">Developing skills to engage with</text>
<text text-anchor="start" x="235.64" y="-69" font-family="arial" font-size="10.00" fill="#000000"> diverse perspectives and opposing</text>
<text text-anchor="start" x="238.43" y="-57" font-family="arial" font-size="10.00" fill="#000000"> views is not essential for effective</text>
<text text-anchor="start" x="236.21" y="-45" font-family="arial" font-size="10.00" fill="#000000"> communication in situations where</text>
<text text-anchor="start" x="234.53" y="-33" font-family="arial" font-size="10.00" fill="#000000"> listening and observing, rather than</text>
<text text-anchor="start" x="247.04" y="-21" font-family="arial" font-size="10.00" fill="#000000"> actively engaging, is the most</text>
<text text-anchor="start" x="273.43" y="-9" font-family="arial" font-size="10.00" fill="#000000"> effective strategy.</text>
</g>
<!-- n4&#45;&gt;n3 -->
<g id="edge2" class="edge">
<title>n4&#45;&gt;n3</title>
<path fill="none" stroke="#ff0000" d="M314,-112.2C314,-120.46 314,-129.05 314,-137.51"/>
<polygon fill="#ff0000" stroke="#ff0000" points="310.5,-137.73 314,-147.73 317.5,-137.73 310.5,-137.73"/>
</g>
<!-- n5 -->
<g id="node6" class="node">
<title>n5</title>
<path fill="#1b9e77" stroke="black" d="M616,-248C616,-248 444,-248 444,-248 438,-248 432,-242 432,-236 432,-236 432,-172 432,-172 432,-166 438,-160 444,-160 444,-160 616,-160 616,-160 622,-160 628,-166 628,-172 628,-172 628,-236 628,-236 628,-242 622,-248 616,-248"/>
<text text-anchor="start" x="482.77" y="-234" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Fan Fiction Distortion</text>
<text text-anchor="start" x="455.54" y="-217" font-family="arial" font-size="10.00" fill="#000000">Fan fiction and social media often</text>
<text text-anchor="start" x="465.83" y="-205" font-family="arial" font-size="10.00" fill="#000000"> distort students&#39; creativity by</text>
<text text-anchor="start" x="446.37" y="-193" font-family="arial" font-size="10.00" fill="#000000"> promoting unoriginal and copyrighted</text>
<text text-anchor="start" x="445.52" y="-181" font-family="arial" font-size="10.00" fill="#000000"> content, rather than fostering genuine</text>
<text text-anchor="start" x="487.22" y="-169" font-family="arial" font-size="10.00" fill="#000000"> artistic expression.</text>
</g>
<!-- n5&#45;&gt;n1 -->
<g id="edge5" class="edge">
<title>n5&#45;&gt;n1</title>
<path fill="none" stroke="#ff0000" d="M529.71,-248.02C529.61,-262.3 529.5,-278.44 529.39,-293.63"/>
<polygon fill="#ff0000" stroke="#ff0000" points="525.89,-293.93 529.32,-303.96 532.89,-293.98 525.89,-293.93"/>
</g>
<!-- n6 -->
<g id="node7" class="node">
<title>n6</title>
<path fill="#1b9e77" stroke="black" d="M616,-112C616,-112 444,-112 444,-112 438,-112 432,-106 432,-100 432,-100 432,-12 432,-12 432,-6 438,0 444,0 444,0 616,0 616,0 622,0 628,-6 628,-12 628,-12 628,-100 628,-100 628,-106 622,-112 616,-112"/>
<text text-anchor="start" x="493.05" y="-98" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Artistic Evolution</text>
<text text-anchor="start" x="452.21" y="-81" font-family="arial" font-size="10.00" fill="#000000">The value of artistic expression lies</text>
<text text-anchor="start" x="452.48" y="-69" font-family="arial" font-size="10.00" fill="#000000"> in its ability to evoke emotions and</text>
<text text-anchor="start" x="441.65" y="-57" font-family="arial" font-size="10.00" fill="#000000"> spark new ideas, regardless of whether</text>
<text text-anchor="start" x="452.49" y="-45" font-family="arial" font-size="10.00" fill="#000000"> it is original or builds upon existing</text>
<text text-anchor="start" x="444.43" y="-33" font-family="arial" font-size="10.00" fill="#000000"> works, making the distinction between</text>
<text text-anchor="start" x="461.93" y="-21" font-family="arial" font-size="10.00" fill="#000000"> original and unoriginal content</text>
<text text-anchor="start" x="506.67" y="-9" font-family="arial" font-size="10.00" fill="#000000"> irrelevant.</text>
</g>
<!-- n6&#45;&gt;n5 -->
<g id="edge4" class="edge">
<title>n6&#45;&gt;n5</title>
<path fill="none" stroke="#ff0000" d="M530,-112.2C530,-124.44 530,-137.39 530,-149.55"/>
<polygon fill="#ff0000" stroke="#ff0000" points="526.5,-149.56 530,-159.56 533.5,-149.56 526.5,-149.56"/>
</g>
<!-- n7 -->
<g id="node8" class="node">
<title>n7</title>
<path fill="#1b9e77" stroke="black" d="M830,-248C830,-248 658,-248 658,-248 652,-248 646,-242 646,-236 646,-236 646,-172 646,-172 646,-166 652,-160 658,-160 658,-160 830,-160 830,-160 836,-160 842,-166 842,-172 842,-172 842,-236 842,-236 842,-242 836,-248 830,-248"/>
<text text-anchor="start" x="698.7" y="-234" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Innovation Incubator</text>
<text text-anchor="start" x="666.21" y="-217" font-family="arial" font-size="10.00" fill="#000000">Unrestricted freedom of expression</text>
<text text-anchor="start" x="664.81" y="-205" font-family="arial" font-size="10.00" fill="#000000"> enables students to develop critical</text>
<text text-anchor="start" x="676.21" y="-193" font-family="arial" font-size="10.00" fill="#000000"> thinking, problem&#45;solving, and</text>
<text text-anchor="start" x="667.05" y="-181" font-family="arial" font-size="10.00" fill="#000000"> communication skills, essential for</text>
<text text-anchor="start" x="662.31" y="-169" font-family="arial" font-size="10.00" fill="#000000"> academic and professional success.</text>
</g>
<!-- n7&#45;&gt;n1 -->
<g id="edge6" class="edge">
<title>n7&#45;&gt;n1</title>
<path fill="none" stroke="#00ff00" d="M680.66,-248.02C657.5,-263.74 631.03,-281.71 606.76,-298.2"/>
<polygon fill="#00ff00" stroke="#00ff00" points="604.58,-295.44 598.28,-303.96 608.51,-301.23 604.58,-295.44"/>
</g>
<!-- n8 -->
<g id="node9" class="node">
<title>n8</title>
<path fill="#1b9e77" stroke="black" d="M829,-402C829,-402 657,-402 657,-402 651,-402 645,-396 645,-390 645,-390 645,-314 645,-314 645,-308 651,-302 657,-302 657,-302 829,-302 829,-302 835,-302 841,-308 841,-314 841,-314 841,-390 841,-390 841,-396 835,-402 829,-402"/>
<text text-anchor="start" x="684.93" y="-388" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Focus on Fundamentals&#45;1</text>
<text text-anchor="start" x="664.37" y="-371" font-family="arial" font-size="10.00" fill="#000000">Restricting access to fan fiction and</text>
<text text-anchor="start" x="674.94" y="-359" font-family="arial" font-size="10.00" fill="#000000"> social media in schools allows</text>
<text text-anchor="start" x="662.99" y="-347" font-family="arial" font-size="10.00" fill="#000000"> students to prioritize core academic</text>
<text text-anchor="start" x="678.81" y="-335" font-family="arial" font-size="10.00" fill="#000000"> subjects and develop a solid</text>
<text text-anchor="start" x="661.04" y="-323" font-family="arial" font-size="10.00" fill="#000000"> foundation in STEM fields, literature,</text>
<text text-anchor="start" x="697.72" y="-311" font-family="arial" font-size="10.00" fill="#000000"> and critical thinking.</text>
</g>
<!-- n8&#45;&gt;n0 -->
<g id="edge8" class="edge">
<title>n8&#45;&gt;n0</title>
<path fill="none" stroke="#00ff00" d="M743,-402.27C743,-412.68 743,-423.55 743,-433.63"/>
<polygon fill="#00ff00" stroke="#00ff00" points="739.5,-433.79 743,-443.79 746.5,-433.79 739.5,-433.79"/>
</g>
<!-- n9 -->
<g id="node10" class="node">
<title>n9</title>
<path fill="#1b9e77" stroke="black" d="M1046.5,-408C1046.5,-408 871.5,-408 871.5,-408 865.5,-408 859.5,-402 859.5,-396 859.5,-396 859.5,-308 859.5,-308 859.5,-302 865.5,-296 871.5,-296 871.5,-296 1046.5,-296 1046.5,-296 1052.5,-296 1058.5,-302 1058.5,-308 1058.5,-308 1058.5,-396 1058.5,-396 1058.5,-402 1052.5,-408 1046.5,-408"/>
<text text-anchor="start" x="901.43" y="-394" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Focus on Fundamentals&#45;2</text>
<text text-anchor="start" x="881.43" y="-377" font-family="arial" font-size="10.00" fill="#000000">By limiting access to non&#45;academic</text>
<text text-anchor="start" x="879.76" y="-365" font-family="arial" font-size="10.00" fill="#000000"> online content, schools can redirect</text>
<text text-anchor="start" x="883.36" y="-353" font-family="arial" font-size="10.00" fill="#000000"> students&#39; attention to foundational</text>
<text text-anchor="start" x="893.93" y="-341" font-family="arial" font-size="10.00" fill="#000000"> subjects, fostering a stronger</text>
<text text-anchor="start" x="869.75" y="-329" font-family="arial" font-size="10.00" fill="#000000"> understanding of complex concepts and</text>
<text text-anchor="start" x="903.38" y="-317" font-family="arial" font-size="10.00" fill="#000000"> better retention of critical</text>
<text text-anchor="start" x="932" y="-305" font-family="arial" font-size="10.00" fill="#000000"> information.</text>
</g>
<!-- n9&#45;&gt;n0 -->
<g id="edge10" class="edge">
<title>n9&#45;&gt;n0</title>
<path fill="none" stroke="#00ff00" d="M861.43,-408.11C842.86,-418.6 823.88,-429.32 806.71,-439.02"/>
<polygon fill="#00ff00" stroke="#00ff00" points="804.89,-436.02 797.91,-443.99 808.33,-442.12 804.89,-436.02"/>
</g>
<!-- n10 -->
<g id="node11" class="node">
<title>n10</title>
<path fill="#1b9e77" stroke="black" d="M1045,-260C1045,-260 873,-260 873,-260 867,-260 861,-254 861,-248 861,-248 861,-160 861,-160 861,-154 867,-148 873,-148 873,-148 1045,-148 1045,-148 1051,-148 1057,-154 1057,-160 1057,-160 1057,-248 1057,-248 1057,-254 1051,-260 1045,-260"/>
<text text-anchor="start" x="914.55" y="-246" font-family="arial" font-weight="bold" font-size="10.00" fill="#000000">Knowledge Pyramid</text>
<text text-anchor="start" x="872.58" y="-229" font-family="arial" font-size="10.00" fill="#000000">A strong grasp of foundational subjects</text>
<text text-anchor="start" x="894.82" y="-217" font-family="arial" font-size="10.00" fill="#000000"> allows students to recognize</text>
<text text-anchor="start" x="876.48" y="-205" font-family="arial" font-size="10.00" fill="#000000"> relationships between different ideas</text>
<text text-anchor="start" x="876.2" y="-193" font-family="arial" font-size="10.00" fill="#000000"> and concepts, creating a hierarchical</text>
<text text-anchor="start" x="874.53" y="-181" font-family="arial" font-size="10.00" fill="#000000"> structure of knowledge that enhances</text>
<text text-anchor="start" x="893.99" y="-169" font-family="arial" font-size="10.00" fill="#000000"> retention and recall of critical</text>
<text text-anchor="start" x="931.5" y="-157" font-family="arial" font-size="10.00" fill="#000000"> information.</text>
</g>
<!-- n10&#45;&gt;n9 -->
<g id="edge9" class="edge">
<title>n10&#45;&gt;n9</title>
<path fill="none" stroke="#00ff00" d="M959,-260.2C959,-268.46 959,-277.05 959,-285.51"/>
<polygon fill="#00ff00" stroke="#00ff00" points="955.5,-285.73 959,-295.73 962.5,-285.73 955.5,-285.73"/>
</g>
</g>
</svg>
</div></argdown-map></figure>



## What can I do with it?

Raw syncIALO is great for "distilling" more specific datatsets.

* You can use syncIALO to build **datasets for pretraining, SFT, DPO or RLVR**.
* You can create **challenging benchmarks** to probe reasoning skills of LLMs.
* You can create **tailored few-shot examples** for generating argument maps with LLMs.
* You can use syncIALO data as **seeds** for multi-agent deliberation and personalization of LLMs.

For any of this, you will have to transform the syncIALO debates and distill more specific tasks.

To start with, you could sample submaps and simply verbalize them as dialogues, which then serve as training texts... But we can do better by exploiting the rich information contained in syncIALO.

This recipe for creating a reasoning task describes a more interesting distillation procedure: 

1. Sample a submap (serves as _answer_).
2. Process or distort submap (serves as *input_args*).
3. Ask the model to create an argument map given *input_args* (serves as _prompt_).

For example:

|Prompt|Answer|
|---|---|
|Here's a list of statements ... Reconstruct these as an argument map!|Argument map|
|Consider these three maps ... Merge them into a single argument map!|Argument map|
|Here's a flawed reconstruction ... Revise and improve!|Argument map|

The [`deep-argmap-conversations`](https://huggingface.co/datasets/DebateLabKIT/deep-argmap-conversations) dataset has been distilled accordingly and illustrates further argument mapping tasks that can be created from raw syncIALO.

Similarly, you can distill DPO data:

|Prompt|Chosen|Rejected|
|---|---|---|
|Here's a list of statements ... Reconstruct these as an argument map!|Argument map|Shuffled argument map|
|...|...|...|

If you instruct the LLM to generate argument maps in parsable format (like yaml, mermaid or Argdown), you have sheer infite possibilities to verify solutions and create RLVR data:

|Prompt|Reward|
|---|---|
|Here's a list of claims ... Reconstruct these as a yaml argument map!|Valid yaml?|
|Here's a list of claims ... Reconstruct these as an argument map with _k_ nodes!|Valid yaml with _k_ nodes?|
|...|...|

(Granted, syncIALO is not be strictly necessary for such RLVR training, which might nonetheless profit from diverse and well-designed syncIALO prompts.) 

Moreover, multiple choice tasks can be easily created like so:

|Prompt|Options|
|---|---|
|Consider this argument map ... What is x (SOME_GRAPH_PROPERTY)? | x=a, x=b ... |
|Here's a list of statements ... Which map adequately captures the argumenation?| a) argument map, b) shuffled map ... |
|...|...|

This is cool for improving CoT / reasoning quality with RL and verifiable rewards, and of course for benchmarking LLMs.

But syncIALO can help during **inference**, too. Suppose you want your model to reconstruct a given text as a –– say: Argdown –– argument map of a certain size. If the model struggles, you can create diverse few-shot examples tailored to the problem at hand _ad nauseam_, and thus guide the model.  

Personas datasets are helpful for increasing the diversity in synthetic datasets, for broad solution space exploration during inference, and for calibrating agentic AI systems. syncIALO can play a similiar role and complement existing personas datasets: For example, one may additionally charactierize a persona through a stance they adopt in a debate, or an argument they have put forth, endorsed or criticized. 

So, syncIALO is really multi-purpose. Let's explore together what you can do with it!



## How did you build it?

We've set up a dynamic pipeline that mimics a comprehensive argument mapping process. An LLM-based agent simulates a critical thinker who seeks, assesses and stores novel arguments.

The argument map is built recursively by adding pros and cons to the leaf nodes until a maximum depth has been reached. The AI agent identifies the premises of a target argument _A_ before conceiving further arguments that either support or attack _A_. It selects candidate arguments in terms of salience and diversity, and checks for duplicates (via semantic similarity) before adding arguments to the argument map.

To increase diversity, we sample topics and motions by randomly choosing tags from a diverse tag cloud. We also let the AI critical thinker adopt a randomly drawn persona whenever it generates a new candidate argument.

The LLM-based agent is powered by different ❤️ open models, depending on the workflow step. We've used `meta-llama/Llama-3.1-405B` for generating and assessing arguments, and a finetuned Llama-3.1-8B model for less demanding generative tasks such as formatting. `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` serves as our multi-purpose classifier and we use `sentence-transformers/all-MiniLM-L6-v2` to generate sentence embeddings.

The pipeline is built on top of ❤️ open source frameworks:

- [LangChain](https://github.com/langchain-ai/langchain)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [networkx](https://github.com/networkx/networkx)
- [prefect](https://github.com/PrefectHQ/prefect)

We're releasing the syncIALO dataset together with the identically named [python package](https://github.com/debatelab/syncIALO), which we have used to build the synthetic dataset.


## What is the broader background?

Philosophically, syncIALO is inspired by the [Rylian](https://plato.stanford.edu/entries/ryle/) idea that epistemic competence is closely tied to argumentative language. One's epistemic competence consists, to a great part, in the ability to produce utterances in accordance with the norms of logical, evidential or scientific reasoning. Argument mapping and critical thinking may help one to excel in this domain. That's why they might provide useful resources for training and probing AI systems. 

The wonderful [kialo.com](https://kialo.com) project can be credited with having solved the problem of designing intuitive yet effective online collaborative debating / argument mapping platforms. It's a pleasure to see [how successful](https://en.wikipedia.org/wiki/Kialo) they are.

The informal argument maps amassed on the Kialo site are a gold mine for NLP researchers, AI engineers, computational sociologists, and Critical Thinking scholars. Yet, the mine is legally barred (for them): Debate data downloaded or scraped from the website may not be used for research or commercial purposes in the absence of explicit permission or license agreement.

That has been a further motivation for creating the syncIALO corpora, which may serve as a drop-in replacements for the Kialo data. (But it's clear that syncIALO is no universal substitute: A cognitive scientists, for example, who studies _empirically_ how humans _actually_ argue might find syncIALO of little help.)


## Who's behind this?

syncIALO has been conceived and built by the DebateLab Team at KIT. You find us at [HuggingFace](https://huggingface.co/DebateLabKIT) and [GitHub](https://github.com/debatelab), or can follow [our blog](https://debatelab.github.io/).

**🤗 Hugging Face has sponsored the syncIALO project through inference time / compute credits. 🙏 We gratefully acknowledge the genereous support. 🫶**


## How can I get involved?

You can help to improve syncIALO and to **overcome its current limitations**, for example by contributing pipelines to

- check the data (argumentative relations, wording, appropriate labeling)
- measure local and global diversity (claim embeddings)
- spot and remove claim duplicates
- build improved versions through argumentative refinement and re-wiring

You might also consider to

- create new corpora (varying llms, topic tags, graph configs)
- translate an existing debate corpus (we already have a pipeline for this)

Yet, most importantly, we invite you to

- **build with syncIALO** and **share your work**. 

Don't hesitate to reach out!

