Autonomy Workload Model — source
--------------------------------
Regenerates every figure in the Branes.ai autonomy-compute document series.

  derive.py    Forward derivation of each stage unit cost, itemized op and byte
               counts at a stated configuration. Nothing here is back-solved.
  pipeline.py  Cost, precision-class, service-time and occupancy model.
  profiles.py  The eighteen mission profiles as explicit sensor suites and rates.
  book.py      Emits BranesAI-Workload-Data-Annex.xlsx from the above.

  python -m pip install openpyxl
  python book.py

Changing one rate in profiles.py and re-running regenerates the workbook, which
is the property that makes a reviewer's disagreement cheap to test.

Compute-graph layer
-------------------
  graphmodel.py  Twelve operators and the typed, sized arcs between them, computed
                 from the same mission configurations as the workload model.
  draw.py        Renderer for the full single-page graph.
  draw2.py       Renderer for the two canonical halves used in the document.
  gengraphs.py   Renders the mission variants.

  python -m pip install matplotlib pillow
  python draw2.py && python gengraphs.py
