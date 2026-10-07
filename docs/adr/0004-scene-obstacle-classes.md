# Ranking and collision use scene-specific obstacles

Driving benchmarks score a small set of traffic participants, not every detector class. This project does the same. Road scenes rank and warn on people and vehicles. Indoor scenes rank and warn on people and large things. Stuff regions and small props are out. People and cars remain the only size priors.

One shared list of every COCO thing would make the nearest obstacle a cup or a remote. A driving-only list would drop the chairs and tables that matter indoors.
