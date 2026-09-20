#!/usr/bin/env bash
# Netlify build: assemble the static bundle under web/.
#
# The Python sources are copied rather than duplicated in the repo so the
# browser runs the SAME files the tests run -- no forked copy to drift.
set -euo pipefail

mkdir -p web/py web/comms

for f in amr_msgs.py comms.py coordination.py features.py learned.py \
         perception.py planner.py sim2d.py sim_manager.py warehouse_map.py; do
  cp "$f" web/py/
done

# Same href as the tornado route (/comms/) so the button works in both builds.
cp communication_protocols/amr_simulation.html web/comms/index.html

echo "bundle ready: $(ls web/py | wc -l | tr -d ' ') python modules + comms page"
