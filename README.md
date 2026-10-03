# farmer-palantir

**Open data intelligence for Swedish farms, and the robots that act on it.**
Pick a field on a map of Sweden and farmer-palantir builds one picture of it
from open data: elevation, slope, aerial imagery, sun and SMHI weather. The
same field data then becomes a MuJoCo driving simulator: a scripted expert
drives crop rows in it, NVIDIA GR00T N1.7 is fine-tuned on those drives, and
the policy is compressed (fewer denoising steps, FP8/NVFP4, pruned DiT blocks)
with every candidate scored in the simulator of that exact field, so the
result can run on a small computer in the tractor cab. Free and open source
under Apache-2.0.

New here? Read [explanation.txt](explanation.txt) for the plain-language
version, and [docs/robot-mvp.md](docs/robot-mvp.md) for the technical design
and module contracts.

## Quickstart

### Dry run on a laptop (no GPU)

```bash
pip install -e '.[dev]'
MUJOCO_GL=osmesa python -m pytest -q     # test_gr00t_compat skips without an Isaac-GR00T checkout

mkdir -p demo/job
python - <<'EOF'
import json
spec = json.load(open("examples/farm-uppsala.robotspec.json"))
spec["site"]["source"] = "synthetic"     # no network needed
spec.update(n_demos=5, eval_episodes=5)
json.dump(spec, open("demo/job/robotspec.json", "w"))
EOF
echo '{"img_size": 96, "clip_width": 320, "clip_height": 180, "video_seconds": 20}' > demo/job/config.json
FARMPAL_DRY_RUN=1 MUJOCO_GL=osmesa FARMPAL_SITES=demo/sites \
  python robot_pipeline.py --job demo/job
```

On a laptop CPU this takes a quarter of an hour or so; most of it is
rendering the video clips with OSMesa. Stages already done are skipped on a
rerun; `--from <stage>` reruns from a stage and `--only <stage>` runs one.
Set `"source": "auto"` to fetch real data (Lantmäteriet when credentials are
set, else Copernicus, else synthetic).

### Real run on the B200

```bash
NVME=/mnt/nvme HF_TOKEN=hf_... bash scripts/setup_robot_vm.sh   # once; safe to re-run
source /mnt/nvme/robot.env
python robot_pipeline.py --job /mnt/nvme/jobs/uppsala   # after copying a robotspec.json there
```

`setup_robot_vm.sh` mounts the blank local NVMe, installs Isaac-GR00T and its
venv (the finetune, compress and simeval stages run there), creates a venv of
its own for everything else (`$NVME/venv-farmpal`, this repo installed
editable), downloads the weights and runs the GR00T compatibility tests. The
GR00T backbone (`nvidia/Cosmos-Reason2-2B`) is gated on Hugging Face: request
access with the account whose `HF_TOKEN` you use.

## The map page

```bash
source /mnt/nvme/robot.env
FARMPAL_TOKEN=$(openssl rand -hex 24) bash scripts/serve_api.sh
ssh -L 8700:127.0.0.1:8700 <vm>      # from your laptop
```

Open <http://localhost:8700/map>, paste the token, click a field, choose the
square size (100-1000 m) and press "Build site". The API (`api/server.py`)
binds to localhost only; every route except `/health` and `/map` needs
`Authorization: Bearer $FARMPAL_TOKEN`.

| Route | What |
|---|---|
| `GET /health` | ok, dry run, GPU name |
| `GET /map` | the map picker page |
| `POST /sites`, `GET /sites`, `GET /sites/{id}`, `GET /sites/{id}/preview` | build, list and show sites |
| `POST /robot-jobs`, `GET /robot-jobs`, `GET /robot-jobs/{id}` | start (RobotSpec body, optional `config`), list, state |
| `POST /robot-jobs/{id}/stop`, `POST /robot-jobs/{id}/resume` | stop (frees the GPU), resume (`{"from": stage}`) |
| `GET /robot-jobs/{id}/events` | progress as server-sent events |
| `GET /robot-jobs/{id}/eval`, `/graph`, `/video` | outputs |

One job runs at a time because it holds the GPU; a second start gets 409 with
the running job's id. Jobs started by hand over SSH are seen and can be
stopped too.

## Outputs

In `<job>/out/`:

- `training.mp4`: the field, the expert, the fine-tuned model and each
  compressed candidate driving, with title cards.
- `footprint_vs_performance.png`: model size against success rate in the sim.
- `eval.json`: per candidate size, parameters, bits per weight, latency,
  success rate, cross-track error, progress and its clip, all on the same
  seeded episodes, plus the expert's scores.
- `clips/`: one MP4 per candidate and one of the expert.

`<job>/data/lerobot/` holds the demos as a LeRobot v2 dataset with a GR00T
`NEW_EMBODIMENT` modality config. Logs are in `<job>/logs/<stage>.log`.

## Data sources and licences

| Source | Used for | Licence and access |
|---|---|---|
| Lantmäteriet Markhöjdmodell (1 m DEM) and Ortofoto | elevation, aerial photo | CC BY 4.0, credit "© Lantmäteriet". Needs a free Geotorget account (geotorget.lantmateriet.se): order the free products, then set `LANTMATERIET_USER` and `LANTMATERIET_PASSWORD` |
| SMHI open data (point forecast, metobs daily precipitation) | weather in the field summary and the sim | open data, CC BY 4.0, no account |
| Copernicus DEM GLO-30 (public AWS bucket) | elevation fallback without a Lantmäteriet account | free and open Copernicus data, no account |
| EOX Sentinel-2 cloudless, 2024 layer | imagery fallback | CC BY-NC-SA 4.0: **non-commercial only**, used only as a fallback |
| Synthetic | tests and dry runs | made up, not real data |

GR00T N1.7 weights are under the NVIDIA Open Model License (commercial use
allowed, keep the attribution). Our code is Apache-2.0, see [LICENSE](LICENSE).

## Configuration

Environment variables are read as `FARMPAL_<NAME>`, falling back to
`LOBBOT_<NAME>` (this code started in the LobBot repo): `DRY_RUN`, `SITES`,
`JOBS`, `TOKEN`, `CACHE`, `WEATHER`, `GR00T_REPO`, `GR00T_PY`, `GR00T_MODEL`,
`MODELS`. The table in [docs/robot-mvp.md](docs/robot-mvp.md) has defaults.
Per-job overrides (image size, row length, clip size, seeds, candidate list)
go in `<job>/config.json`; see `robot/stages.py:RobotConfig`.

## Layout

```
farmsim/           map pick to site (sources/, geo, weather), MuJoCo sim, expert, video, LeRobot export
robot/             GR00T adapter, fine-tune, compression candidates, footprint, sim eval, report
common/            RobotSpec, progress protocol, job directory layout, env knobs
api/server.py      HTTP API and the map page
robot_pipeline.py  runs the stages: site, demos, finetune, compress, simeval, report
scripts/           setup_robot_vm.sh (B200 setup), serve_api.sh
examples/          farm-uppsala.robotspec.json
```

## Status and honest limits

- Built and tested without a GPU: map picker, site building, simulator,
  expert, video, LeRobot export, compression maths, the API and the full
  pipeline in dry run.
- **Not yet run for real**: GR00T fine-tuning and compression of the real
  model need the B200.
- **Dry-run numbers are not results.** In dry run the "models" are the expert
  with noise and lag scaled to each candidate's size. The video, the graph and
  `eval.json` are labelled DRY RUN. Only show numbers from a real run.
- The simulator uses simple vehicle physics, not wheel and soil contact.
- GR00T was pre-trained on robot arms and humanoids, not tractors; it learns
  driving only from our demos.
- The model would only plan path and speed. Steering control, ISOBUS
  implements and the safety system (ISO 18497) are separate and not in this
  repo.
