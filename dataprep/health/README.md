# Health — MedMosaic long-form

```bash
dataprep/run_health.sh audio   --src /raw/medmosaic --out /derived/medmosaic
dataprep/run_health.sh queries --src /raw/medmosaic --out /derived/medmosaic
```

Prepared queries are published at [cjli/harp](https://huggingface.co/cjli/harp), so the `queries` stage is only needed to change the benchmark.

## Getting the corpus

[MedMosaic](https://huggingface.co/datasets/icml-anon-submission/medmosaic-dataset), CC BY 4.0. The `audio` stage downloads it, fetching only `long_form/`; override the repo with `$MEDMOSAIC_REPO`.

## audio

Nothing to build — `long_form/` is used as-is, with no merging or cropping.

## queries

| Script | Reads | Writes |
|---|---|---|
| `generate_queries.py` | `res.md`, `long_form/Long_Form_metadata.json` | `queries.jsonl` |

`res.md` holds the manual respiratory-event annotations. Each of the 106 episodes yields a `reported` and an `exhibited` query — 212 in total.

## filter_coughs.py

Outside both stages. It ranks [Coswara](https://github.com/iiscleap/Coswara-Data) cough recordings by SNR and clipping into `cough_quality.csv`, which nothing downstream currently reads. Keep it for provenance of how the cough set was chosen; run it by hand to redo that selection, with Coswara unpacked at `<src>/Coswara-Data/`.

## Citation

```bibtex
@article{rajgarhia2026medmosaic,
  title   = {{MedMosaic}: A Challenging Large Scale Benchmark of Diverse Medical Audio},
  author  = {Rajgarhia, Harshit and Ojha, Shuubham and Shaik, Asif and
             Pothanapalli, Akhil and Lokesh, Rachuri and Mukherji, Abhishek and
             Desikan, Prasanna},
  journal = {arXiv preprint arXiv:2605.00969},
  year    = {2026}
}
```
