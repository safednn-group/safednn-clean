# SafeDNN Clean

Find incorrect boxes in object detection datasets.

## Requirements

See docstring of [safednn-clean.py](safednn-clean.py).

## Usage

Run `python3 safednn-clean.py --help` for detailed information.

Example:

```sh
python3 safednn-clean.py \
	-o quality.json \
	data/instances_val2017.json \
	data/instances_val2017-fasterrcnn.json
```

## Cite

[Combating noisy labels in object detection datasets][1]
(arXiv:2211.13993 [cs.CV])

```
@misc{chachuła2022combating,
      title={Combating noisy labels in object detection datasets}, 
      author={Krystian Chachuła and Adam Popowicz and Jakub Łyskawa and Bartłomiej Olber and Piotr Frątczak and Krystian Radlak},
      year={2022},
      eprint={2211.13993},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[1]: https://arxiv.org/abs/2211.13993

## License

[GNU GENERAL PUBLIC LICENSE Version 3](LICENSE)
