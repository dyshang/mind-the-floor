# Example label files

Two synthetic files, so every command in this README and in the paper's
Appendix D runs as written. Neither contains benchmark data.

- `labels.json` — 500 answers from a 10-option multiple-choice split, of which
  only 8 options are realized. That is the case `--num-classes` exists for: the
  modal-option floor is computed over what the split contains, while the
  macro-F1 and balanced-accuracy floors are `1/K` over the declared set.
- `answers.json` — 500 open-ended short answers. A few small integers carry
  real mass and the rest is a long tail, so the modal-answer floor is low
  (`0.052`) while the water-filling `pass@k` floor is much higher.
