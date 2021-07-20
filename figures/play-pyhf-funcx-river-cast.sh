#!/bin/bash

export cast_file=pyhf-funcx-river.cast
if [ ! -f "${cast_file}" ]; then
  curl -sL https://asciinema.org/a/425477.cast?dl=1 -o "${cast_file}"
fi

asciinema play "${cast_file}"

unset cast_file
