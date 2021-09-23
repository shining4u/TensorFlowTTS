# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code based on https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
"""Japanese related helpers."""

_pad = "pad"
_eos = "eos"
_punctuation = "?!！？。.、,-"
_special = ""

_hiragana = [chr(_) for _ in range(0x3041, 0x3096)]
_katakana = [chr(_) for _ in range(0x30A0, 0x30FD)]

# _letters = _hiragana + _katakana
_letters = _katakana

# TODO: number, alphabet

symbols = [_pad] + list(_special) + list(_punctuation) + _letters + [_eos]

_symbol_to_id = {c: i for i, c in enumerate(symbols)}
_id_to_symbol = {i: c for i, c in enumerate(symbols)}
