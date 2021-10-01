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
import re

import suji
import jaconv
import MeCab

_tagger = None

_number_pattern = re.compile(r"([0-9１２３４５６７８９０]+)")
_phone_pattern = re.compile(r"(\d{3,4}(ー|-)\d{3,4}((ー|-)\d{3,4})?)")
_numbering_pattern = re.compile(r"([0-9１２３４５６７８９０]+)(号)")

_pad = "pad"
_eos = "eos"

_punctuations = [
    "、", "。", "?", "!"
]

_specials = [
    "-", "ー",
]

_sym_leads = [
    # normal
    "A", "K", "S", "T", "N", "H", "M", "Y", "R", "W", "NN",
    # daku
    "V", "G", "Z", "D",      "B",
    #
                             "P",
    # small ka, ke
    "k"
]

_sym_basic_vowels = [
    # basic vowels
    "a", "i", "u", "e", "o",
]

_sym_composite_vowels = [
    # ャ
    #      "iya", "uya", "eya",
    "aya", "iya", "uya", "eya", "oya",
    # ュ
    #      "iyu", "uyu", "eyu",
    "ayu", "iyu", "uyu", "eyu", "oyu",
    # ョ
    #      "iyo", "uyo", "eyo",
    "ayo", "iyo", "uyo", "eyo", "oyo",
    # ァ
    # "aa",       "ua",         "oa",
    "aa",  "ia",  "ua",  "ea",  "oa",
    # ィ
    #      "ii",  "ui",  "ei",
    "ai",  "ii",  "ui",  "ei",  "oi",
    # ゥ
    #             "uu",         "ou",
    "au",  "iu",  "uu",  "eu",  "ou",
    # ェ
    #      "ie",  "ue",
    "ae",  "ie",  "ue",  "ee",  "oe",
    # ォ
    #             "uo",         "oo",
    "ao",  "io",  "uo",  "eo",  "oo",
    # ヮ
    # not exists
    "awa", "iwa", "uwa", "ewa", "owa",
]

_sym_vowels = _sym_basic_vowels + _sym_composite_vowels

_sym_tails = [
    "tsu",
]

_wrong_letter = [
    ("Y", "i"), ("Y", "e"),
    ("W", "u"),
    ("Y", "e"), ("Y", "e"), ("Y", "e"),
    ("k", "i"), ("k", "u"), ("k", "o"),
]

_letters = _sym_leads + _sym_vowels + _sym_tails

symbols = [_pad] + _specials + _punctuations + _letters + [_eos]

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_katakana = [
    (a, b)
        for a in _sym_leads
        for b in _sym_basic_vowels
    if (a, b) not in _wrong_letter
    and a != "NN"
] + [("NN",)]

_katakana_composite_vowels = [
    "ャ",      "ュ",       "ョ",
    "ァ", "ィ", "ゥ", "ェ", "ォ",
    "ヮ",
]

_translate_table = {
    "ア": ("A", "a"),
    "イ": ("A", "i"),
    "ウ": ("A", "u"),
    "エ": ("A", "e"),
    "オ": ("A", "o"),

    "カ": ("K", "a"),
    "キ": ("K", "i"),
    "ク": ("K", "u"),
    "ケ": ("K", "e"),
    "コ": ("K", "o"),

    "サ": ("S", "a"),
    "シ": ("S", "i"),
    "ス": ("S", "u"),
    "セ": ("S", "e"),
    "ソ": ("S", "o"),

    "タ": ("T", "a"),
    "チ": ("T", "i"),
    "ツ": ("T", "u"),
    "テ": ("T", "e"),
    "ト": ("T", "o"),

    "ナ": ("N", "a"),
    "ニ": ("N", "i"),
    "ヌ": ("N", "u"),
    "ネ": ("N", "e"),
    "ノ": ("N", "o"),

    "ハ": ("H", "a"),
    "ヒ": ("H", "i"),
    "フ": ("H", "u"),
    "ヘ": ("H", "e"),
    "ホ": ("H", "o"),

    "マ": ("M", "a"),
    "ミ": ("M", "i"),
    "ム": ("M", "u"),
    "メ": ("M", "e"),
    "モ": ("M", "o"),

    "ヤ": ("Y", "a"),
    "ユ": ("Y", "u"),
    "ヨ": ("Y", "o"),

    "ラ": ("R", "a"),
    "リ": ("R", "i"),
    "ル": ("R", "u"),
    "レ": ("R", "e"),
    "ロ": ("R", "o"),

    "ワ": ("W", "a"),
    "ヰ": ("W", "i"),
    #"?"
    "ヱ": ("W", "e"),
    "ヲ": ("W", "o"),


    "ガ": ("G", "a"),
    "ギ": ("G", "i"),
    "グ": ("G", "u"),
    "ゲ": ("G", "e"),
    "ゴ": ("G", "o"),

    "ザ": ("Z", "a"),
    "ジ": ("Z", "i"),
    "ズ": ("Z", "u"),
    "ゼ": ("Z", "e"),
    "ゾ": ("Z", "o"),

    "ダ": ("D", "a"),
    "ヂ": ("D", "i"),
    "ヅ": ("D", "u"),
    "デ": ("D", "e"),
    "ド": ("D", "o"),

    "バ": ("B", "a"),
    "ビ": ("B", "i"),
    "ブ": ("B", "u"),
    "ベ": ("B", "e"),
    "ボ": ("B", "o"),

    "パ": ("P", "a"),
    "ピ": ("P", "i"),
    "プ": ("P", "u"),
    "ペ": ("P", "e"),
    "ポ": ("P", "o"),

    "ャ": ("ya", ),
    "ュ": ("yo", ),
    "ョ": ("yu", ),

    "ァ": ("a", ),
    "ィ": ("i", ),
    "ゥ": ("u", ),
    "ェ": ("e", ),
    "ォ": ("o", ),

    "ヮ": ("wa", ),

    "ヷ": ("V", "a"),
    "ヸ": ("V", "i"),
    "ヴ": ("V", "u"),
    "ヹ": ("V", "e"),
    "ヺ": ("V", "o"),

    "ヵ": ("k", "a"),  # should preprocessed
    "ヶ": ("k", "e"),  # should preprocessed
    "ッ": ("tsu", ),

    "ン": ("NN", ),
}

# dict
numbering_dict = {
    "1": "いち",
    "2": "に",
    "3": "さん",
    "4": "し",
    "5": "ご",
    "6": "ろく",
    "7": "しち",
    "8": "はち",
    "9": "きゅう",
    "0": "ぜろ",
    # "10": "じゅう",
    # "100": "ひゃく",
    # "1000": "せん",
}

# user word dict
# for mecab
_user_dict_prev = {
    "玉璽": "ぎょくじ",
    "奉り候": "たてまつりそうろう",
    "内頸": "ないけい",
    "珂是古": "かぜこ",
    "中隔": "ちゅうがく",
    "冥熏": "めいくん",
    "非可換": "ひかかん",
    "立替払": "たてかえばらい",
    "下賤": "げせん",
    "天塩炭礦鉄道": "てしおたんこうてつどう",
    "葉身": "ようしん",
    "心形": "しんけい",
    "披針形": "ひしんけい",
    "仙醸": "せんじょう",
    "側弯": "そくわん",
    "叩き": "たたき",
    "回転斬": "かいてんぎり",
    "一閃斬": "いっせんぎり",
    "志々雄": "ししお",
    "逓伝哨": "ていでんしょう",
    "逓騎哨": "ていきしょう",
    "逓自転車哨": "ていじてんしゃしょう",
    "音撃": "おんげき",
    "狡噛": "こうがみ",
    "軟木": "なんぼく",
    "雹": "ひょう",
    "刺胞": "しほう",
    "胞子嚢": "ほうしのう",
    "掲記": "けいき",
    "杯細胞様": "さかずきさいぼうよう",
    "果胞": "かほう",
    "後掲": "ごけい",
    "剝す": "はがす",
    "剝い": "はい",
    "剝げ": "はげ",
    "剝が": "はが",
}
_user_dict_post = {
    "圧痛": "あっつう",
    "頸部": "けいぶ",
    "杳":   "よう",
    "剝製":  "はくせい",
    "爛火":  "らんか",
    "溶結": "ようけつ",
    "療":   "りょう",
    "扈":   "こ",
    "四川省": "しせんしょう",
    "雅安県": "があんけん",
    "高頤墓闕": "こういぼけつ",
    "灌漑": "がんがい",
    "橈骨": "とうこつ",
    "龐煖": "ほうけん",
    "闘蛇": "とうじゃ",
    "受器": "うけざら",
    "弐章": "にしょう",
    "胥吏": "かんり",
    "靄": "もや",
    "熔": "と",
    "蟲": "むし",
    "聚慎": "じゅしん",
    "腔": "くう",

    "A": "えい",
    "H": "えいち",
}


def is_allowed_symbol(char):
    return char in symbols


def translate_to_symbols(text):
    result = []
    for i, char in enumerate(text):
        if char in _translate_table:
            if char not in _katakana_composite_vowels:
                result += _translate_table[char]
            else:
                # convert composit vowel
                last_symbol = result[-1]
                last_char = text[i - 1]
                if last_symbol not in _sym_basic_vowels or last_char in _katakana_composite_vowels:
                    result += _translate_table[char]
                else:
                    result = result[:-1] + [last_symbol + _translate_table[char][0]]
        elif is_allowed_symbol(char):
            result.append(char)
        else:
            print(f"Unknown char({char}) in text.")
    return result


def normalize_to_katakana(text):
    for c in [" ", "　", "「", "」", "『", "』", "・", "【", "】", "（", "）", "(", ")"]:
        text = text.replace(c, "")
    text = text.replace("!", "！")
    text = text.replace("?", "？")

    text = _normalize_delimitor(text)
    text = jaconv.normalize(text)
    text = _convert_numbers(text)
    text = _convert_user_dict(text, _user_dict_prev)
    text = _mix_pronunciation(text)
    # text = kakasi_convert(text)
    text = _convert_user_dict(text, _user_dict_post)
    text = jaconv.hira2kata(text)
    text = _add_punctuation(text)
    return text


def _get_tagger():
    global _tagger
    if not _tagger:
        _tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    return _tagger


def _yomi(mecab_result):
    tokens = []
    yomis = []
    for line in mecab_result.split("\n")[:-1]:
        s = line.split("\t")
        if len(s) == 1:
            break
        token, rest = s
        rest = rest.split(",")
        tokens.append(token)
        yomi = rest[7] if len(rest) > 7 else None
        yomi = None if yomi == "*" else yomi
        yomis.append(yomi)
    return tokens, yomis


def _mix_pronunciation(text):
    tokens, yomis = _yomi(_get_tagger().parse(text))
    return "".join(
        yomis[idx] if yomis[idx] is not None else tokens[idx]
        for idx in range(len(tokens)))


def _add_punctuation(text):
    last = text[-1]
    if last not in [".", ",", "、", "。", "！", "？", "!", "?"]:
        text = text + "。"
    return text


def _normalize_delimitor(text):
    text = text.replace(",", "、")
    text = text.replace(".", "。")
    text = text.replace("，", "、")
    text = text.replace("．", "。")
    return text


def _convert_user_dict(text, user_dict):
    for word, furigana in user_dict.items():
        if word in text:
            # print(f"User replace '{word}' to '{furigana}' in {text}, ret={text.replace(word, furigana)}")
            text = text.replace(word, furigana)
    return text


def _convert_numbers(text):
    text = _convert_phone_number(text)
    text = _convert_numbering(text)
    text = _convert_number_to_kanji(text, _number_pattern)
    return text


def _convert_phone_number(text):
    return _convert_number_seq(text, _phone_pattern, ['-', 'ー'])


def _convert_numbering(text):
    return _convert_number_seq(text, _numbering_pattern)


def _convert_number_seq(text, pattern, delimiters=[]):
    match = pattern.search(text)
    while match is not None:
        pronunce = []
        numbers = match.group(1)
        for digit in list(numbers):
            if digit in numbering_dict:
                pronunce.append(numbering_dict[digit])
            elif digit in delimiters:
                pronunce.append(digit)
            else:
                print(f"unknown char: {digit} in {text}")
        text = text.replace(numbers, ''.join(pronunce))
        match = pattern.search(text)
    return text


def _convert_number_to_kanji(text, pattern):
    match = pattern.search(text)
    while match is not None:
        numbers = match.group(1)
        text = text.replace(numbers, suji.kansuji(numbers, False))
        match = pattern.search(text)
    return text
